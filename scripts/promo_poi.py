"""CPU projection of OpenSCAD geometry, CFD textures and a conceptual POI mosaic.

OpenSCAD builds the actual meshes used in every frame; Graphviz lays out the
inference route. LF color values are native CFD samples. The teal output is an
explicitly synthetic workflow illustration, never the packaged HF reference.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
import subprocess

import cairosvg
from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cfd_io import sha
from replot_sod_promo import memory_gate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/promo"
SRC = ROOT / "scripts/figure_tools"
WIDTH, HEIGHT, FRAME_COUNT = 1200, 750, 132
WINDOW = (.30, .50, .30, .50)
PATCH_SIDE = .04
SHARP, SMOOTH = (.34, .44), (.46, .48)
INK, MUTED = "#162536", "#596879"
BLUE, GREEN, TEAL = "#2563eb", "#159947", "#087f8c"
GRID_SIZE = 9


def read_off(path):
    lines = [s.strip() for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")]
    header = lines.pop(0).split()
    if header[0] != "OFF":
        raise ValueError(f"Not an OFF mesh: {path}")
    nv, nf, _ = map(int, header[1:] or lines.pop(0).split())
    vertices = np.array([list(map(float, s.split())) for s in lines[:nv]])
    faces = [list(map(int, s.split()))[1:] for s in lines[nv:nv+nf]]
    if vertices.shape != (nv, 3) or not np.isfinite(vertices).all() or not faces:
        raise ValueError(f"Invalid OpenSCAD mesh: {path}")
    if any(min(f) < 0 or max(f) >= nv or len(f) < 3 for f in faces):
        raise ValueError(f"Invalid face indices: {path}")
    return vertices, faces


def prepare_tools():
    """Execute production tools sequentially; keep small, reviewable exports."""
    folder = OUT / "poi_geometry"
    folder.mkdir(exist_ok=True, parents=True)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", LIBGL_ALWAYS_SOFTWARE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    commands = []

    def run(command):
        available = memory_gate()
        if available < 700:
            raise RuntimeError("CAD export deferred below 700 MiB effective available RAM")
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        commands.append({"command": command, "exit_code": result.returncode,
                         "memory_before_mib": available})

    for part in ("plane", "patch", "sample", "pin"):
        path = folder / f"{part}.off"
        run(["openscad", "--hardwarnings", "-o", str(path.relative_to(ROOT)),
             "-D", f'part="{part}"', str((SRC / "poi_planes.scad").relative_to(ROOT))])
        # OpenSCAD leaves trailing spaces on vertex rows; normalize text only.
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
        read_off(path)
    route = OUT / "poi_pipeline.svg"
    run(["dot", "-Tsvg", str((SRC / "poi_pipeline.dot").relative_to(ROOT)),
         "-o", str(route.relative_to(ROOT))])
    return {
        "execution": "OpenSCAD OFF meshes + Graphviz SVG, used directly by CPU compositor; no GPU",
        "commands": commands,
        "versions": {tool: subprocess.run([tool, flag], capture_output=True, text=True, check=True).stderr.strip()
                     for tool, flag in (("openscad", "--version"), ("dot", "-V"))},
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in
                    (SRC / "poi_planes.scad", SRC / "poi_pipeline.dot")},
        "exports": {str(path.relative_to(OUT)): sha(path) for path in
                    [*(folder / f"{p}.off" for p in ("plane", "patch", "sample", "pin")), route]},
    }


def ease(value):
    value = max(0., min(1., value))
    return value * value * (3 - 2 * value)


@dataclass(frozen=True)
class State:
    point: tuple[float, float]
    lift: float
    tilt: float
    stage: int
    filled: int


def state_at(frame):
    if frame < 30:
        return State(SHARP, 29*(1-ease(frame/13)), 52, 0, 0)
    if frame < 58:
        return State(SMOOTH, 29*(1-ease((frame-30)/12)), 52, 1, 0)
    if frame < 80:
        return State(SMOOTH, 0, 52*(1-ease((frame-58)/21)), 2, 0)
    filled = min(GRID_SIZE**2, math.floor((frame-79)/36*GRID_SIZE**2))
    index = max(0, filled-1)
    row, col = divmod(index, GRID_SIZE)
    # Reuse a common physical POI across both native grids. Every target lies
    # on an LF1 node, so the closed .04-square patches have 5x5 / 9x9 nodes.
    point = (.32+.02*col, .48-.02*row)
    return State(point, 0, 0, 3, filled)


def native_patch(field, point):
    x, y = field[0, :, 0], field[:, 0, 1]
    half = PATCH_SIDE/2
    ix = np.flatnonzero((x >= point[0]-half-1e-7) & (x <= point[0]+half+1e-7))
    iy = np.flatnonzero((y >= point[1]-half-1e-7) & (y <= point[1]+half+1e-7))
    return x[ix], y[iy], field[:, :, 2][np.ix_(iy, ix)]


@lru_cache(maxsize=64)
def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


class Scene:
    def __init__(self, fields, norm):
        self.fields, self.norm = fields[:2], norm
        self.meshes = {name: read_off(OUT / "poi_geometry" / f"{name}.off")
                       for name in ("plane", "patch", "sample", "pin")}
        # Actual native cell footprints, clipped to the shared physical crop.
        # Precompute only static colors and geometry, never an RGB frame stack.
        self.cells = []
        for field in self.fields:
            n = field.shape[0]-1
            cells = []
            for j in range(round(WINDOW[2]*n), round(WINDOW[3]*n)+1):
                for i in range(round(WINDOW[0]*n), round(WINDOW[1]*n)+1):
                    x0, x1 = max(.30, (i-.5)/n), min(.50, (i+.5)/n)
                    y0, y1 = max(.30, (j-.5)/n), min(.50, (j+.5)/n)
                    xyz = np.array([[500*(x-.4), 500*(y-.4), .02]
                                    for x, y in ((x0,y0), (x1,y0), (x1,y1), (x0,y1))])
                    color = tuple(int(v*255) for v in colormaps["inferno"](norm(field[j, i, 2]))[:3])
                    cells.append((xyz, color))
            self.cells.append(cells)
        import io
        self.route = Image.open(io.BytesIO(cairosvg.svg2png(url=str(OUT / "poi_pipeline.svg"), output_width=1080))).convert("RGBA")

    def project(self, vertices, center, tilt):
        theta, yaw = math.radians(tilt), math.radians(-12*tilt/52)
        x, y, z = vertices.T
        xx, yy = x*math.cos(yaw)-y*math.sin(yaw), x*math.sin(yaw)+y*math.cos(yaw)
        points = np.column_stack((center[0]+2.5*xx, center[1]-2.5*(yy*math.cos(theta)+z*math.sin(theta))))
        depth = -yy*math.sin(theta)+z*math.cos(theta)
        return points, depth

    def mesh(self, d, name, offset, center, tilt, color):
        vertices, faces = self.meshes[name]
        xyz = vertices + offset
        projected, depth = self.project(xyz, center, tilt)
        rgb = np.array(hex_rgb(color))
        for face in sorted(faces, key=lambda f: float(depth[f].mean())):
            points = xyz[face]
            normal = np.cross(points[1]-points[0], points[2]-points[0])
            length = np.linalg.norm(normal)
            brightness = .78+.22*max(0., np.dot(normal/length, [-.25, -.35, .902])) if length else 1
            d.polygon([tuple(v) for v in projected[face]], fill=tuple((rgb*brightness).astype(int)))

    def frame(self, number):
        state = state_at(number)
        im = Image.new("RGB", (WIDTH, HEIGHT), "white")
        d = ImageDraw.Draw(im)

        def text(x, y, value, size=18, color=INK, bold=False, anchor="lt"):
            d.text((x, y), value, font=font(size, bold), fill=color, anchor=anchor)

        text(34, 24, "Same space. More local information.", 31, INK, True)
        text(1166, 34, "NNLCI / LOCAL TO GLOBAL", 14, TEAL, True, "rt")
        stages = ("01  Sharp transition", "02  Smooth region", "03  Turn to 2D", "04  Assemble POIs")
        for i, label in enumerate(stages):
            x = 34+292*i
            active = state.stage == i
            d.rounded_rectangle((x, 76, x+267, 111), radius=8, fill="#e8f5f3" if active else "#f5f7fa")
            text(x+12, 85, label, 16, TEAL if active else MUTED, active)
        d.line([(779, 137), (779, 612)], fill="#dfe7ee", width=1)

        for index, (field, center, color) in enumerate(zip(self.fields, ((208,300), (568,300)), (BLUE,GREEN))):
            x = center[0]-125
            text(x, 139, f"LF{index+1}", 27, color, True)
            text(x+80, 147, f"{100*(index+1)}² intervals", 16, MUTED)
            self.mesh(d, "plane", (0,0,0), center, state.tilt, "#b8c8d5")
            for vertices, tint in self.cells[index]:
                points, _ = self.project(vertices, center, state.tilt)
                d.polygon([tuple(v) for v in points], fill=tint)
            px, py = ((value-.4)*500 for value in state.point)
            self.mesh(d, "patch", (px,py,0), center, state.tilt, "#ffffff")
            xx, yy, values = native_patch(field, state.point)
            for sy in yy:
                for sx in xx:
                    self.mesh(d, "sample", ((sx-.4)*500,(sy-.4)*500,.5), center, state.tilt, "#ffffff")
            self.mesh(d, "pin", (px,py,state.lift+1.2), center, state.tilt, color)
            # Enlarged patch: same physical support and display bounds, native
            # sample locations only. Density is nearest-node, never smoothed.
            left, top, side = center[0]-68, 474, 136
            half = PATCH_SIDE/2
            for j, sy in enumerate(yy):
                for i, sx in enumerate(xx):
                    n = field.shape[0]-1
                    xa, xb = max(state.point[0]-half,sx-.5/n), min(state.point[0]+half,sx+.5/n)
                    ya, yb = max(state.point[1]-half,sy-.5/n), min(state.point[1]+half,sy+.5/n)
                    box = (left+(xa-state.point[0]+half)/PATCH_SIDE*side,
                           top+(state.point[1]+half-yb)/PATCH_SIDE*side,
                           left+(xb-state.point[0]+half)/PATCH_SIDE*side,
                           top+(state.point[1]+half-ya)/PATCH_SIDE*side)
                    tint = tuple(int(v*255) for v in colormaps["inferno"](self.norm(values[j,i]))[:3])
                    d.rectangle(box, fill=tint)
            d.rectangle((left,top,left+side,top+side), outline=color, width=2)
            for sy in yy:
                for sx in xx:
                    dx = left+(sx-state.point[0]+half)/PATCH_SIDE*side
                    dy = top+(state.point[1]+half-sy)/PATCH_SIDE*side
                    # Outline boundary nodes too, so all 25 / 81 are visible.
                    d.ellipse((dx-2.3,dy-2.3,dx+2.3,dy+2.3), fill="white", outline="#596879", width=1)
            text(center[0], 452, f"{len(xx)} × {len(yy)} native nodes", 17, color, True, "mt")
            text(center[0]+85, 507, "Same", 16, MUTED)
            text(center[0]+85, 531, "0.04 × 0.04", 15, MUTED)
            text(center[0]+85, 555, "patch", 16, MUTED)
        text(388, 427, f"Shared POI  ({state.point[0]:.2f}, {state.point[1]:.2f})", 15, MUTED, anchor="mt")

        text(821, 145, "HF-like prediction", 25, TEAL, True)
        text(821, 178, "Schematic · illustrative colors", 16, MUTED)
        side, left, top = 288, 826, 230
        tile = side/GRID_SIZE
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rank = row*GRID_SIZE+col
                u, v = (col+.5)/GRID_SIZE, 1-(row+.5)/GRID_SIZE
                # Deliberately synthetic teal motif. No HF or model values enter
                # this panel; a sharp interface and a smooth bump explain assembly.
                value = .16+.32*math.exp(-9*((u-.68)**2+(v-.7)**2))+.42*(v > .20+.50*u)
                base, light = np.array([8,112,125]), np.array([219,244,230])
                tint = tuple((base*(1-value)+light*value).astype(int)) if rank < state.filled else "#f1f6f7"
                xa, ya = left+col*tile, top+row*tile
                if rank < state.filled and rank >= state.filled-3 and number < 116:
                    # New groups visibly settle into their 2D puzzle position.
                    ya -= 7
                d.rounded_rectangle((xa+1,ya+1,xa+tile-1,ya+tile-1), radius=3, fill=tint)
        text(970, 538, "Point predictions → field", 20, TEAL, True, "mt")
        text(970, 570, "Tiles group many target points", 15, MUTED, anchor="mt")
        text(970, 596, "No model output shown", 15, MUTED, anchor="mt")
        # Actual Graphviz export forms the workflow strip in the shipped frames.
        im.paste(self.route, ((WIDTH-self.route.width)//2, 633), self.route)
        if state.stage == 3 and number < 116:
            # A pulse follows the graph's inference direction while new point
            # groups settle into the mosaic; it carries no numerical value.
            pulse = (number-80) % 9 / 8
            x, y = 85+1030*pulse, 714
            d.line([(85,y),(1115,y)], fill="#e2efed", width=2)
            d.ellipse((x-4,y-4,x+4,y+4), fill=TEAL)
        text(34, 732, "LF density: independent CFD · common crop & scale", 13, MUTED)
        text(1166, 732, "Workflow motion · fixed physical time", 13, MUTED, anchor="rt")
        return im


def hex_rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1,3,5))


def render_static(scene):
    # First-frame poster includes an airborne POI and both equal-sized patches.
    scene.frame(0).save(OUT / "nnlci_motion_poster.png", optimize=True)
    # Permanent storyboard retains landing, camera transition and completed field.
    sheet = Image.new("RGB", (WIDTH, HEIGHT*3), "white")
    for index, frame in enumerate((18, 79, 131)):
        sheet.paste(scene.frame(frame), (0, index*HEIGHT))
    sheet.save(OUT / "nnlci_local_to_hf_concept.png", optimize=True)
    return ["nnlci_motion_poster.png", "nnlci_local_to_hf_concept.png"]


def animate(scene):
    # Palette samples include both actual LF textures and the synthetic output.
    palette = scene.frame(131).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = []
    for index in range(FRAME_COUNT):
        if index % 24 == 0:
            print(f"POI animation: {index}/{FRAME_COUNT}", flush=True)
        frames.append(scene.frame(index).quantize(palette=palette, dither=Image.Dither.NONE))
    durations = [90]*FRAME_COUNT
    durations[0], durations[29], durations[57], durations[79], durations[-1] = 600, 700, 700, 600, 1800
    frames[0].save(OUT / "nnlci_local_to_hf.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=1)
    # Pillow may coalesce identical holds; record the decoded deliverable.
    with Image.open(OUT / "nnlci_local_to_hf.gif") as gif:
        actual_frames, total_ms = gif.n_frames, 0
        for i in range(actual_frames):
            gif.seek(i)
            total_ms += gif.info["duration"]
    return {"file": "nnlci_local_to_hf.gif", "frames": actual_frames, "rendered_frames": FRAME_COUNT,
            "dimensions": [WIDTH, HEIGHT], "duration_ms": total_ms,
            "meaning": "Shared POI drops onto equal LF patches, visits sharp/smooth regions, rotates to 2D, then assembles schematic point groups; no physical evolution or inferred field"}


def semantics():
    return {
        "patch": {"physical_side": PATCH_SIDE, "lf1_nodes": [5,5], "lf2_nodes": [9,9],
                  "interpretation": "Closed native-grid patches on common LF1 nodes; not network feature counts"},
        "pois": {"sharp_transition": list(SHARP), "smooth_region": list(SMOOTH),
                 "selection": "Fixed illustrative coordinates in LF fields; no HF-dependent selection"},
        "model_contract": "Two aligned 3x3 stencils, four variables each, 72 scalar features; unchanged",
        "prediction_mosaic": {"kind": "Synthetic workflow illustration, not NNLCI output or HF reference",
                              "expression": "0.16 + 0.32*exp(-9*((u-0.68)^2+(v-0.7)^2)) + 0.42*(v>0.20+0.50*u)",
                              "display": "Teal schematic; 9x9 tiles group many target-point predictions; no physical grid-resolution claim"},
        "camera": "Orthographic; both planes share geometry, projection, crop and scale; tilt 52 to 0 degrees",
    }
