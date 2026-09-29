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
WIDTH, HEIGHT = 1200, 840
WINDOW = (.30, .50, .30, .50)
PATCH_SIDE = .04
SHARP, SMOOTH = (.34, .44), (.46, .48)
INK, MUTED = "#162536", "#596879"
BLUE, GREEN, TEAL = "#2563eb", "#159947", "#087f8c"
GRID_SIZE = 9
INTRO_FRAMES, FRAMES_PER_POINT = 64, 4
ASSEMBLY_END = INTRO_FRAMES + FRAMES_PER_POINT * GRID_SIZE**2
FRAME_COUNT = ASSEMBLY_END + 12


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
    active: int = 0
    phase: int = 0


def state_at(frame):
    if frame < 22:
        return State(SHARP, 29*(1-ease(frame/13)), 52, 0, 0)
    if frame < 42:
        return State(SMOOTH, 29*(1-ease((frame-22)/12)), 52, 1, 0)
    if frame < INTRO_FRAMES:
        return State(SMOOTH, 0, 52*(1-ease((frame-42)/21)), 2, 0)
    index = min(GRID_SIZE**2-1, (frame-INTRO_FRAMES)//FRAMES_PER_POINT)
    phase = (frame-INTRO_FRAMES) % FRAMES_PER_POINT if frame < ASSEMBLY_END else 3
    filled = index + (phase == 3)
    row, col = divmod(index, GRID_SIZE)
    # One coordinate, one prediction, one display cell. No grouped/skipped POIs.
    point = (.32+.02*col, .48-.02*row)
    return State(point, 0, 0, 3, filled, index, phase)


def native_patch(field, point):
    x, y = field[0, :, 0], field[:, 0, 1]
    half = PATCH_SIDE/2
    ix = np.flatnonzero((x >= point[0]-half-1e-7) & (x <= point[0]+half+1e-7))
    iy = np.flatnonzero((y >= point[1]-half-1e-7) & (y <= point[1]+half+1e-7))
    return x[ix], y[iy], field[:, :, 2][np.ix_(iy, ix)]


def stencil(point):
    """Nine aligned feature locations, independent of native mesh resolution.

    The outer .04 square is context; the stencil spans .02 in both directions.
    Display coordinates illustrate alignment, not the bootstrap's exact
    upsampling/dilation implementation. Both LF sources supply nine values.
    """
    return np.array([(point[0]+i*.01, point[1]+j*.01)
                     for j in (-1, 0, 1) for i in (-1, 0, 1)])


def prediction_color(index):
    """Illustrative scalar color at one POI; never HF data or MLP inference."""
    row, col = divmod(index, GRID_SIZE)
    u, v = (col+.5)/GRID_SIZE, 1-(row+.5)/GRID_SIZE
    value = .16+.32*math.exp(-9*((u-.68)**2+(v-.7)**2))+.42*(v > .20+.50*u)
    base, light = np.array([8,112,125]), np.array([219,244,230])
    return tuple((base*(1-value)+light*value).astype(int))


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

        def arrow(points, color=TEAL, width=2):
            d.line(points, fill=color, width=width, joint="curve")
            x, y = points[-1]
            angle = math.atan2(y-points[-2][1], x-points[-2][0])
            d.polygon([(x,y), (x-10*math.cos(angle)+5*math.sin(angle), y-10*math.sin(angle)-5*math.cos(angle)),
                       (x-10*math.cos(angle)-5*math.sin(angle), y-10*math.sin(angle)+5*math.cos(angle))], fill=color)

        text(34, 23, "Two 3 × 3 patches. One point at a time.", 31, INK, True)
        text(34, 65, "LF2: finer source mesh, the same nine input locations.", 20, MUTED)
        text(1166, 32, "NNLCI", 16, TEAL, True, "rt")
        steps = ("01  Sample 3 × 3", "02  MLP", "03  One POI", "04  Place & repeat")
        active_step = 0 if state.stage < 3 else (0, 1, 2, 3)[state.phase]
        for i, label in enumerate(steps):
            x = 34+292*i
            active = active_step == i
            d.rounded_rectangle((x, 105, x+267, 140), radius=8, fill="#e8f5f3" if active else "#f5f7fa")
            text(x+12, 114, label, 16, TEAL if active else MUTED, active)
        d.line([(674, 162), (674, 666)], fill="#dfe7ee", width=1)

        for index, (field, center, color) in enumerate(zip(self.fields, ((177,320), (482,320)), (BLUE,GREEN))):
            x = center[0]-125
            text(x, 167, f"LF{index+1}", 27, color, True)
            text(x+74, 175, f"{field.shape[0]-1}² mesh", 17, MUTED)
            self.mesh(d, "plane", (0,0,0), center, state.tilt, "#b8c8d5")
            for vertices, tint in self.cells[index]:
                points, _ = self.project(vertices, center, state.tilt)
                d.polygon([tuple(v) for v in points], fill=tint)
            px, py = ((value-.4)*500 for value in state.point)
            self.mesh(d, "patch", (px,py,0), center, state.tilt, "#ffffff")
            # Exactly nine highlighted inputs on BOTH LF planes. Finer native
            # texture remains context and never becomes extra feature markers.
            for sx, sy in stencil(state.point):
                self.mesh(d, "sample", ((sx-.4)*500,(sy-.4)*500,.5), center, state.tilt, "#ffffff")
            self.mesh(d, "pin", (px,py,state.lift+1.2), center, state.tilt, color)
            xx, yy, values = native_patch(field, state.point)
            left, top, side = center[0]-68, 492, 136
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
            for sx, sy in stencil(state.point):
                dx = left+(sx-state.point[0]+half)/PATCH_SIDE*side
                dy = top+(state.point[1]+half-sy)/PATCH_SIDE*side
                d.ellipse((dx-4,dy-4,dx+4,dy+4), fill="white", outline=INK, width=1)
            # The center is one of the nine samples, not a tenth input.
            d.ellipse((center[0]-8,top+side/2-8,center[0]+8,top+side/2+8), outline="white", width=2)
            text(center[0], 465, "3 × 3 inputs", 21, color, True, "mt")
            text(center[0], 639, "9 locations · 4 variables", 16, MUTED, anchor="mt")
        region = ("Sharp transition", "Smooth region", "Turn to 2D", "Scan target locations")[state.stage]
        text(329, 681, f"{region} · POI ({state.point[0]:.2f}, {state.point[1]:.2f})", 16, MUTED, anchor="mt")

        # Both nine-point inputs enter a visible MLP, which emits ONE spatial
        # prediction. Four components at that point are not four output cells.
        arrow([(252,564),(293,564),(293,670),(650,670),(650,263),(704,263)], BLUE, 2)
        arrow([(559,564),(650,564),(650,263),(704,263)], GREEN, 2)
        model_active = state.stage == 3 and state.phase == 0
        d.rounded_rectangle((716,209,931,301), radius=12,
                            fill="#d8efeb" if model_active else "#eaf5f5", outline=TEAL, width=3 if model_active else 1)
        text(823, 227, "NNLCI MLP", 26, TEAL, True, "mt")
        text(823, 268, "72 inputs → 4 outputs", 16, TEAL, anchor="mt")
        arrow([(940,254),(1048,254)])
        text(1081, 167, "One POI", 24, TEAL, True, "mt")
        text(1081, 205, "ρ̂, û, v̂, p̂", 20, MUTED, anchor="mt")
        color = prediction_color(state.active)
        show_value = state.stage == 3 and state.phase >= 1
        d.rectangle((1065,238,1097,270), fill=color if show_value else "#eff5f5", outline=TEAL, width=2)
        d.ellipse((1078,251,1084,257), fill="white" if show_value else TEAL)
        text(1081, 289, "Only this point", 16, TEAL, anchor="mt")

        text(714, 331, "HF-like reconstruction puzzle", 25, TEAL, True)
        text(714, 365, "Schematic · illustrative colors", 16, MUTED)
        side, left, top = 288, 826, 404
        tile = side/GRID_SIZE
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rank = row*GRID_SIZE+col
                xa, ya = left+col*tile, top+row*tile
                tint = prediction_color(rank) if rank < state.filled else "#f1f6f7"
                d.rectangle((xa+1,ya+1,xa+tile-1,ya+tile-1), fill=tint)
                # A dot is the POI; its small grid square is just a display cell.
                if rank < state.filled:
                    d.ellipse((xa+tile/2-1.5,ya+tile/2-1.5,xa+tile/2+1.5,ya+tile/2+1.5), fill="#eefaf7")
        row, col = divmod(state.active, GRID_SIZE)
        tx, ty = left+(col+.5)*tile, top+(row+.5)*tile
        if state.stage == 3:
            d.rectangle((tx-tile/2,ty-tile/2,tx+tile/2,ty+tile/2), outline=TEAL, width=3)
            if state.phase in (1,2):
                # Visible correspondence: the same output cell travels to its
                # coordinate; no patch-sized block is predicted or placed.
                arrow([(1110,254),(1152,254),(1152,ty),(tx+tile/2+4,ty)], "#8ebeba")
            if state.phase == 2:
                fx, fy = 1152, (254+ty)/2
                d.rectangle((fx-15,fy-15,fx+15,fy+15), fill=color, outline=TEAL, width=2)
                d.ellipse((fx-2,fy-2,fx+2,fy+2), fill="white")
        text(714, 430, "1 cell", 20, TEAL, True)
        text(714, 460, "= 1 POI", 18, TEAL)
        text(714, 522, f"{state.filled} / 81", 21, TEAL, True)
        text(714, 554, "placed", 16, MUTED)
        text(714, 611, "Repeat", 18, TEAL, True)
        text(714, 640, "at every x", 16, MUTED)
        if state.filled == GRID_SIZE**2:
            d.rectangle((left-2,top-2,left+side+2,top+side+2), outline=TEAL, width=2)
        # Actual Graphviz output, not a hand-drawn substitute.
        im.paste(self.route, ((WIDTH-self.route.width)//2, 725), self.route)
        text(34, 822, "LF density: independent CFD · fixed time · same crop & scale", 12, MUTED)
        text(1166, 822, "Puzzle: workflow illustration, no model output", 12, MUTED, anchor="rt")
        return im


def hex_rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1,3,5))


def render_static(scene):
    # A partially assembled puzzle keeps the pointwise mechanism visible even
    # if motion is disabled. The first GIF frame uses this same opening image.
    scene.frame(INTRO_FRAMES + 40*FRAMES_PER_POINT + 2).save(OUT / "nnlci_motion_poster.png", optimize=True)
    # Permanent storyboard: fixed-size inputs, one point in flight, full field.
    sheet = Image.new("RGB", (WIDTH, HEIGHT*3), "white")
    for index, frame in enumerate((18, INTRO_FRAMES + 40*FRAMES_PER_POINT + 2, FRAME_COUNT-1)):
        sheet.paste(scene.frame(frame), (0, index*HEIGHT))
    sheet.save(OUT / "nnlci_local_to_hf_concept.png", optimize=True)
    return ["nnlci_motion_poster.png", "nnlci_local_to_hf_concept.png"]


def animate(scene):
    available = memory_gate()
    if available < 1500:
        raise RuntimeError("Animation deferred below 1500 MiB effective available RAM")
    # Palette samples include both actual LF textures and the synthetic output.
    palette = scene.frame(FRAME_COUNT-1).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [scene.frame(INTRO_FRAMES + 40*FRAMES_PER_POINT + 2).quantize(palette=palette, dither=Image.Dither.NONE)]
    for index in range(FRAME_COUNT):
        if index % 24 == 0:
            print(f"POI animation: {index}/{FRAME_COUNT}", flush=True)
        frames.append(scene.frame(index).quantize(palette=palette, dither=Image.Dither.NONE))
    durations = [1200] + [60]*FRAME_COUNT
    for index in (0, 21, 41, 63):
        durations[index+1] = 400
    # Slow the first two complete passes so sample / predict / move / place
    # can be followed; every later point still gets all four phases.
    for index in range(INTRO_FRAMES, INTRO_FRAMES+2*FRAMES_PER_POINT):
        durations[index+1] = 350
    durations[-1] = 1600
    frames[0].save(OUT / "nnlci_local_to_hf.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=1)
    # Pillow may coalesce identical holds; record the decoded deliverable.
    with Image.open(OUT / "nnlci_local_to_hf.gif") as gif:
        actual_frames, total_ms = gif.n_frames, 0
        for i in range(actual_frames):
            gif.seek(i)
            total_ms += gif.info["duration"]
    return {"file": "nnlci_local_to_hf.gif", "frames": actual_frames, "rendered_frames": FRAME_COUNT+1,
            "dimensions": [WIDTH, HEIGHT], "duration_ms": total_ms,
            "meaning": "Two nine-point LF stencils feed the MLP; one output POI moves into one puzzle cell. All 81 POIs are shown separately; no physical evolution or inferred field",
            "memory_before_mib": available}


def semantics():
    return {
        "patch": {"context_physical_side": PATCH_SIDE, "stencil_physical_span": .02,
                  "lf1_input_locations": [3,3], "lf2_input_locations": [3,3],
                  "native_mesh_spacing": {"lf1": .01, "lf2": .005},
                  "interpretation": "Exactly nine white feature markers per LF input at aligned physical positions. Finer LF2 texture is background context, not more features.",
                  "display_contract": "Illustrative common stencil coordinates; the packaged bootstrap instead upsamples both LF arrays and extracts dilation-4 3x3 stencils. No preprocessing or input contract changes."},
        "pois": {"sharp_transition": list(SHARP), "smooth_region": list(SMOOTH),
                 "selection": "Fixed illustrative coordinates in LF fields; no HF-dependent selection"},
        "model_contract": "Two aligned 3x3 stencils, four variables each, 72 scalar features; unchanged",
        "prediction_mosaic": {"kind": "Synthetic workflow illustration, not NNLCI output or HF reference",
                              "expression": "0.16 + 0.32*exp(-9*((u-0.68)^2+(v-0.7)^2)) + 0.42*(v>0.20+0.50*u)",
                              "display": "9x9 illustrative target grid; one display cell represents exactly one POI, with four output variables at that location. No physical HF-resolution claim.",
                              "assembly": "81 unique coordinates in row-major order, four phases each: sample/MLP, one output, travel, place. Each placement adds exactly one cell; existing cells remain fixed.",
                              "preview": "First GIF frame previews partial reconstruction; then the complete empty-to-full sequence starts."},
        "camera": "Orthographic; both planes share geometry, projection, crop and scale; tilt 52 to 0 degrees",
    }
