"""CPU projection of OpenSCAD geometry and a dense, accelerated POI assembly.

OpenSCAD builds the meshes used in the sampling introduction; Graphviz lays out the
inference route. LF color values are native CFD samples. Assembly uses the
packaged HF reference as a labeled visual stand-in, never as model predictions.
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
GRID_SIZE = 801  # Packaged 800-interval grid includes both domain endpoints.
POINT_COUNT = GRID_SIZE**2
INTRO_FRAMES, SWEEP_FRAMES = 44, 88
# First show the empty domain and the upper boundary, then skip temporal steps.
# Each interval represents every row-major POI between two displayed counts.
ASSEMBLY_COUNTS = (0, 1, GRID_SIZE) + tuple(
    GRID_SIZE + (POINT_COUNT-GRID_SIZE)*k//SWEEP_FRAMES
    for k in range(1, SWEEP_FRAMES+1))
ASSEMBLY_END = INTRO_FRAMES + len(ASSEMBLY_COUNTS)
FRAME_COUNT = ASSEMBLY_END + 1
TARGET_BOX = (724, 246, 430)


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


def point_at(index):
    """Row-major order in display space, including all four boundaries."""
    row, col = divmod(index, GRID_SIZE)
    return col/(GRID_SIZE-1), 1-row/(GRID_SIZE-1)


def complete_stencil(point):
    samples = stencil(point)
    return bool(np.all((samples >= 0) & (samples <= 1)))


def state_at(frame):
    if not 0 <= frame < FRAME_COUNT:
        raise ValueError(f"Frame outside animation: {frame}")
    if frame < 14:
        return State(SHARP, 29*(1-ease(frame/10)), 52, 0, 0)
    if frame < 28:
        return State(SMOOTH, 29*(1-ease((frame-14)/10)), 52, 1, 0)
    if frame < INTRO_FRAMES:
        return State(SMOOTH, 0, 52*(1-ease((frame-28)/15)), 2, 0)
    filled = ASSEMBLY_COUNTS[min(frame-INTRO_FRAMES, len(ASSEMBLY_COUNTS)-1)]
    active = max(0, filled-1)
    return State(point_at(active), 0, 0, 4 if frame == ASSEMBLY_END else 3,
                 filled, active, 3)


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


@lru_cache(maxsize=64)
def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


class Scene:
    def __init__(self, fields, norm):
        self.fields, self.norm = fields[:2], norm
        self.hf = fields[2]
        if self.hf.shape != (GRID_SIZE, GRID_SIZE, 6):
            raise ValueError("Assembly requires the complete 801 × 801 packaged field")
        # Native HF colors, with y increasing upward. This is a REFERENCE reveal.
        self.reference = Image.fromarray(colormaps["inferno"](
            norm(self.hf[::-1, :, 2]), bytes=True)[:, :, :3])
        self.full_lf = [Image.fromarray(colormaps["inferno"](
            norm(field[::-1, :, 2]), bytes=True)[:, :, :3]).resize(
                (250, 250), Image.Resampling.NEAREST) for field in self.fields]
        self._hero = None
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
        if state.stage == 4:
            return self.hero().copy()
        if state.stage == 3:
            return self.assembly_frame(state)
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
        d.rectangle((1065,238,1097,270), fill="#eff5f5", outline=TEAL, width=2)
        d.ellipse((1078,251,1084,257), fill=TEAL)
        text(1081, 289, "Only this point", 16, TEAL, anchor="mt")
        text(714, 331, "Next: assemble the full field", 24, TEAL, True)
        text(714, 365, "801 × 801 POIs · temporal skip", 18, MUTED)
        # Preview the target geometry without inventing a low-resolution puzzle.
        d.rectangle((826,404,1114,692), fill="#f1f6f7")
        text(970, 523, "641,601", 34, TEAL, True, "mt")
        text(970, 572, "target locations", 18, MUTED, anchor="mt")
        text(714, 710, "Reveal colors: packaged HF reference", 16, MUTED)
        # Actual Graphviz output, not a hand-drawn substitute.
        im.paste(self.route, ((WIDTH-self.route.width)//2, 725), self.route)
        text(34, 822, "LF density: independent CFD · fixed time · same crop & scale", 12, MUTED)
        text(1166, 822, "Assembly schematic · no model output", 12, MUTED, anchor="rt")
        return im


    def assembly_image(self, filled):
        """Reveal exactly `filled` native HF nodes before display downsampling."""
        if not 0 <= filled <= POINT_COUNT:
            raise ValueError("Invalid POI count")
        result = Image.new("RGB", self.reference.size, "#f1f6f7")
        rows, cols = divmod(filled, GRID_SIZE)
        if rows:
            result.paste(self.reference.crop((0, 0, GRID_SIZE, rows)), (0, 0))
        if cols:
            result.paste(self.reference.crop((0, rows, cols, rows+1)), (0, rows))
        return result

    def patch_inset(self, im, field, point, left, top, side, color):
        """No clamping, padding, wrapping or feature markers at missing patches."""
        if not complete_stencil(point):
            return
        d = ImageDraw.Draw(im)
        xx, yy, values = native_patch(field, point)
        half = PATCH_SIDE/2
        for j, sy in enumerate(yy):
            for i, sx in enumerate(xx):
                n = field.shape[0]-1
                xa, xb = max(point[0]-half,sx-.5/n), min(point[0]+half,sx+.5/n)
                ya, yb = max(point[1]-half,sy-.5/n), min(point[1]+half,sy+.5/n)
                box = (left+(xa-point[0]+half)/PATCH_SIDE*side,
                       top+(point[1]+half-yb)/PATCH_SIDE*side,
                       left+(xb-point[0]+half)/PATCH_SIDE*side,
                       top+(point[1]+half-ya)/PATCH_SIDE*side)
                tint = tuple(int(v*255) for v in colormaps["inferno"](self.norm(values[j,i]))[:3])
                d.rectangle(box, fill=tint)
        d.rectangle((left,top,left+side,top+side), outline=color, width=2)
        for sx, sy in stencil(point):
            dx = left+(sx-point[0]+half)/PATCH_SIDE*side
            dy = top+(point[1]+half-sy)/PATCH_SIDE*side
            d.ellipse((dx-4,dy-4,dx+4,dy+4), fill="white", outline=INK, width=1)
        cx, cy = left+side/2, top+side/2
        d.ellipse((cx-8,cy-8,cx+8,cy+8), outline="white", width=2)

    def assembly_frame(self, state):
        im = Image.new("RGB", (WIDTH, HEIGHT), "white")
        d = ImageDraw.Draw(im)

        def text(x, y, value, size=18, color=INK, bold=False, anchor="lt"):
            d.text((x, y), value, font=font(size, bold), fill=color, anchor=anchor)

        valid = complete_stencil(state.point)
        text(34, 23, "801 × 801 POIs. The full field, fast.", 31, INK, True)
        text(34, 65, "Assembly schematic · temporal skip · HF reference colors, not predictions", 19, MUTED)
        d.rounded_rectangle((34,105,1166,140), radius=8, fill="#e8f5f3")
        text(50, 114, "Two 3 × 3 patches → MLP → one POI → repeat over the entire domain", 18, TEAL, True)
        d.line([(674,162),(674,700)], fill="#dfe7ee", width=1)
        for index, (field, cx, color) in enumerate(zip(self.fields, (177,482), (BLUE,GREEN))):
            left, top = cx-125, 203
            text(left, 164, f"LF{index+1}", 27, color, True)
            text(left+74, 173, f"{field.shape[0]-1}² mesh", 17, MUTED)
            im.paste(self.full_lf[index], (left,top))
            dx, dy = left+250*state.point[0], top+250*(1-state.point[1])
            d.ellipse((dx-6,dy-6,dx+6,dy+6), fill=color, outline="white", width=2)
            text(cx, 470, "3 × 3 inputs", 21, color, True, "mt")
            self.patch_inset(im, field, state.point, cx-68, 503, 136, color)
            text(cx, 651, "9 locations · 4 variables" if valid else "Boundary: no full patch",
                 16, MUTED, anchor="mt")
        text(329, 690, f"Full domain · POI ({state.point[0]:.4f}, {state.point[1]:.4f})",
             16, MUTED, anchor="mt")
        text(724, 168, "NNLCI MLP → one POI" if valid else "Boundary · patch panels blank",
             22, TEAL if valid else MUTED, True)
        text(724, 205, "72 features → 4 variables at one location" if valid else "Reference reveal continues across the boundary",
             16, MUTED)
        left, top, side = TARGET_BOX
        im.paste(self.assembly_image(state.filled).resize((side,side), Image.Resampling.NEAREST), (left,top))
        d.rectangle((left-1,top-1,left+side,top+side), outline="#bdccd5")
        if 0 < state.filled < POINT_COUNT:
            x, y = left+side*state.point[0], top+side*(1-state.point[1])
            d.line((left,y,left+side,y), fill="white", width=1)
            d.ellipse((x-4,y-4,x+4,y+4), fill=TEAL, outline="white", width=1)
        text(724, 690, f"{state.filled:,} / {POINT_COUNT:,} POIs", 24, TEAL, True)
        im.paste(self.route, ((WIDTH-self.route.width)//2, 735), self.route)
        text(34, 822, "LF density: independent CFD · full domain · fixed physical time", 12, MUTED)
        text(1166, 822, "Temporal skip illustrates assembly, not measured speed", 12, MUTED, anchor="rt")
        return im

    def hero(self):
        """Final full-domain reference plus a clearly located, enlarged detail."""
        if self._hero is not None:
            return self._hero
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, ConnectionPatch
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                             "text.color": INK, "axes.labelcolor": MUTED,
                             "xtick.color": MUTED, "ytick.color": MUTED})
        fig = plt.figure(figsize=(12, 8.4), dpi=100, facecolor="white")
        fig.text(.035,.956,"The complete high-fidelity field.",fontsize=27,weight="bold")
        fig.text(.035,.913,"Config 3 · packaged CFD reference · density ρ",fontsize=14,color=MUTED)
        fig.text(.968,.915,"801 × 801 POIs",fontsize=16,color=TEAL,weight="bold",ha="right")
        full = fig.add_axes((.067,.157,.505,.72143))
        detail = fig.add_axes((.665,.34,.30,.42857))
        x, y, z = self.hf[0,:,0], self.hf[:,0,1], self.hf[:,:,2]
        full.imshow(z, origin="lower", extent=(-.5/800,1+.5/800,-.5/800,1+.5/800),
                    cmap="inferno", norm=self.norm, interpolation="nearest")
        full.set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",xticks=[0,.5,1],yticks=[0,.5,1])
        window = (.24,.54,.22,.52)
        ix, iy = (x>=window[0]) & (x<=window[1]), (y>=window[2]) & (y<=window[3])
        zz = z[np.ix_(iy,ix)]
        levels = np.linspace(self.norm.vmin,self.norm.vmax,81)
        detail.contourf(x[ix],y[iy],zz,levels=levels,cmap="inferno",norm=self.norm,antialiased=False)
        detail.contour(x[ix],y[iy],zz,levels=np.linspace(self.norm.vmin,self.norm.vmax,17)[1:-1],
                       colors=INK,linewidths=.35,alpha=.30)
        detail.set(xlim=window[:2],ylim=window[2:],aspect="equal",xlabel="x",ylabel="y",
                   xticks=[.25,.4,.5],yticks=[.25,.4,.5])
        detail.set_title("Interaction detail",loc="left",fontsize=15,weight="bold",pad=14)
        full.add_patch(Rectangle((window[0],window[2]),.30,.30,fill=False,edgecolor="white",linewidth=1.4))
        fig.add_artist(ConnectionPatch((.54,.52),(.24,.52),coordsA="data",coordsB="data",
                                      axesA=full,axesB=detail,color="#aebcc5",linewidth=.8))
        for ax in (full,detail):
            ax.tick_params(length=3,pad=4)
            for spine in ax.spines.values():
                spine.set_color("#aebcc5")
        fig.text(.665,.247,"641,601 locations · 800 intervals / axis",fontsize=11,color=TEAL)
        fig.text(.665,.215,"Detail: contour display of the same HF values",fontsize=10,color=MUTED)
        cbax = fig.add_axes((.665,.165,.30,.018))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=self.norm,cmap="inferno"),cax=cbax,
                         orientation="horizontal",ticks=[self.norm.vmin,(self.norm.vmin+self.norm.vmax)/2,self.norm.vmax],format="%.3f")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=10,length=2)
        fig.text(.665,.108,"Density ρ · shared global scale",fontsize=10,color=MUTED)
        fig.text(.035,.064,"All grid locations revealed. Assembly timing is schematic.",fontsize=13,color=TEAL,weight="bold")
        fig.text(.035,.025,"HF reference, not NNLCI output · fixed physical time · no new inference or speedup measurement",fontsize=11,color=MUTED)
        fig.canvas.draw()
        self._hero = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        plt.close(fig)
        return self._hero


def hex_rgb(color):
    return tuple(int(color[i:i+2], 16) for i in (1,3,5))


def render_static(scene):
    # Use the final high-fidelity field as the motion-free opening, too.
    scene.hero().save(OUT / "nnlci_motion_poster.png", optimize=True)
    scene.hero().save(OUT / "nnlci_hf_final.png", optimize=True)
    sheet = Image.new("RGB", (WIDTH, HEIGHT*3), "white")
    for index, frame in enumerate((13, INTRO_FRAMES+len(ASSEMBLY_COUNTS)*2//3, FRAME_COUNT-1)):
        sheet.paste(scene.frame(frame), (0,index*HEIGHT))
    sheet.save(OUT / "nnlci_local_to_hf_concept.png", optimize=True)
    return ["nnlci_motion_poster.png", "nnlci_hf_final.png", "nnlci_local_to_hf_concept.png"]


def animate(scene):
    available = memory_gate()
    if available < 1500:
        raise RuntimeError("Animation deferred below 1500 MiB effective available RAM")
    # Sample the finished scientific image and the input scene for one palette.
    samples = Image.new("RGB", (WIDTH,HEIGHT*2))
    samples.paste(scene.hero(),(0,0))
    samples.paste(scene.frame(13),(0,HEIGHT))
    # Reserve most colors for the scientific scale. Unweighted quantization of
    # these mostly white layouts otherwise produces visible bands in the HF.
    accents = samples.quantize(colors=64,method=Image.Quantize.MEDIANCUT)
    density_colors = colormaps["inferno"](np.linspace(0,1,192),bytes=True)[:,:3]
    palette = Image.new("P", (1,1))
    palette.putpalette(density_colors.ravel().tolist()+accents.getpalette()[:64*3])
    del samples
    frames = []
    for index in range(FRAME_COUNT):
        if index % 24 == 0:
            print(f"POI animation: {index}/{FRAME_COUNT}", flush=True)
        frames.append(scene.frame(index).quantize(palette=palette,dither=Image.Dither.NONE))
    durations = [40]*FRAME_COUNT
    for index in (13,27,43):
        durations[index] = 300
    for index in range(INTRO_FRAMES,INTRO_FRAMES+3):
        durations[index] = 180
    durations[-2] = 350
    durations[-1] = 4500
    frames[0].save(OUT / "nnlci_local_to_hf.gif",save_all=True,append_images=frames[1:],
                   duration=durations,loop=0,optimize=False,disposal=1)
    with Image.open(OUT / "nnlci_local_to_hf.gif") as gif:
        actual_frames,total_ms = gif.n_frames,0
        for i in range(actual_frames):
            gif.seek(i)
            total_ms += gif.info["duration"]
    return {"file":"nnlci_local_to_hf.gif","frames":actual_frames,"rendered_frames":FRAME_COUNT,
            "dimensions":[WIDTH,HEIGHT],"duration_ms":total_ms,"final_hold_ms":durations[-1],
            "palette":"192 inferno scale colors + 64 adaptive annotation colors; no dithering",
            "meaning":"801x801 reference locations revealed in row-major order with temporal skips; schematic assembly, not inference, physical evolution or measured speed",
            "memory_before_mib":available}


def semantics():
    return {
        "patch":{"context_physical_side":PATCH_SIDE,"stencil_physical_span":.02,
                 "lf1_input_locations":[3,3],"lf2_input_locations":[3,3],
                 "native_mesh_spacing":{"lf1":.01,"lf2":.005},
                 "interpretation":"Nine aligned feature locations on each LF source; native LF2 mesh is finer",
                 "boundary":"Both lower panels blank when any of the 3x3 display locations lies outside [0,1]^2; no padding, clamping, or boundary prediction invented",
                 "display_contract":"Illustrative aligned coordinates; packaged bootstrap upsamples then extracts dilation-4 3x3 stencils. Preprocessing unchanged."},
        "pois":{"sharp_transition":list(SHARP),"smooth_region":list(SMOOTH),
                "selection":"Fixed LF display examples, then all target coordinates in deterministic row-major order; no HF-dependent selection"},
        "model_contract":"Two 3x3 stencils, four variables each, 72 features to four variables at ONE POI; unchanged; no inference performed",
        "reference_assembly":{"kind":"Packaged HF density, explicitly labeled as reference colors; not NNLCI predictions",
                              "grid_nodes":[GRID_SIZE,GRID_SIZE],"intervals_per_axis":800,"point_count":POINT_COUNT,
                              "domain_xy":[0,1,0,1],"order":"Top-left to bottom-right in display space, including every boundary node",
                              "displayed_counts":list(ASSEMBLY_COUNTS),
                              "assembly":"Every node between successive counts is revealed in the native 801x801 image; only temporal steps are skipped. Existing values stay fixed.",
                              "timing":"Short accelerated assembly, explicitly labeled temporal skip; not a measured NNLCI speedup",
                              "final":"Full-domain HF density with enlarged interaction detail, held 4500 ms; also saved as a lossless PNG",
                              "final_rendering":"Full field: native node colors, nearest display resampling. Detail: 80 contour intervals and 15 isolines, shared global linear density scale."},
        "camera":"Intro uses OpenSCAD meshes, shared crop, tilt 52 to 0 degrees. Sweep switches both LF panels and the target to full domain.",
    }


if __name__ == "__main__":
    raise SystemExit("Run scripts/render_gatech_promo.py to validate sources and render")
