"""Two CPU-only SVG/Python animations: visual comparison and a paced lesson.

Reads independently solved LF CFD and a labeled HF reference. No inference,
synthetic LF, altered CFD values, or model-performance claim. Historical PR #7
assets remain reproducible with render_gatech_promo.py.
"""
from __future__ import annotations

import argparse
import base64
from html import escape
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nnlci-matplotlib")
import cairosvg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image, __version__ as pillow_version

from cfd_io import read_field, sha
from replot_sod_promo import memory_gate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/promo"
CFD = ROOT / "problems/2d_riemann/data/promo_cfd/provenance.json"
W, H = 960, 540
WINDOW = (.26, .52, .26, .52)
POINTS = ((.34, .44), (.46, .48), (.40, .36))
INK, MUTED = "#162536", "#596879"
BLUE, GREEN, TEAL, RED = "#2563eb", "#159947", "#087f8c", "#dc2626"
ROLES = (BLUE, GREEN, TEAL)
VARIABLES = ("ρ", "u", "v", "p")
VARIABLE_COLORS = ("#087f8c", "#7c64b5", "#bd741b", "#c5444d")
GRID = 801
TOTAL = GRID * GRID
# Each state includes every intervening native location, not spatial subsampling.
COUNTS = (0, 1, GRID) + tuple(GRID + (TOTAL-GRID)*k//24 for k in range(1, 25))
TITLES = ("Choose the same point", "Four variables at each location",
          "Assemble the input vector", "Predict one point", "Learn from HF labels",
          "Place the point. Repeat.")
HOLDS = (4200, 4200, 5000, 5000, 5000)


class SVG:
    def __init__(self, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
                      f'<title>{escape(title)}</title>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#82929f"/></marker></defs>',
                      f'<rect width="{W}" height="{H}" fill="white"/>']

    def text(self, x, y, value, size=20, color=INK, bold=False, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')

    def rect(self, x, y, w, h, fill="none", stroke="none", radius=0, width=2):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def line(self, coords, color="#b4c1ce", arrow=False, dash=False):
        self.parts.append(f'<path d="{coords}" fill="none" stroke="{color}" stroke-width="2"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="6 5"' if dash else '')+'/>')

    def dot(self, x, y, radius=5, fill="white", stroke=INK):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def image(self, image, x, y, size):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        self.parts.append(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{data}"/>')

    def bytes(self):
        return ("\n".join(self.parts)+"\n</svg>\n").encode()

    def raster(self):
        return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=self.bytes()))).convert("RGB")


def load_sources():
    manifest = json.loads(CFD.read_text())
    sources = [{"path": str((CFD.parent / run["field"]).relative_to(ROOT)),
                "sha256": run["field_sha256"]} for run in manifest["runs"]]
    sources.append({key: manifest["packaged_hf_reference"][key] for key in ("path", "sha256")})
    for source in sources:
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"CFD source hash mismatch: {source['path']}")
    fields = [read_field(ROOT / source["path"]) for source in sources]
    if [f.shape for f in fields] != [(101, 101, 6), (201, 201, 6), (801, 801, 6)]:
        raise ValueError("Expected independent 100/200 and packaged 800 interval grids")
    for field in fields[:2]:
        stride = (GRID-1)//(field.shape[0]-1)
        if np.allclose(field[:, :, 2:], fields[2][::stride, ::stride, 2:]):
            raise ValueError("LF must not be an HF subsample")
    return fields, sources


def samples(point):
    """Nine aligned illustrative locations; spacing is fixed in physical space."""
    return np.array([(point[0]+i*.01, point[1]+j*.01)
                     for j in (1, 0, -1) for i in (-1, 0, 1)])


def complete(point):
    return bool(np.all((samples(point) >= 0) & (samples(point) <= 1)))


class Artwork:
    def __init__(self, fields):
        self.fields = fields
        self.norm = Normalize(min(f[:, :, 2].min() for f in fields),
                              max(f[:, :, 2].max() for f in fields))
        self.reference = Image.fromarray(colormaps["inferno"](
            self.norm(fields[2][::-1, :, 2]), bytes=True)[:, :, :3])
        self.details = [self.contours(f) for f in fields]

    def contours(self, field):
        # Standard contour interpolation of the *native* CFD values, identical
        # levels/window/scale for all fields. No blur or filtered data arrays.
        fig = plt.figure(figsize=(3.2, 3.2), dpi=100)
        ax = fig.add_axes((0, 0, 1, 1))
        x, y = field[0, :, 0], field[:, 0, 1]
        ix = np.flatnonzero((x >= WINDOW[0]-.02) & (x <= WINDOW[1]+.02))
        iy = np.flatnonzero((y >= WINDOW[2]-.02) & (y <= WINDOW[3]+.02))
        ax.contourf(x[ix], y[iy], field[:, :, 2][np.ix_(iy, ix)],
                    levels=np.linspace(self.norm.vmin, self.norm.vmax, 160),
                    cmap="inferno", norm=self.norm, antialiased=False)
        ax.set(xlim=WINDOW[:2], ylim=WINDOW[2:], aspect="equal")
        ax.set_axis_off()
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)
        return image

    def locate(self, point, x, y, size, window=WINDOW):
        return (x+(point[0]-window[0])/(window[1]-window[0])*size,
                y+(window[3]-point[1])/(window[3]-window[2])*size)

    def patch(self, svg, level, point, x, y, size, active=4):
        # Blank at the boundary; never clip nine samples into the domain.
        if not complete(point):
            return
        field = self.fields[level]
        n = field.shape[0]-1
        values = [field[round(py*n), round(px*n), 2] for px, py in samples(point)]
        colors = colormaps["inferno"](self.norm(values), bytes=True)
        step = size/3
        for k, color in enumerate(colors):
            col, row = k % 3, k // 3
            fill = "#"+"".join(f"{v:02x}" for v in color[:3])
            svg.rect(x+col*step, y+row*step, step, step, fill, "white", width=1)
            svg.dot(x+(col+.5)*step, y+(row+.5)*step, 4, "white", ROLES[level])
        col, row = active % 3, active // 3
        svg.dot(x+(col+.5)*step, y+(row+.5)*step, 10, "none", "white")
        svg.rect(x, y, size, size, stroke=ROLES[level], width=3)

    def reveal(self, count):
        if not 0 <= count <= TOTAL:
            raise ValueError(count)
        pixels = np.full((GRID, GRID, 3), (241, 246, 247), dtype=np.uint8)
        pixels.reshape(-1, 3)[:count] = np.asarray(self.reference).reshape(-1, 3)[:count]
        return Image.fromarray(pixels)

    def opening(self, point):
        svg = SVG("Conceptual comparison: independent LF1/LF2 CFD and HF output target")
        for level, x in enumerate((32, 342, 652)):
            title = ("LF1", "LF2", "Output target · HF")[level]
            svg.text(x+138, 35, title, 23, ROLES[level], True, "middle")
            svg.image(self.details[level], x, 60, 276)
            px, py = self.locate(point, x, 60, 276)
            if level < 2:
                side = .02/(WINDOW[1]-WINDOW[0])*276
                svg.rect(px-side/2, py-side/2, side, side, stroke="white")
                svg.line(f"M {px} {py+side/2} L {x+138} 367", ROLES[level])
                self.patch(svg, level, point, x+66, 367, 144)
            else:
                svg.dot(px, py, 5, "none", "white")
                svg.image(self.reference, x+66, 367, 144)
                wx, wy = self.locate((WINDOW[0], WINDOW[3]), x+66, 367, 144, (0, 1, 0, 1))
                svg.rect(wx, wy, (WINDOW[1]-WINDOW[0])*144,
                         (WINDOW[3]-WINDOW[2])*144, stroke="white", width=1)
        return svg

    def method(self, stage, count=TOTAL, active=4, collected=72):
        svg = SVG("Schematic: "+TITLES[stage])
        svg.text(32, 42, f"0{stage+1}", 19, TEAL, True)
        svg.text(79, 42, TITLES[stage], 28, INK, True)
        for k in range(6):
            svg.rect(32+k*151, 66, 140, 4, TEAL if k == stage else "#e8eef2", radius=2)
        svg.text(32, 520, "Schematic · Config 3 · field artwork is CFD reference, not model output", 16, MUTED)
        if stage == 0:
            for level, x in enumerate((95, 545)):
                svg.text(x+105, 113, ("LF1", "LF2")[level], 24, ROLES[level], True, "middle")
                svg.image(self.details[level], x, 135, 210)
                px, py = self.locate(POINTS[0], x, 135, 210)
                svg.rect(px-9, py-9, 18, 18, stroke="white")
                svg.line(f"M {px} {py+9} L {x+255} 367", ROLES[level])
                self.patch(svg, level, POINTS[0], x+210, 337, 114)
                svg.text(x+105, 388, "3 × 3 locations", 20, ROLES[level], True, "middle")
            svg.text(480, 477, "Same physical point, case and time", 23, INK, False, "middle")
        elif stage == 1:
            for level, x in enumerate((65, 300)):
                svg.text(x+81, 164, ("LF1", "LF2")[level], 23, ROLES[level], True, "middle")
                self.patch(svg, level, POINTS[0], x, 190, 162, active)
            svg.line("M 480 270 H 575", arrow=True)
            for k, (variable, color, name) in enumerate(zip(VARIABLES, VARIABLE_COLORS,
                                                         ("density", "x velocity", "y velocity", "pressure"))):
                svg.rect(608, 136+k*68, 284, 54, "#f5f8fa", radius=9)
                svg.text(633, 172+k*68, variable, 27, color, True)
                svg.text(691, 170+k*68, name, 21)
            svg.text(264, 414, "Every dot holds four values", 23, INK, False, "middle")
        elif stage == 2:
            for level, x in enumerate((73, 339)):
                svg.text(x+92, 161, f"LF{level+1} · 36 values", 21, ROLES[level], True, "middle")
                for row, (variable, color) in enumerate(zip(VARIABLES, VARIABLE_COLORS)):
                    svg.text(x-18, 208+row*41, variable, 22, color, True, "end")
                    for col in range(9):
                        svg.rect(x+col*21, 186+row*41, 17, 29, color, radius=3)
                svg.text(x+92, 383, "9 locations per variable", 16, MUTED, anchor="middle")
            svg.line("M 548 260 H 610", arrow=True)
            svg.rect(639, 180, 269, 166, "#f4f8f9", radius=12)
            for row in range(4):
                for col in range(18):
                    # LF1's four variable blocks, then LF2's four blocks.
                    index = row*18+col
                    color = VARIABLE_COLORS[(index % 36)//9] if index < collected else "#dce5eb"
                    svg.rect(654+col*13, 198+row*33, 9, 23, color, radius=2)
            svg.text(773, 383, "LF1, then LF2", 20, TEAL, True, "middle")
            svg.text(480, 463, "2 × 9 × 4 = 72 features", 29, INK, True, "middle")
        elif stage in (3, 4):
            svg.text(112, 213, "72 inputs", 22, BLUE, True, "middle")
            for row in range(6):
                for col in range(12):
                    svg.rect(49+col*11, 236+row*12, 7, 8, BLUE if row < 3 else GREEN, radius=1)
            svg.line("M 196 272 H 263", arrow=True)
            svg.rect(280, 224, 158, 96, "#f4f7fa", "#d6e0e8", 12)
            svg.text(359, 262, "Fixed training", 18, anchor="middle")
            svg.text(359, 290, "transforms", 19, bold=True, anchor="middle")
            svg.line("M 452 272 H 501", arrow=True)
            svg.rect(518, 209, 143, 126, "#eaf5f5", "#abd2d5", 12)
            svg.text(590, 265, "NNLCI", 27, TEAL, True, "middle")
            svg.text(590, 301, "MLP", 20, TEAL, anchor="middle")
            svg.line("M 676 272 H 728", TEAL, arrow=True)
            svg.rect(746, 238, 35, 35, TEAL)
            svg.dot(763.5, 255.5, 3, "white", "white")
            svg.text(806, 249, "ρ̂, û, v̂, p̂", 24, TEAL)
            svg.text(832, 302, "One point", 21, TEAL, True, "middle")
            if stage == 3:
                svg.text(480, 432, "Four predicted variables at the center point", 24, anchor="middle")
            else:
                svg.rect(30, 378, 466, 100, "#fff5f5", radius=10)
                svg.text(48, 406, "TRAINING ONLY", 16, RED, True)
                svg.text(51, 448, "HF label at this point", 20, RED)
                svg.line("M 275 443 H 350", RED, arrow=True)
                svg.rect(366, 419, 96, 45, "white", RED, 6)
                svg.text(414, 449, "Loss", 21, RED, True, "middle")
                svg.line("M 832 327 V 443 H 476", TEAL, arrow=True)
                svg.line("M 414 417 V 356 H 590 V 341", RED, arrow=True, dash=True)
                svg.text(657, 384, "Update weights", 18, RED)
        else:
            # This explicitly labeled illustration includes boundary reference
            # values; no neural boundary prediction is claimed.
            svg.text(68, 123, "LF1", 22, BLUE, True)
            svg.text(248, 123, "LF2", 22, GREEN, True)
            row, col = divmod(max(0, count-1), GRID)
            point = (col/(GRID-1), 1-row/(GRID-1))
            for level, x in enumerate((65, 245)):
                self.patch(svg, level, point, x, 149, 120)
            svg.text(215, 307, "3 × 3 → MLP → one POI", 21, TEAL, True, "middle")
            svg.text(215, 349, "Blank near boundaries:", 17, MUTED, anchor="middle")
            svg.text(215, 375, "a full stencil is unavailable", 17, MUTED, anchor="middle")
            svg.line("M 396 259 H 487", TEAL, arrow=True)
            svg.image(self.reveal(count), 510, 113, 350)
            svg.text(685, 100, "HF reference · assembly illustration", 18, TEAL, True, "middle")
            svg.text(685, 489, f"{count:,} / {TOTAL:,} locations", 20, TEAL, True, "middle")
            svg.text(215, 435, "Temporal skips", 21, bold=True, anchor="middle")
            svg.text(215, 463, "801 × 801 target grid", 20, anchor="middle")
        return svg

    def finished(self):
        svg = SVG("Finished assembly illustration: full HF CFD reference and interaction detail")
        svg.text(32, 42, "The complete field", 28, INK, True)
        svg.text(928, 42, "HF reference · not model output", 20, TEAL, True, "end")
        svg.image(self.reference, 40, 95, 385)
        svg.image(self.details[2], 575, 127, 305)
        px, py = self.locate((WINDOW[0], WINDOW[3]), 40, 95, 385, (0, 1, 0, 1))
        side = (WINDOW[1]-WINDOW[0])*385
        svg.rect(px, py, side, side, stroke="white")
        svg.line(f"M {px+side} {py} L 561 127 M {px+side} {py+side} L 561 432", TEAL)
        svg.text(727, 470, "Interaction detail", 22, TEAL, True, "middle")
        svg.text(32, 521, "801 × 801 locations · 641,601 / 641,601 · same density scale", 20, MUTED)
        return svg


def save_gif(path, frames):
    """Only indexed frames retained; one SVG raster is live at a time."""
    images, durations = [], []
    palette = None
    for svg, duration in frames:
        raster = svg.raster()
        if palette is None:
            # Reserve UI colors as well as the scientific colormap.
            colors = [tuple(c[:3]) for c in colormaps["inferno"](np.linspace(0, 1, 224), bytes=True)]
            from PIL import ImageColor
            colors += [ImageColor.getrgb(c) for c in ("white", INK, MUTED, BLUE, GREEN, TEAL, RED,
                       "#f1f6f7", "#f4f7fa", "#eaf5f5", "#fff5f5", *VARIABLE_COLORS)]
            colors += [(v, v, v) for v in range(0, 256, 16)]
            palette = Image.new("P", (1, 1))
            flat = [int(v) for color in colors[:256] for v in color]
            palette.putpalette(flat+[255]*(768-len(flat)))
        images.append(raster.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(duration)
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, disposal=1, optimize=False)
    with Image.open(path) as gif:
        return {"file": path.name, "frames": gif.n_frames, "duration_ms": sum(durations),
                "size": [W, H], "final_hold_ms": durations[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    available = memory_gate()
    if available < 700:
        raise SystemExit("Static rendering deferred below 700 MiB available RAM")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fields, sources = load_sources()
    art = Artwork(fields)
    names = []
    poster = art.opening(POINTS[0])
    for ext, data in (("svg", poster.bytes()), ("png", cairosvg.svg2png(bytestring=poster.bytes()))):
        name = "nnlci_comparison_poster."+ext
        (out / name).write_bytes(data)
        names.append(name)
    storyboard = Image.new("RGB", (2*W, 3*H), "white")
    for stage in range(6):
        slide = art.method(stage, count=449*GRID+272)
        name = f"nnlci_method_step_{stage+1}.svg"
        (out / name).write_bytes(slide.bytes())
        names.append(name)
        storyboard.paste(slide.raster(), ((stage % 2)*W, (stage//2)*H))
    name = "nnlci_method_storyboard.png"
    storyboard.save(out / name)
    names.append(name)
    name = "nnlci_method_final.png"
    art.finished().raster().save(out / name)
    names.append(name)
    animation_memory = memory_gate()
    motion = {}
    if not args.static_only and animation_memory >= 2500:
        def opening_frames():
            for index, point in enumerate(POINTS):
                yield art.opening(point), 5000 if index == 2 else 2400
                if index < 2:
                    for step in range(1, 7):
                        point_t = tuple(a+(b-a)*step/7 for a, b in zip(point, POINTS[index+1]))
                        yield art.opening(point_t), 180

        def method_frames():
            for stage, duration in enumerate(HOLDS):
                if stage == 1:
                    for active in range(9):
                        yield art.method(stage, active=active), 300
                    yield art.method(stage), duration-2700
                elif stage == 2:
                    yield art.method(stage, collected=0), 700
                    for collected in range(9, 73, 9):
                        yield art.method(stage, collected=collected), 250
                    yield art.method(stage), duration-2700
                else:
                    yield art.method(stage), duration
            for count in COUNTS[:-1]:
                yield art.method(5, count), 160 if count > GRID else 700
            yield art.method(5, TOTAL), 1200
            yield art.finished(), 4500

        for key, frames in (("comparison", opening_frames()), ("method", method_frames())):
            # Recheck before encoding each animation, after releasing the first.
            if memory_gate() < 2500:
                motion[key] = {"deferred": True, "reason": "Below 2500 MiB available RAM"}
                (out / f"nnlci_{key}.gif").unlink(missing_ok=True)
                continue
            path = out / f"nnlci_{key}.gif"
            motion[key] = save_gif(path, frames)
            names.append(path.name)
    else:
        for key in ("comparison", "method"):
            motion[key] = {"deferred": True, "reason": "--static-only" if args.static_only else "Below 2500 MiB available RAM"}
            (out / f"nnlci_{key}.gif").unlink(missing_ok=True)
    if out == OUT:
        readme = ROOT / "README.md"
        content = readme.read_text()
        import re
        for key, fallback in (("comparison", "nnlci_comparison_poster.png"), ("method", "nnlci_method_storyboard.png")):
            target = fallback if motion[key].get("deferred") else motion[key]["file"]
            # Image embeds only; permanent static links stay available.
            content = re.sub(r'(!\[[^\]]*\]\(docs/assets/promo/)(?:nnlci_'+key+r'\.gif|'+re.escape(fallback)+r')(\))',
                             lambda m: m[1]+target+m[2], content)
        readme.write_text(content)
    report = {
        "operation": "Re-render independent CFD comparison and a paced conceptual lesson; no inference or training",
        "backend": "CPU: Matplotlib contours + editable SVG + CairoSVG + Pillow",
        "sources": sources,
        "cfd_provenance": {"path": str(CFD.relative_to(ROOT)), "sha256": sha(CFD)},
        "renderers": {name: sha(ROOT / "scripts" / name) for name in
                      (Path(__file__).name, "cfd_io.py", "replot_sod_promo.py")},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                     "cairosvg": cairosvg.__version__, "pillow": pillow_version},
        "display": {"variable": "density", "shared_window_xy": WINDOW,
                    "global_density_limits": [float(art.norm.vmin), float(art.norm.vmax)],
                    "details": "160 shared contour levels interpolated from native CFD; no blur/sharpening/data smoothing",
                    "patches": "9 nearest native samples per LF, aligned illustrative offsets of +/-0.01; not the bootstrap dilation-4 preprocessing",
                    "assembly": "All 641601 native HF reference colors in row-major order; temporal skips, not physical time or model inference",
                    "boundary": "Blank patches where full stencil is unavailable; boundary reference values are not predicted",
                    "feature_order": "LF1 then LF2; variable-major rho/u/v/p, row-major nine locations per variable"},
        "method": {"chapter_titles": TITLES, "chapter_holds_ms": HOLDS, "assembly_counts": COUNTS,
                   "hf_route": "HF label -> loss <- predicted point; loss updates MLP only during training"},
        "memory": {"before_render_mib": available, "before_animation_mib": animation_memory,
                   "static_threshold_mib": 700, "animation_threshold_mib": 2500,
                   "storage": "At most 51 indexed 960x540 frames (~26 MiB), one transient SVG/RGB frame; animations encoded sequentially"},
        "animations": motion,
        "outputs": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size} for name in names},
    }
    (out / "followup_provenance.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"animations": motion, "outputs": names}, indent=2))


if __name__ == "__main__":
    main()
