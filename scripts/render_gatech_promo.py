"""Render independently solved LF fields, an HF target, and the NNLCI workflow.

The animation moves sampling highlights, never field values. HF is a reference,
not a model prediction. All three tiles use the same physical crop, color scale,
display size and nearest-node display rule; LF fields are never derived from HF.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nnlci-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
from cfd_io import read_field, sha
from replot_sod_promo import memory_gate
from render_config3_contours import scientific_panels
from render_promo_mechanism import render as mechanism

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/promo"
LF = ROOT / "problems/2d_riemann/data/promo_cfd"
HF = ROOT / "problems/2d_riemann/data/raw/flow_standard_config3.dat"
WINDOW = (.30, .50, .30, .50)
INK, MUTED, BLUE, GREEN, TEAL = "#162536", "#596879", "#2563eb", "#159947", "#087f8c"
WIDTH, HEIGHT = 1080, 430


class Canvas:
    def __init__(self, scale=1):
        self.scale = scale
        self.im = Image.new("RGB", (round(WIDTH*scale), round(HEIGHT*scale)), "white")
        self.d = ImageDraw.Draw(self.im)

    def xy(self, values):
        return tuple(round(v*self.scale) for v in values)

    def text(self, x, y, label, size=18, color=INK, bold=False, anchor="lt"):
        font = ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", round(size*self.scale))
        self.d.text(self.xy((x, y)), label, font=font, fill=color, anchor=anchor)

    def line(self, points, color="#b9c5d1", width=1):
        self.d.line([self.xy(p) for p in points], fill=color, width=max(1, round(width*self.scale)))

    def rect(self, box, fill=None, outline=None, width=1, radius=0):
        self.d.rounded_rectangle(self.xy(box), radius=round(radius*self.scale), fill=fill,
                                outline=outline, width=max(1, round(width*self.scale)))

    def dot(self, x, y, color, r=4):
        self.d.ellipse(self.xy((x-r, y-r, x+r, y+r)), fill=color)

    def arrow(self, start, end, color=INK, dashed=False):
        if dashed:
            for t in np.arange(0, 1, .16):
                self.line([(start[0]+t*(end[0]-start[0]), start[1]),
                           (start[0]+min(t+.08, 1)*(end[0]-start[0]), end[1])], color, 2)
        else:
            self.line([start, end], color, 2)
        x, y = end
        self.d.polygon([self.xy(p) for p in [(x, y), (x-8, y-5), (x-8, y+5)]], fill=color)

    def save(self, name):
        self.im.save(OUT / name, dpi=(240, 240), optimize=True)


def field_tile(field, norm, pixels):
    # Sample display-pixel centers at the nearest original node. Every source is
    # treated identically; no decimation, blur, sharpening or field interpolation.
    n = field.shape[0]-1
    xx = WINDOW[0]+(np.arange(pixels)+.5)/pixels*(WINDOW[1]-WINDOW[0])
    yy = WINDOW[3]-(np.arange(pixels)+.5)/pixels*(WINDOW[3]-WINDOW[2])
    values = field[np.ix_(np.rint(yy*n).astype(int), np.rint(xx*n).astype(int))][:, :, 2]
    return Image.fromarray((colormaps["inferno"](norm(values))[:, :, :3]*255).astype(np.uint8))


def workflow(fields, norm, phase=0, scale=1):
    c = Canvas(scale)
    c.text(28, 26, "LF1", 25, BLUE, True)
    c.text(272, 26, "LF2", 25, GREEN, True)
    c.text(828, 26, "HF target", 25, TEAL, True)
    c.text(1048, 402, "Workflow schematic", 14, MUTED, anchor="rt")
    for x, field, label in zip((28, 272, 828), fields, ("100 × 100", "200 × 200", "800 × 800")):
        c.im.paste(field_tile(field, norm, round(220*scale)), c.xy((x, 84)))
        c.text(x+110, 321, label, 17, MUTED, anchor="mt")
    c.rect((555, 168, 737, 253), fill="#eaf5f5", outline="#a5cfd1", radius=12)
    c.text(646, 194, "NNLCI", 29, TEAL, True, "mt")
    c.arrow((751, 210), (813, 210), TEAL, dashed=True)
    routes = [[(138, 350), (138, 380), (526, 380), (526, 222), (549, 222)],
              [(382, 350), (382, 358), (504, 358), (504, 199), (549, 199)]]
    for points, color in zip(routes, (BLUE, GREEN)):
        c.line(points, "#d4dfe7", 2)
        c.arrow((points[-1][0]-12, points[-1][1]), points[-1], color)
        # A moving pair of local samples makes the route legible without captions.
        segments = [math.dist(a, b) for a, b in zip(points, points[1:])]
        distance = (phase % 1)*sum(segments)
        for a, b, length in zip(points, points[1:], segments):
            if distance <= length:
                t = distance / length
                c.dot(a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), color, 5)
                break
            distance -= length
    # The same physical location is highlighted on both native LF grids.
    # Stencils in this opening are visual sampling cues; the mechanism explains alignment.
    px = .40 + .02*math.sin(2*math.pi*phase)
    py = .40 + .02*math.cos(2*math.pi*phase)
    for x, n, color in ((28, 100, BLUE), (272, 200, GREEN)):
        cx = x+(px-WINDOW[0])/(WINDOW[1]-WINDOW[0])*220
        cy = 84+(WINDOW[3]-py)/(WINDOW[3]-WINDOW[2])*220
        step = 220/(WINDOW[1]-WINDOW[0])/n
        c.rect((cx-step*1.5, cy-step*1.5, cx+step*1.5, cy+step*1.5), outline="white", width=2)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                c.dot(cx+i*step, cy+j*step, "white", 1.8)
        c.dot(cx, cy, color, 3.2)
    # Network activation is a presentation cue, not a predicted field value.
    c.line([(606, 237), (686, 237)], "#c5dedf", 2)
    c.dot(606+80*(phase % 1), 237, TEAL, 3)
    return c


def comparison(fields, norm):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                         "svg.fonttype": "none", "svg.hashsalt": "NNLCI",
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), dpi=200)
    fig.subplots_adjust(left=.065, right=.97, top=.76, bottom=.255, wspace=.25)
    fig.text(.045, .938, "Independent CFD. Three resolutions.", size=22, weight="bold")
    fig.text(.045, .877, "Config 3 · density · nominal t = 0.25 · identical physical window and color scale", color=MUTED, size=12)
    for ax, field, label, color in zip(axes, fields,
            ("LF1 · 100 × 100", "LF2 · 200 × 200", "HF · 800 × 800"), (BLUE, GREEN, TEAL)):
        x, y = field[0, :, 0], field[:, 0, 1]
        ix = (x >= WINDOW[0]-.02) & (x <= WINDOW[1]+.02)
        iy = (y >= WINDOW[2]-.02) & (y <= WINDOW[3]+.02)
        art = ax.pcolormesh(x[ix], y[iy], field[:, :, 2][np.ix_(iy, ix)],
                           cmap="inferno", norm=norm, shading="nearest", rasterized=True)
        ax.set(xlim=WINDOW[:2], ylim=WINDOW[2:], aspect="equal", xlabel="x")
        ax.set_xticks([.3, .4, .5]); ax.set_yticks([.3, .4, .5])
        ax.set_title(label, color=color, size=15, weight="bold", loc="left", pad=10)
    axes[0].set_ylabel("y", rotation=0)
    cbax = fig.add_axes((.32, .13, .36, .025))
    fig.colorbar(art, cax=cbax, orientation="horizontal", label="Density ρ")
    fig.text(.045, .025, "LF: new CPU CFD solves  ·  HF: packaged CFD reference  ·  No NNLCI output shown", size=11, color=MUTED)
    names = []
    for ext in ("png", "svg", "pdf"):
        name = f"config3_independent_resolution.{ext}"
        fig.savefig(OUT / name, facecolor="white")
        if ext == "svg":
            target = OUT / name
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines())+"\n")
        names.append(name)
    plt.close(fig)
    return names


def animate(fields, norm):
    palette = workflow(fields, norm).im.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = []
    for index in range(40):
        frame = workflow(fields, norm, phase=index/40).im
        frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    duration = [90]*40
    duration[0] = 900
    frames[0].save(OUT / "nnlci_local_to_hf.gif", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True, disposal=1)
    return {"file": "nnlci_local_to_hf.gif", "frames": len(frames), "dimensions": [WIDTH, HEIGHT],
            "duration_ms": sum(duration), "meaning": "Sampling and signal highlights only; no physical time evolution, solver iterations or NNLCI predictions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    available = memory_gate()
    if available < 700:
        raise SystemExit("Rendering deferred: less than 700 MiB effective available RAM")
    manifest = json.loads((LF / "provenance.json").read_text())
    paths = [LF / run["field"] for run in manifest["runs"]] + [HF]
    for path, run in zip(paths, manifest["runs"]):
        if sha(path) != run["field_sha256"]:
            raise ValueError(f"CFD provenance mismatch: {path}")
    if sha(HF) != manifest["packaged_hf_reference"]["sha256"]:
        raise ValueError("Packaged HF changed since CFD provenance was recorded")
    fields = [read_field(path) for path in paths]
    if [f.shape for f in fields] != [(101, 101, 6), (201, 201, 6), (801, 801, 6)]:
        raise ValueError("Expected independent 100/200 and packaged 800 interval grids")
    norm = Normalize(min(f[:, :, 2].min() for f in fields), max(f[:, :, 2].max() for f in fields))
    OUT.mkdir(parents=True, exist_ok=True)
    names = comparison(fields, norm) + mechanism()
    for name in ("nnlci_motion_poster.png", "nnlci_local_to_hf_concept.png"):
        workflow(fields, norm, scale=2).save(name)
        names.append(name)
    names += [scientific_panels(fields[-1], True), scientific_panels(fields[-1], False)]
    animation_available = memory_gate()
    if not args.static_only and animation_available >= 1500:
        motion = animate(fields, norm)
        names.append(motion["file"])
        opening = motion["file"]
    else:
        motion = {"deferred": True, "reason": "--static-only" if args.static_only else "Below 1500 MiB effective available memory"}
        opening = "nnlci_motion_poster.png"
        (OUT / "nnlci_local_to_hf.gif").unlink(missing_ok=True)
    readme = ROOT / "README.md"
    readme.write_text(re.sub(r"(\]\(docs/assets/promo/)nnlci_(?:local_to_hf\.gif|motion_poster\.png)(\))",
                            lambda m: m[1]+opening+m[2], readme.read_text(), count=1))
    metadata = {
        "operation": "Render independent LF CFD solves and packaged HF reference; animate workflow highlights only",
        "renderers": {name: sha(ROOT / "scripts" / name) for name in
                      ("render_gatech_promo.py", "render_promo_mechanism.py", "render_config3_contours.py", "cfd_io.py")},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__, "pillow": pillow_version},
        "backend": "CPU: Agg, Pillow, CairoSVG",
        "cfd_provenance": {"path": str((LF / "provenance.json").relative_to(ROOT)), "sha256": sha(LF / "provenance.json")},
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path), "shape": list(field.shape)} for path, field in zip(paths, fields)],
        "opening_and_comparison": {"window_xy": list(WINDOW), "density_scale": [float(norm.vmin), float(norm.vmax)],
            "display": "Identical tile dimensions, crop and global linear color scale. Nearest native node; no HF downsampling to create LF, no smoothing/sharpening.",
            "resolution": "100/200/800 intervals per axis; 101/201/801 nodes", "hf": "Packaged CFD target, not prediction"},
        "hf_contours": {"window_xy": [.24, .54, .22, .52], "source": "Packaged HF/bootstrap source only", "rendering": "80 filled intervals, 15 isolines; standard contour interpolation; full-domain min/max for each variable"},
        "evidence_boundary": "LF1/LF2 now come from independent CPU executions of repository CFD kernels. One case, no retraining, no inference, no GPU parity test. Packaged HF execution log unavailable; nominal setup from standard config. Existing bootstrap and two-epoch checkpoint unchanged.",
        "memory": {"before_render_mib": available, "before_animation_mib": animation_available,
                   "static_threshold_mib": 700, "animation_threshold_mib": 1500,
                   "animation_storage": "40 indexed 1080x430 frames; about 18 MiB raw index storage, no high-resolution frame stack"},
        "animation": motion,
        "outputs": {name: {"sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size} for name in names},
    }
    (OUT / "gatech_provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"outputs": names, "animation": motion}, indent=2))


if __name__ == "__main__":
    main()
