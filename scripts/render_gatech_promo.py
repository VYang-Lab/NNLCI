"""Render independently solved LF fields, an HF target, and the NNLCI workflow.

The POI animation uses actual OpenSCAD meshes and a Graphviz route. Each LF input
has exactly nine stencil locations on a differently resolved source mesh. The
MLP emits one POI, placed in one schematic puzzle cell, then repeats everywhere.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/nnlci-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import __version__ as pillow_version
from cfd_io import read_field, sha
from replot_sod_promo import memory_gate
from render_config3_contours import scientific_panels
from render_promo_mechanism import render as mechanism
from promo_poi import Scene, prepare_tools, render_static, animate, semantics, WIDTH, HEIGHT, FRAME_COUNT, WINDOW

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/promo"
LF = ROOT / "problems/2d_riemann/data/promo_cfd"
HF = ROOT / "problems/2d_riemann/data/raw/flow_standard_config3.dat"
INK, MUTED, BLUE, GREEN, TEAL = "#162536", "#596879", "#2563eb", "#159947", "#087f8c"


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
    tool_report = prepare_tools()
    names += list(tool_report["exports"])
    scene = Scene(fields, norm)
    names += render_static(scene)
    names += [scientific_panels(fields[-1], True), scientific_panels(fields[-1], False)]
    animation_available = memory_gate()
    if not args.static_only and animation_available >= 1500:
        motion = animate(scene)
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
        "operation": "Render two 3x3 LF inputs, MLP to one POI, and one-cell-at-a-time HF-like reconstruction; HF data is comparison-only",
        "renderers": {name: sha(ROOT / "scripts" / name) for name in
                      ("render_gatech_promo.py", "promo_poi.py", "render_promo_mechanism.py", "render_config3_contours.py", "cfd_io.py")},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__, "pillow": pillow_version},
        "backend": "CPU: OpenSCAD, Graphviz, Agg, Pillow, CairoSVG",
        "production_tools": tool_report,
        "poi_workflow": semantics(),
        "cfd_provenance": {"path": str((LF / "provenance.json").relative_to(ROOT)), "sha256": sha(LF / "provenance.json")},
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path), "shape": list(field.shape)} for path, field in zip(paths, fields)],
        "opening_and_comparison": {"window_xy": list(WINDOW), "density_scale": [float(norm.vmin), float(norm.vmax)],
            "display": "Identical tile dimensions, crop and global linear color scale. Nearest native node; no HF downsampling to create LF, no smoothing/sharpening.",
            "resolution": "100/200/800 intervals per axis; 101/201/801 nodes", "hf": "Packaged CFD target, not prediction"},
        "hf_contours": {"window_xy": [.24, .54, .22, .52], "source": "Packaged HF/bootstrap source only", "rendering": "80 filled intervals, 15 isolines; standard contour interpolation; full-domain min/max for each variable"},
        "evidence_boundary": "LF1/LF2 now come from independent CPU executions of repository CFD kernels. One case, no retraining, no inference, no GPU parity test. Packaged HF execution log unavailable; nominal setup from standard config. Existing bootstrap and two-epoch checkpoint unchanged.",
        "memory": {"before_render_mib": available, "before_animation_mib": animation_available,
                   "static_threshold_mib": 700, "animation_threshold_mib": 1500,
                   "animation_storage": f"{FRAME_COUNT+1} indexed {WIDTH}x{HEIGHT} frames; {(FRAME_COUNT+1)*WIDTH*HEIGHT/2**20:.1f} MiB raw index storage; one RGB frame at a time"},
        "animation": motion,
        "outputs": {name: {"sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size} for name in names},
    }
    (OUT / "gatech_provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"outputs": names, "animation": motion}, indent=2))


if __name__ == "__main__":
    main()
