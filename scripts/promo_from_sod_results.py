"""Render real Sod checkpoint density comparisons; no CFD solves or training.

Run with MPLBACKEND=Agg python scripts/promo_from_sod_results.py.
Preprocessing, feature order, architecture and boundaries mirror Sod predict.py.
"""

from pathlib import Path
import glob
import hashlib
import json
import argparse
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d
import torch
from torch import nn
from numpy.lib.stride_tricks import sliding_window_view


ROOT = Path(__file__).resolve().parents[1]
PROBLEM = ROOT / "problems/1d_sod"
DATA = PROBLEM / "data/raw"
CHECKPOINT = PROBLEM / "models/best_model_sod_final.pth"
OUT = ROOT / "docs/assets/promo"
CASES = ("P1.25", "P2.25", "P3.75", "P4.25", "P5.50", "P6.00")
X = np.linspace(0, 1, 800)
SUBTITLE = ("Sod P1.25 · best_model_sod_final.pth · fidelity lift from "
            "packaged demo (not a speedup claim)")
COLORS = ("#bd741b", "#087f8c", "#23334d")
LABELS = ("LF1 · 50 points", "NNLCI · LF1 + LF2 inputs", "HF · 800 points")
LOW_MEMORY = False


def memory_gate():
    """Check before every plot/GIF; any low reading disables animation."""
    global LOW_MEMORY
    report = subprocess.check_output(["free", "-m"], text=True)
    print(report, flush=True)
    available = int(next(line for line in report.splitlines()
                         if line.startswith("Mem:")).split()[-1])
    # Container limits can be tighter than the host value reported by free.
    limit_path = Path("/sys/fs/cgroup/memory.max")
    usage_path = Path("/sys/fs/cgroup/memory.current")
    if limit_path.exists() and usage_path.exists():
        limit = limit_path.read_text().strip()
        if limit != "max":
            available = min(available, (int(limit) - int(usage_path.read_text())) // 2**20)
    print(f"Effective available memory: {available} MiB", flush=True)
    LOW_MEMORY |= available < 2500
    return not LOW_MEMORY


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [18] + [400] * 8 + [3]
        self.actfunc = nn.GELU()
        self.linears = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:]))

    def forward(self, values):
        for layer in self.linears[:-1]:
            values = self.actfunc(layer(values))
        return self.linears[-1](values)


def read_field(path):
    return pd.read_csv(path, sep=r"\s+")[["den", "vel", "pres"]].values


def align(values):
    source_x = np.linspace(0, 1, len(values))
    return np.column_stack([
        interp1d(source_x, values[:, v], kind="linear",
                 fill_value="extrapolate")(X) for v in range(3)])


def get_scalers():
    files = glob.glob(str(DATA / "train_P*_800.dat"))
    if not files:
        raise FileNotFoundError("No Sod training data for z-score parameters")
    fields = np.array([align(read_field(
        DATA / f"train_{Path(f).name.split('_')[1]}_50.dat")) for f in files])
    return fields.mean(axis=(0, 1)), fields.std(axis=(0, 1))


def predict_case(model, mean, std, case):
    lf1 = align(read_field(DATA / f"pred_{case}_50.dat"))
    lf2 = align(read_field(DATA / f"pred_{case}_100.dat"))
    windows = [sliding_window_view((field - mean) / (std + 1e-6),
                                  window_shape=3, axis=0).reshape(-1, 9)
               for field in (lf1, lf2)]
    inputs = np.concatenate(windows, axis=1)
    with torch.inference_mode():
        inner = model(torch.FloatTensor(inputs)).numpy() * (std + 1e-6) + mean
    prediction = lf1.copy()  # predict.py uses LF1 at both boundary points.
    prediction[1:-1] = inner
    hf = read_field(DATA / f"pred_{case}_800.dat")
    densities = (lf1[:, 0], prediction[:, 0], hf[:, 0])
    assert all(a.shape == (800,) and np.isfinite(a).all() for a in densities)
    return densities


def limits(fields):
    low, high = np.min(fields), np.max(fields)
    return low - .06 * (high - low), high + .08 * (high - low)


def style_axis(ax, ylim):
    ax.set(xlim=(0, 1), ylim=ylim, xlabel="Position x")
    ax.grid(alpha=.16)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, filename):
    fig.savefig(OUT / filename, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUT / filename}")


def triptych(fields, filename, hero=False):
    memory_gate()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=120,
                             sharex=True, sharey=True)
    for ax, values, color, label in zip(axes, fields, COLORS, LABELS):
        ax.plot(X, values, color=color, lw=2)
        ax.set_title(label, fontsize=11, fontweight="bold")
        style_axis(ax, limits(fields))
    axes[0].set_ylabel("Density")
    fig.suptitle("From coarse flow to learned refinement" if hero else
                 "LF → NNLCI → HF · comparison sequence", fontsize=17,
                 fontweight="bold", y=.97)
    fig.text(.5, .86, SUBTITLE, ha="center", fontsize=9)
    fig.text(.5, .025, "Same case, coordinates and density scale · "
             "HF is the reference · not iterative solver steps",
             ha="center", fontsize=9)
    fig.subplots_adjust(top=.73, bottom=.18, left=.065, right=.985, wspace=.12)
    save(fig, filename)


def overlay(fields):
    memory_gate()
    fig, (ax, error_ax) = plt.subplots(
        2, 1, figsize=(9, 6.4), dpi=140, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    for values, color, label, ls in zip(fields, COLORS, LABELS, ("--", "-", ":")):
        ax.plot(X, values, color=color, label=label, ls=ls, lw=2)
    style_axis(ax, limits(fields))
    ax.set(xlabel="", ylabel="Density", title="Sod P1.25 · field and reference error")
    ax.legend(frameon=False)
    for values, color, label in zip(fields[:2], COLORS[:2], LABELS[:2]):
        error_ax.plot(X, values - fields[2], color=color, label=label, lw=1.5)
    error_ax.axhline(0, color=COLORS[2], lw=.8)
    errors = [values - fields[2] for values in fields[:2]]
    extent = max(np.max(np.abs(errors)), 1e-12) * 1.1
    style_axis(error_ax, (-extent, extent))
    error_ax.set_ylabel("Density − HF")
    fig.text(.5, .025, "Signed pointwise differences on the same 800-point grid · "
             "boundary predictions copied from LF1", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, "sod_P1.25_density_overlay.png")


def sequence(fields):
    gif = OUT / "sod_P1.25_fidelity_lift.gif"
    frames = []
    for i, (values, color, label) in enumerate(zip(fields, COLORS, LABELS), 1):
        if not memory_gate():
            break
        fig, ax = plt.subplots(figsize=(6.4, 4), dpi=100)
        ax.plot(X, values, color=color, lw=2.3)
        style_axis(ax, limits(fields))
        ax.set(ylabel="Density", title=f"{i}/3   {label}")
        fig.suptitle("Sod P1.25 · LF → NNLCI → HF", fontweight="bold")
        fig.text(.5, .045, "Comparison frames, not solver iterations · no speedup claim",
                 ha="center", fontsize=8)
        fig.text(.5, .012, "Checkpoint: best_model_sod_final.pth", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .075, 1, .94))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))
        plt.close(fig)
    if memory_gate() and len(frames) == 3:
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=[1600, 1600, 1600], loop=0)
        print(f"Wrote {gif} (3 frames, 640 × 400)")
    else:
        for frame in frames:
            frame.close()
        frames.clear()
        gif.unlink(missing_ok=True)
        print("Memory below 2500 MiB: GIF skipped; permanent static strip is available.")
    for frame in frames:
        frame.close()


def sweep(results):
    memory_gate()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), dpi=130, sharex=True, sharey=True)
    ylim = limits(list(results.values()))
    for ax, (case, fields) in zip(axes.flat, results.items()):
        for values, color, label, ls in zip(fields, COLORS, LABELS, ("--", "-", ":")):
            ax.plot(X, values, color=color, label=label, ls=ls, lw=1.7)
        style_axis(ax, ylim)
        status = "inside training pressure range" if float(case[1:]) <= 5 else "beyond training pressure range"
        ax.set_title(f"Sod {case}\n{status}", fontweight="bold", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Density")
    fig.suptitle("Packaged Sod prediction cases · density fidelity comparison",
                 fontsize=15, fontweight="bold")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(.5, .035))
    fig.text(.5, .015, "best_model_sod_final.pth · LF1/LF2: 50/100 points → "
             "800-point target · no speedup claim", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .94))
    save(fig, "sod_pred_density_sweep.png")


def write_provenance():
    sources = sorted(DATA.glob("train_P*_50.dat")) + sorted(DATA.glob("train_P*_800.dat"))
    sources += [DATA / f"pred_{case}_{grid}.dat" for case in CASES for grid in (50, 100, 800)]
    sources += [CHECKPOINT]
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "script": "scripts/promo_from_sod_results.py",
        "cases": list(CASES),
        "variables": ["density"],
        "target_points": len(X),
        "lf_points": [50, 100],
        "boundary_rule": "First and last predictions copied from aligned LF1",
        "normalization": "Training LF1 mean/std; epsilon 1e-6",
        "interpretation": "Checkpoint inference comparisons; no timing benchmark",
        "versions": {"numpy": np.__version__, "torch": torch.__version__,
                     "matplotlib": matplotlib.__version__},
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    (OUT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true",
                        help="Render PNGs and arrays; skip and remove any stale GIF")
    args = parser.parse_args()
    memory_gate()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = OUT / "arrays"
    arrays.mkdir(exist_ok=True)
    model = NeuralNet().eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    mean, std = get_scalers()
    results = {case: predict_case(model, mean, std, case) for case in CASES}
    fields = results["P1.25"]
    np.save(arrays / "x.npy", X)
    for name, values in zip(("lf1", "nnlci", "hf"), fields):
        np.save(arrays / f"sod_P1.25_{name}_density.npy", values)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelcolor": "#23334d",
                         "text.color": "#23334d", "figure.facecolor": "#fafcfe"})
    triptych(fields, "sod_P1.25_fidelity_lift_triptych.png", hero=True)
    triptych(fields, "sod_P1.25_fidelity_lift_strip.png")
    overlay(fields)
    sweep(results)
    if args.static_only:
        (OUT / "sod_P1.25_fidelity_lift.gif").unlink(missing_ok=True)
        print("Static-only requested: PNG strip available; GIF skipped.")
    else:
        sequence(fields)
    write_provenance()


if __name__ == "__main__":
    main()
