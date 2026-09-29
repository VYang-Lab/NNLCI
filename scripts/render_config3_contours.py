"""Numerical contour views of the packaged HF CFD source (not predictions)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/promo"
INK, MUTED, TEAL = "#172033", "#465366", "#087f8c"
WINDOW = (0.24, 0.54, 0.22, 0.52)
DPI = 240


def scientific_panels(field, detail):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(10, 7), dpi=DPI)
    axes = [fig.add_axes((left, .25, .38, .543)) for left in (.075, .57)]
    fig.set_facecolor("white")
    fig.text(.035, .96, "ILLUSTRATIVE FIELD VISUALIZATION  /  CONFIG 3", size=12, color=TEAL, weight="bold")
    fig.text(.035, .901, "2D structure. Up close." if detail else "The complete packaged field.", size=24, weight="bold")
    fig.text(.035, .851, "Packaged CFD bootstrap source · density and pressure", size=13, color=MUTED)
    x, y = field[0, :, 0], field[:, 0, 1]
    for ax, channel, name, cmap in zip(axes, (2, 5), ("Density  ρ", "Pressure  p"), ("inferno", "viridis")):
        all_values = field[:, :, channel]
        norm = Normalize(float(all_values.min()), float(all_values.max()))
        if detail:
            ix = (x >= WINDOW[0]) & (x <= WINDOW[1])
            iy = (y >= WINDOW[2]) & (y <= WINDOW[3])
            xx, yy, z = x[ix], y[iy], all_values[np.ix_(iy, ix)]
        else:
            xx, yy, z = x, y, all_values
        levels = np.linspace(norm.vmin, norm.vmax, 81)
        art = ax.contourf(xx, yy, z, levels=levels, cmap=cmap, norm=norm, antialiased=False)
        ax.contour(xx, yy, z, levels=np.linspace(norm.vmin, norm.vmax, 17)[1:-1],
                   colors="#172033", linewidths=.35, alpha=.32)
        ax.set_aspect("equal")
        ax.set_title(name, loc="left", fontsize=17, weight="bold", pad=13)
        ax.set_xlabel("x", labelpad=1)
        ax.set_ylabel("y", rotation=0, labelpad=8)
        ax.tick_params(labelsize=11, length=3)
        if detail:
            ax.set_xticks([.25, .35, .45, .50])
            ax.set_yticks([.25, .35, .45, .50])
        else:
            ax.set_xticks([0, .5, 1])
            ax.set_yticks([0, .5, 1])
            ax.add_patch(Rectangle((WINDOW[0], WINDOW[2]), WINDOW[1]-WINDOW[0], WINDOW[3]-WINDOW[2],
                                   facecolor="none", edgecolor="white", linewidth=1.5))
        cbax = fig.add_axes((ax.get_position().x0, .139, .38, .021))
        cb = fig.colorbar(art, cax=cbax, orientation="horizontal",
                          ticks=[norm.vmin, (norm.vmin+norm.vmax)/2, norm.vmax], format="%.3f")
        cb.ax.tick_params(labelsize=11, length=2)
        cb.outline.set_visible(False)
    fig.text(.035, .057, "Interaction detail · full-domain color scales" if detail else "White boxes locate the interaction detail", size=12, color=MUTED)
    fig.text(.035, .020, "Bootstrap source visualization; not NNLCI prediction or held-out validation.", size=12, color=MUTED)
    name = "config3_contour_hero.png" if detail else "config3_full_domain.png"
    fig.savefig(OUT / name, dpi=DPI, facecolor="white")
    plt.close(fig)
    return name
