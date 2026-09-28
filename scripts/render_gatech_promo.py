"""Render the visual-first README from packaged data; no CFD or inference.

The prediction array is a coordinate-free sampled table, not a square field.
Scientific panels therefore use the packaged CFD bootstrap source. Concept
panels show subsamples of that SAME field to illustrate the refinement goal.
"""
from __future__ import annotations

import argparse
import hashlib
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
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from replot_sod_promo import memory_gate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/promo"
SOURCE = ROOT / "problems/2d_riemann/data/raw/flow_standard_config3.dat"
PREDICTION = ROOT / "problems/2d_riemann/results/NNLCI_Output/predictions_config3.npy"
INK, MUTED, LIGHT = "#172033", "#465366", "#e4eaf1"
BLUE, GREEN, TEAL, RED = "#2563eb", "#159947", "#087f8c", "#dc2626"
# Physical-coordinate detail, used consistently throughout. Full domain is also exported.
WINDOW = (0.24, 0.54, 0.22, 0.52)
DPI = 240


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_field():
    with SOURCE.open() as stream:
        if stream.readline().startswith("version https://git-lfs"):
            raise RuntimeError("Retrieve the source with git lfs pull before rendering.")
        for line in stream:
            if "ZONE" in line:
                break
        # Fortran sometimes writes a three-digit exponent without an E.
        rows = np.loadtxt((re.sub(r"(\d)([+-]\d{3})", r"\1E\2", s) for s in stream))
    if rows.shape != (801 * 801, 6) or not np.isfinite(rows).all():
        raise ValueError("Expected finite X/Y/density/u/v/pressure on 801 x 801 nodes.")
    field = rows.reshape(801, 801, 6)
    np.testing.assert_allclose(field[:, :, 0], np.broadcast_to(np.linspace(0, 1, 801), (801, 801)))
    np.testing.assert_allclose(field[:, :, 1], np.broadcast_to(np.linspace(0, 1, 801)[:, None], (801, 801)))
    return field


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


class Canvas:
    """Vector-like, high-resolution drawing in 960 px display coordinates."""
    def __init__(self, height=430, scale=2.5):
        self.scale = scale
        self.im = Image.new("RGB", (round(960*scale), round(height*scale)), "white")
        self.d = ImageDraw.Draw(self.im)

    def coords(self, box):
        return tuple(round(v*self.scale) for v in box)

    def text(self, x, y, label, size=18, color=INK, bold=False):
        font = ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", round(size*self.scale))
        self.d.text(self.coords((x, y)), label, font=font, fill=color, anchor="lt")

    def line(self, points, color=LIGHT, width=1):
        self.d.line([self.coords(p) for p in points], fill=color, width=max(1,round(width*self.scale)))

    def rect(self, box, fill=None, outline=LIGHT, width=1, radius=0):
        self.d.rounded_rectangle(self.coords(box), radius=round(radius*self.scale), fill=fill, outline=outline, width=round(width*self.scale))

    def dot(self, x, y, color, r=4):
        self.d.ellipse(self.coords((x-r, y-r, x+r, y+r)), fill=color)

    def arrow(self, start, end, color=INK, width=2):
        self.line([start, end], color, width)
        a = math.atan2(end[1]-start[1], end[0]-start[0])
        x, y = end
        points = [(x, y), (x-9*math.cos(a)+5*math.sin(a), y-9*math.sin(a)-5*math.cos(a)),
                  (x-9*math.cos(a)-5*math.sin(a), y-9*math.sin(a)+5*math.cos(a))]
        self.d.polygon([self.coords(p) for p in points], fill=color)

    def field(self, field, box, stride=1):
        sampled = field[::stride, ::stride]
        x, y = sampled[0, :, 0], sampled[:, 0, 1]
        z = sampled[:, :, 2][np.ix_((y >= WINDOW[2]) & (y <= WINDOW[3]), (x >= WINDOW[0]) & (x <= WINDOW[1]))]
        v = field[:, :, 2]
        # No invented field or learned output: color-map the unchanged CFD values.
        rgba = colormaps["inferno"](Normalize(v.min(), v.max())(z))[::-1]
        image = Image.fromarray((rgba[:, :, :3]*255).astype(np.uint8))
        # Nearest-neighbor display keeps subsample cells visible in the conceptual input tiles.
        image = image.resize((round((box[2]-box[0])*self.scale), round((box[3]-box[1])*self.scale)), Image.Resampling.NEAREST)
        self.im.paste(image, self.coords(box[:2]))

    def save(self, name):
        self.im.save(OUT / name, dpi=(DPI, DPI), optimize=True)


def stencil(c, x, y, step, color):
    c.rect((x-step*1.6, y-step*1.6, x+step*1.6, y+step*1.6), outline="white", width=3)
    for i in (-1, 0, 1):
        c.line([(x-step, y+i*step), (x+step, y+i*step)], "white")
        c.line([(x+i*step, y-step), (x+i*step, y+step)], "white")
        for j in (-1, 0, 1):
            c.dot(x+i*step, y+j*step, color, 2.5)
    c.dot(x, y, "white", 3.5)


def concept(field, active=None, scale=2.5):
    c = Canvas(scale=scale)
    c.text(30, 20, "SCHEMATIC / CONCEPTUAL", 14, TEAL, True)
    title = "Local CFD → high-fidelity-like fields" if active is None else (
        "1. Gather two local neighborhoods",
        "2. Map local inputs to a refined value",
        "3. Assemble a high-fidelity-like field")[active]
    c.text(30, 51, title, 30, bold=True)
    c.text(30, 109, "LOCAL INPUTS", 18, BLUE, True)
    c.text(420, 109, "LEARNED LIFT", 18, TEAL, True)
    c.text(694, 109, "HF-LIKE GOAL", 18, TEAL, True)
    for x, stride, label, color in [(30, 8, "LF1", BLUE), (192, 4, "LF2", GREEN)]:
        c.field(field, (x, 146, x+142, 288), stride)
        c.rect((x,146,x+142,288), outline=color, width=2)
        stencil(c, x+68, 222, 10 if stride == 8 else 7, color)
        c.text(x, 301, label, 20, color, True)
    c.text(30, 337, "Two local neighborhoods", 17)
    c.text(30, 364, "Same location · same time", 15, MUTED)
    c.arrow((350, 219), (404, 219), TEAL)
    c.rect((418, 169, 608, 270), fill="#eef8f8", outline=TEAL, width=2, radius=12)
    c.text(448, 184, "NNLCI", 31, TEAL, True)
    c.text(437, 233, "Local refinement", 17)
    c.text(432, 295, "Repeat across the field", 15, MUTED)
    c.arrow((622, 219), (678, 219), TEAL)
    c.field(field, (694, 146, 930, 382))
    c.rect((694,146,930,382), outline=TEAL, width=2)
    if active is not None:
        boxes = [(22,138,342,294), (412,163,614,276), (688,140,936,388)]
        c.rect(boxes[active], outline=TEAL, width=3, radius=3)
    c.line([(30, 398), (930,398)])
    c.text(30, 411, "Same CFD field, subsampled to illustrate the goal; not a prediction comparison.", 14, MUTED)
    return c


def cover(field, scale=2.5):
    c = Canvas(scale=scale)
    c.text(30, 22, "SCHEMATIC / CONCEPTUAL", 14, TEAL, True)
    c.text(30, 69, "NNLCI", 58, bold=True)
    c.text(32, 157, "Coarse CFD.", 39, bold=True)
    c.text(32, 211, "Finer fields.", 39, TEAL, True)
    c.text(33, 287, "Local inputs → learned fidelity lift", 21)
    c.text(33, 331, "High-fidelity-like reconstruction is the goal.", 16, MUTED)
    c.field(field, (568, 22, 936, 390))
    c.rect((568,22,936,390), outline=LIGHT)
    c.rect((584,34,809,65), fill="white", outline="white", radius=3)
    c.text(595, 43, "REFINEMENT GOAL IMAGERY", 13, TEAL, True)
    c.line([(30, 398), (930,398)])
    c.text(30, 411, "Packaged CFD imagery · conceptual workflow · not time evolution or measured prediction", 14, MUTED)
    return c


def mechanism():
    c = Canvas(height=340)
    c.text(30,20,"SCHEMATIC / CONCEPTUAL",14,TEAL,True)
    c.text(30,51,"Two neighborhoods. One refined value.",30,bold=True)
    for x, color, label in [(84,BLUE,"LF1"),(224,GREEN,"LF2")]:
        c.text(x-31,111,label,18,color,True)
        for i in (-1,0,1):
            c.line([(x-29,172+i*29),(x+29,172+i*29)],LIGHT)
            c.line([(x+i*29,143),(x+i*29,201)],LIGHT)
            for j in (-1,0,1): c.dot(x+i*29,172+j*29,color,5 if (i or j) else 8)
    c.text(31,226,"Same point · aligned stencils",15,MUTED)
    c.arrow((281,172),(335,172))
    c.text(352,113,"TOKEN",18,bold=True)
    for i in range(8): c.rect((350+i*13,147,359+i*13,197),fill=BLUE if i<4 else GREEN,outline=None,radius=2)
    c.text(350,220,"LF1 | LF2",16,MUTED)
    c.arrow((469,172),(521,172))
    c.rect((535,138,686,204),fill="#eef8f8",outline=TEAL,radius=9,width=2)
    c.text(562,157,"NNLCI",26,TEAL,True)
    c.arrow((701,172),(758,172),TEAL)
    c.dot(787,172,TEAL,12)
    c.text(761,113,"PREDICTION",18,TEAL,True)
    c.text(761,220,"HF-like value",16,MUTED)
    c.rect((350,276,711,319),fill="#fff6f6",outline="#f1cccc",radius=6)
    c.text(365,289,"HF supervisor · training only",18,RED,True)
    c.arrow((718,297),(760,297),RED)
    c.text(782,288,"Loss",19,bold=True)
    c.arrow((825,246),(825,274),TEAL)
    c.text(31,283,"At inference:",16,bold=True)
    c.text(31,307,"LF inputs + fixed transforms",15,MUTED)
    c.save("nnlci_local_mechanism.png")


def animation(field):
    # Small indexed frames: hold story beats; fade through white to avoid ghost text.
    opening = cover(field, scale=1).im
    still = concept(field, scale=1).im
    palette_source = Image.new("RGB",(960,860))
    palette_source.paste(opening,(0,0)); palette_source.paste(still,(0,430))
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames, durations = [], []
    def add(im, duration):
        frames.append(im.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(duration)
    def transition(start, end):
        white = Image.new("RGB", start.size, "white")
        for n in range(1,5): add(Image.blend(start,white,n/5),70)
        for n in range(1,5): add(Image.blend(white,end,n/4),70)
    add(opening,1600)
    transition(opening,still)
    for active in range(3):
        c = concept(field,active=active,scale=1)
        add(c.im,1000)
    add(still,1800)
    transition(still,opening)
    frames[0].save(OUT/"nnlci_local_to_hf.gif",save_all=True,append_images=frames[1:],
                   duration=durations,loop=0,optimize=True,disposal=1)
    return {"file":"nnlci_local_to_hf.gif","dimensions":[960,430],"frames":len(frames),
            "duration_ms":sum(durations),"meaning":"Conceptual presentation sequence; highlight inputs, map, and goal. No physical time or iterative solver steps."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only",action="store_true",help="Use permanent PNG instead of GIF in README.")
    args=parser.parse_args()
    available=memory_gate()
    OUT.mkdir(parents=True,exist_ok=True)
    field=read_field()
    names=[scientific_panels(field,True),scientific_panels(field,False)]
    cover(field).save("nnlci_motion_poster.png")
    concept(field).save("nnlci_local_to_hf_concept.png")
    mechanism()
    names += ["nnlci_motion_poster.png","nnlci_local_to_hf_concept.png","nnlci_local_mechanism.png"]
    # Recheck after the 240-dpi panels, before allocating animation frames.
    animation_available=memory_gate()
    if not args.static_only and animation_available >= 2500:
        motion=animation(field)
        names.append(motion["file"])
        open_name=motion["file"]
    else:
        motion={"deferred":True,"reason":"--static-only requested" if args.static_only else "Effective available RAM below 2500 MiB"}
        open_name="nnlci_motion_poster.png"
        (OUT/"nnlci_local_to_hf.gif").unlink(missing_ok=True)
    # Keep the opening valid after a memory-gated regeneration.
    readme=ROOT/"README.md"
    text=readme.read_text()
    text=re.sub(r"(\]\(docs/assets/promo/)nnlci_(?:local_to_hf\.gif|motion_poster\.png)(\))",
                lambda m:m[1]+open_name+m[2],text,count=1)
    readme.write_text(text)
    prediction_meta={"used_for_contours":False,"reason":"Sampled training rows have no saved spatial coordinates; do not reshape or interpolate into a purported prediction field."}
    if PREDICTION.stat().st_size>1000:
        pred=np.load(PREDICTION,mmap_mode="r")
        prediction_meta.update({"path":str(PREDICTION.relative_to(ROOT)),"sha256":sha(PREDICTION),
                                "shape":list(pred.shape),"density_range":[float(pred[:,0].min()),float(pred[:,0].max())]})
    metadata={
        "operation":"Read-only rendering of packaged CFD values and explicitly conceptual workflow artwork. No training, CFD solve, or inference.",
        "renderer":"scripts/render_gatech_promo.py","renderer_sha256":sha(Path(__file__)),
        "versions":{"python":sys.version.split()[0],"numpy":np.__version__,"matplotlib":matplotlib.__version__,"pillow":pillow_version},
        "backend":"Agg / Pillow, CPU", "source":{"path":str(SOURCE.relative_to(ROOT)),"sha256":sha(SOURCE),
        "shape":[801,801,6],"columns":["x","y","density","u","v","pressure"],
        "coordinate_check":"X varies along columns; Y along rows; uniform 0..1, 801 nodes; increasing Y plotted upward",
        "identity":"Packaged standard Config 3 CFD field from which bootstrap grids were subsampled; not NNLCI output"},
        "scientific_panels":{"classification":"Illustrative field visualization / bootstrap source",
        "detail_window_xy":list(WINDOW),"full_domain_export":"config3_full_domain.png",
        "density":{"range":[float(field[:,:,2].min()),float(field[:,:,2].max())],"colormap":"inferno"},
        "pressure":{"range":[float(field[:,:,5].min()),float(field[:,:,5].max())],"colormap":"viridis"},
        "normalization":"Linear, complete source min/max per variable; same scales in detail and full domain",
        "rendering":"80 filled contour intervals and 15 isolines; standard contour interpolation only; no smoothing, field fabrication, or numerical modification",
        "dimensions":[2400,1680],"dpi":DPI},
        "concept_assets":{"classification":"Schematic / Conceptual",
        "field_imagery":"Same source density detail; LF1/LF2 artwork subsamples rows and columns with strides 8/4 (matching bootstrap sampling); goal uses stride 1. Nearest-neighbor display. This is not an LF/prediction/HF scientific comparison.",
        "colorbar":"None on concept artwork; numerical colorbars only on scientific panels",
        "mechanism":"Two aligned 3x3 neighborhoods depicted schematically, concatenated LF token, local map, refined value. HF supervisor connects only to training loss; inference uses LF and fixed transforms."},
        "excluded_prediction_array":prediction_meta,
        "evidence_boundary":"Two-epoch Config 3 bootstrap uses subsamples of one CFD solution; default evaluation reads training arrays. Neither these figures nor the animation establish held-out reconstruction, conservation, stability, generalization, or speedup.",
        "memory":{"before_render_mib":available,"before_animation_mib":animation_available,"animation_threshold_mib":2500},
        "animation":motion,"outputs":{name:{"sha256":sha(OUT/name),"bytes":(OUT/name).stat().st_size} for name in names}}
    (OUT/"gatech_provenance.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"outputs":names,"animation":motion,"available_mib":available},indent=2))


if __name__=="__main__":
    main()
