"""Editable SVG master plus PDF/PNG exports for pointwise NNLCI reconstruction."""
from pathlib import Path
from html import escape
import cairosvg

from promo_poi import prediction_color

OUT = Path(__file__).resolve().parents[1] / "docs/assets/promo"
INK, MUTED, BLUE, GREEN, TEAL, RED = "#162536", "#596879", "#2563eb", "#159947", "#087f8c", "#c5444d"


def render():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680" role="img" aria-labelledby="title desc">',
             '<title id="title">NNLCI: two 3 by 3 patches, one POI, a field assembled point by point</title>',
             '<desc id="desc">Schematic. LF1 and LF2 each contribute nine aligned sample locations with four variables. LF2 has a finer underlying mesh, not more input points. The 72 features enter an MLP, which predicts four variables only at the center POI. One output is placed in one grid cell; repeat at every target location to reconstruct the field. Puzzle colors are illustrative, not model results. HF labels reach only the training loss.</desc>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#82929f"/></marker></defs>',
             '<rect width="1200" height="680" fill="white"/>']

    def text(x, y, value, size=18, color=INK, weight=400, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

    def rect(x, y, w, h, fill, stroke="none", radius=12, dash="", width=1.5):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def path(d, color="#cbd5df", width=1.5, arrow=False, dash=""):
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

    def dot(x, y, r, color):
        parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    text(36, 35, "NNLCI  /  LOCAL TO GLOBAL", 14, TEAL, 700)
    text(1164, 35, "SCHEMATIC · ILLUSTRATIVE COLORS", 13, MUTED, 500, "end")
    text(36, 83, "3 × 3 in. One point out. Assemble the field.", 32, INK, 700)
    text(36, 118, "LF2 has a finer source mesh; both LF inputs still contain exactly nine locations.", 19, MUTED)
    for x, n, title in [(36, "01", "Two local patches"), (418, "02", "MLP"), (648, "03", "One POI"), (900, "04", "Reconstruct")]:
        text(x, 175, n, 15, TEAL, 700)
        text(x + 32, 175, title, 19, INK, 700)

    rect(36, 198, 315, 251, "#f7f9fc")
    for cx, label, color, mesh_step in [(116, "LF1", BLUE, 30), (270, "LF2", GREEN, 15)]:
        cy = 306
        text(cx, 232, label, 20, color, 700, "middle")
        for k in range(-60, 61, mesh_step):
            path(f"M {cx-60} {cy+k} H {cx+60}", "#dbe2ec", 1)
            path(f"M {cx+k} {cy-60} V {cy+60}", "#dbe2ec", 1)
        rect(cx-40, cy-40, 80, 80, "none", color, 5)
        # The source mesh differs; model feature markers are nine on each.
        parts.append(f'<g id="{label.lower()}-stencil" aria-label="9 input locations">')
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                dot(cx+i*30, cy+j*30, 5 if i or j else 7, color)
        parts.append('</g>')
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="none" stroke="{INK}" stroke-width="1.2"/>')
        text(cx, 395, "3 × 3", 21, color, 700, "middle")
        text(cx, 428, "9 locations", 15, MUTED, anchor="middle")
    path("M 116 458 V 470 H 270 V 458", "#98a7b6", 1.5)
    text(193, 498, "2 × 9 × 4 = 72 features", 19, INK, 700, "middle")
    text(193, 528, "Same POI, variables and time", 15, MUTED, anchor="middle")
    path("M 360 306 H 409", arrow=True)

    rect(418, 255, 159, 102, "#eaf5f5", "#a5cfd1")
    text(497, 299, "NNLCI", 27, TEAL, 700, "middle")
    text(497, 330, "MLP", 18, TEAL, 700, "middle")
    text(497, 395, "72 → 4", 22, TEAL, 700, "middle")
    text(497, 428, "Fixed transforms", 15, MUTED, anchor="middle")
    path("M 586 306 H 645", TEAL, 2, True)

    # One small square is the display cell for ONE output coordinate.
    color = "#%02x%02x%02x" % prediction_color(40)
    rect(662, 286, 40, 40, color, TEAL, 0)
    dot(682, 306, 3, "white")
    text(713, 313, "ρ̂, û, v̂, p̂", 22, TEAL, 500)
    text(730, 395, "Only the center point", 18, INK, 700, "middle")
    text(730, 428, "One cell = one POI", 17, MUTED, anchor="middle")

    # A partly filled puzzle makes insertion visible in a motion-free diagram.
    left, top, step, filled = 900, 219, 26, 40
    for row in range(9):
        for col in range(9):
            rank = row*9+col
            tint = "#%02x%02x%02x" % prediction_color(rank) if rank < filled else "#eef4f5"
            rect(left+col*step+1, top+row*step+1, step-2, step-2, tint, radius=0)
            if rank < filled:
                dot(left+(col+.5)*step, top+(row+.5)*step, 1.5, "#eefaf7")
    tx, ty = left+4*step, top+4*step
    rect(tx, ty, step, step, "none", TEAL, 0, "4 3", 2)
    # The carried cell has the same color and one center marker as the output.
    path(f"M 682 335 V 369 H 864 V {ty+13} H {tx-6}", TEAL, 2, True)
    rect(846, ty+2, 22, 22, color, TEAL, 0)
    dot(857, ty+13, 2, "white")
    text(1017, 484, "HF-like field", 22, TEAL, 700, "middle")
    text(1017, 511, "Illustration · no model output", 15, MUTED, anchor="middle")
    path("M 1050 532 V 550 H 376 V 471", TEAL, 1.5, True, "5 5")
    text(700, 540, "Place at its coordinate; repeat for every target point", 15, TEAL, anchor="middle")

    # Separate training lane. HF never feeds inference or the colored puzzle.
    path("M 36 573 H 1164", "#e2e8ef", 1)
    text(36, 611, "INFERENCE", 13, TEAL, 700)
    text(36, 643, "LF patches → MLP → one POI → field", 17, INK)
    rect(480, 590, 684, 66, "#fff8f8", radius=10)
    text(496, 612, "TRAINING ONLY", 12, RED, 700)
    text(502, 639, "HF label at POI", 17, RED, 500)
    path("M 650 632 H 855", RED, 1.7, True, "5 5")
    rect(870, 611, 91, 36, "white", "#e4b6ba", 6)
    text(915, 636, "Loss", 18, RED, 700, "middle")
    text(1140, 637, "Predicted POI", 16, MUTED, anchor="end")
    path("M 1009 632 H 974", "#94aaaa", 1.5, True, "4 4")
    parts.append("</svg>")
    svg = "\n".join(parts).encode()
    target = OUT / "nnlci_local_mechanism"
    target.with_suffix(".svg").write_bytes(svg)
    cairosvg.svg2png(bytestring=svg, write_to=str(target.with_suffix(".png")), output_width=2400, output_height=1360)
    cairosvg.svg2pdf(bytestring=svg, write_to=str(target.with_suffix(".pdf")))
    return [target.with_suffix(ext).name for ext in (".svg", ".png", ".pdf")]


if __name__ == "__main__":
    render()
