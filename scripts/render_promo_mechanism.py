"""Editable SVG master plus PDF/PNG exports for the local NNLCI mechanism."""
from pathlib import Path
from html import escape
import cairosvg

OUT = Path(__file__).resolve().parents[1] / "docs/assets/promo"
INK, MUTED, BLUE, GREEN, TEAL, RED = "#162536", "#596879", "#2563eb", "#159947", "#087f8c", "#c5444d"


def render():
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="title desc">',
             '<title id="title">NNLCI: from two local CFD neighborhoods to one refined value</title>',
             '<desc id="desc">Conceptual diagram. Independently solved LF fields are aligned at the same point and time. Two three-by-three stencils, each with four variables, form 72 features. Fixed training transforms feed NNLCI. HF connects only to training loss, never to inference.</desc>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#82929f"/></marker></defs>',
             '<rect width="1200" height="630" fill="white"/>']

    def text(x, y, value, size=18, color=INK, weight=400, anchor="start"):
        parts.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')

    def rect(x, y, w, h, fill, stroke="none", radius=12, dash=""):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" stroke-dasharray="{dash}"/>')

    def path(d, color="#cbd5df", width=1.5, arrow=False, dash=""):
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

    def dot(x, y, r, color):
        parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    text(36, 35, "NNLCI  /  LOCAL REFINEMENT", 14, TEAL, 700)
    text(1164, 35, "SCHEMATIC", 13, MUTED, 500, "end")
    text(36, 83, "Two CFD neighborhoods. One refined value.", 32, INK, 700)
    text(36, 118, "Same physical point and time · independently solved LF inputs", 18, MUTED)
    for x, n, title in [(36, "01", "Align & sample"), (440, "02", "Pack local values"), (770, "03", "Predict")]:
        text(x, 166, n, 15, TEAL, 700)
        text(x + 32, 166, title, 19, INK, 700)
    rect(36, 189, 340, 242, "#f7f9fc")
    for cx, cy, label, color, step in [(116, 282, "LF1", BLUE, 30), (285, 282, "LF2", GREEN, 30)]:
        text(cx, 221, label, 18, color, 700, "middle")
        # Identical physical support; denser native LF2 grid underneath the
        # same aligned 3x3 model stencil (the 72-feature contract is unchanged).
        grid_step = step if label == "LF1" else step/2
        count = 2 if label == "LF1" else 4
        for k in range(-count, count+1):
            path(f"M {cx-60} {cy+k*grid_step} H {cx+60}", "#dbe2ec", 1)
            path(f"M {cx+k*grid_step} {cy-60} V {cy+60}", "#dbe2ec", 1)
        rect(cx-step-9, cy-step-9, 2*step+18, 2*step+18, "none", color, 5)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                dot(cx+i*step, cy+j*step, 4 if i or j else 6, color)
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="none" stroke="{INK}" stroke-width="1.2"/>')
    path("M 116 347 V 368 H 285 V 347", "#98a7b6", 1.5)
    text(206, 394, "Aligned location x", 17, MUTED, anchor="middle")
    text(206, 418, "Density · u · v · pressure", 14, MUTED, anchor="middle")
    path("M 390 282 H 426", arrow=True)

    # Every small square represents one scalar: 2 groups × 9 nodes × 4 variables.
    for x, label, color in [(445, "LF1", BLUE), (567, "LF2", GREEN)]:
        text(x+39, 221, label, 15, color, 700, "middle")
        for row in range(9):
            for col in range(4):
                rect(x+col*20, 241+row*11, 16, 7, color, radius=1)
    path("M 445 352 V 360 H 643 V 352", "#98a7b6", 1.5)
    text(544, 389, "72 features", 21, INK, 700, "middle")
    text(544, 418, "2 × 9 points × 4 variables", 14, MUTED, anchor="middle")
    path("M 662 282 H 749", arrow=True)

    rect(770, 225, 162, 114, "#eaf5f5", "#a5cfd1")
    text(851, 277, "NNLCI", 27, TEAL, 700, "middle")
    text(851, 307, "Learned map", 15, TEAL, anchor="middle")
    path("M 944 282 H 986", TEAL, 2, True)
    rect(1006, 250, 146, 64, "#f0f7f7", "#a5cfd1", 8)
    text(1079, 290, "ρ̂, û, v̂, p̂", 24, TEAL, 500, "middle")
    text(1079, 345, "Refined value", 16, INK, 700, "middle")
    text(851, 389, "Fixed training transforms", 15, MUTED, anchor="middle")
    text(970, 418, "Repeat at target locations", 15, MUTED, anchor="middle")

    # Separate, visually quieter training lane. HF terminates only at the loss.
    path("M 36 465 H 1164", "#e2e8ef", 1)
    text(36, 503, "INFERENCE", 13, TEAL, 700)
    text(36, 532, "LF inputs + fixed transforms", 17, INK)
    text(36, 565, "HF supplies training labels only.", 15, MUTED)
    rect(460, 490, 694, 91, "#fff8f8", radius=10)
    text(480, 516, "TRAINING ONLY", 12, RED, 700)
    text(490, 553, "HF label at x", 18, RED, 500)
    path("M 629 545 H 1023", RED, 1.7, True, "5 5")
    rect(1036, 524, 95, 43, "white", "#e4b6ba", 6)
    text(1084, 552, "Loss", 18, RED, 700, "middle")
    path("M 1159 282 H 1178 V 545 H 1135", "#94aaaa", 1.5, True, "4 4")
    text(36, 610, "Equal physical support · aligned 3 × 3 model stencils; native sampling density is shown in the opener.", 13, MUTED)
    parts.append("</svg>")
    svg = "\n".join(parts).encode()
    target = OUT / "nnlci_local_mechanism"
    target.with_suffix(".svg").write_bytes(svg)
    cairosvg.svg2png(bytestring=svg, write_to=str(target.with_suffix(".png")), output_width=2400, output_height=1260)
    cairosvg.svg2pdf(bytestring=svg, write_to=str(target.with_suffix(".pdf")))
    return [target.with_suffix(ext).name for ext in (".svg", ".png", ".pdf")]


if __name__ == "__main__":
    render()
