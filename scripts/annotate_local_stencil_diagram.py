"""Annotate the NNLCI local-stencil figure for the public README."""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "docs" / "assets" / "nnlci-local-stencil-base.png"
OUTPUT = REPO_ROOT / "docs" / "assets" / "nnlci-local-stencil-explained.png"

INK = "#172033"
MUTED = "#586474"
BLUE = "#2563EB"
GREEN = "#159947"
RED = "#DC2626"
AMBER = "#D97706"


def load_font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_callout(draw, box, number, title, body, color):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill="#F8FAFC", outline=color, width=3)
    circle_x = left + 38
    circle_y = top + 39
    draw.ellipse((circle_x - 22, circle_y - 22, circle_x + 22, circle_y + 22), fill=color)
    number_font = load_font(23, bold=True)
    number_box = draw.textbbox((0, 0), str(number), font=number_font)
    draw.text(
        (circle_x - (number_box[2] - number_box[0]) / 2, circle_y - (number_box[3] - number_box[1]) / 2 - number_box[1]),
        str(number),
        font=number_font,
        fill="white",
    )
    draw.text((left + 75, top + 18), title, font=load_font(25, bold=True), fill=color)
    lines = textwrap.wrap(body, width=43)
    y = top + 63
    for line in lines:
        draw.text((left + 22, y), line, font=load_font(20), fill=INK)
        y += 29


def main():
    base = Image.open(SOURCE).convert("RGBA")
    figure = Image.new("RGBA", (1320, base.height), "white")
    figure.alpha_composite(base, (0, 0))
    draw = ImageDraw.Draw(figure)

    draw.text((655, 34), "How One NNLCI Training Pair Is Built", font=load_font(34, bold=True), fill=INK)
    draw.text(
        (655, 82),
        "LF creates the model input. HF supplies supervision only.",
        font=load_font(21),
        fill=MUTED,
    )

    first = (650, 140, 1285, 330)
    second = (650, 375, 1285, 595)
    third = (650, 640, 1285, 875)

    draw_callout(
        draw,
        first,
        1,
        "Select the HF supervisor",
        "During training, select a point of interest xᵢ on the high-fidelity mesh. Its HF value provides the target yᵢ = Uᴴᶠ(xᵢ).",
        RED,
    )
    draw_callout(
        draw,
        second,
        2,
        "Extract field information",
        "Collect neighboring LF quantities, such as pressure, density, and velocity along the x- and y-axes. Interpolate and organize them at every point of the local patch.",
        BLUE,
    )
    draw_callout(
        draw,
        third,
        3,
        "Construct the training pair",
        "Concatenate the LF local values into the input token Xᵢ. Train the neural network so that fθ(Xᵢ) approaches yᵢ.",
        AMBER,
    )

    label = "Inference uses Xᵢ only; no HF field is required."
    label_font = load_font(18, bold=True)
    text_box = draw.textbbox((0, 0), label, font=label_font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    padding_x = 22
    padding_y = 12
    center_x = (650 + 1285) // 2
    footer = (
        center_x - text_width // 2 - padding_x,
        892,
        center_x + text_width // 2 + padding_x,
        892 + text_height + 2 * padding_y,
    )
    draw.rounded_rectangle(footer, radius=13, fill="#ECFDF5", outline=GREEN, width=2)
    draw.text(
        (footer[0] + padding_x - text_box[0], footer[1] + padding_y - text_box[1]),
        label,
        font=label_font,
        fill=GREEN,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.convert("RGB").save(OUTPUT, optimize=True)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
