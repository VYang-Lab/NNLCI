"""Add a centered LF-to-HF arrow label to a comparison image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "docs" / "assets" / "nnlci-lf-hf-comparison.png"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "nnlci-lf-to-hf.png"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def add_transition_label(source: Path, destination: Path) -> None:
    image = Image.open(source).convert("RGBA")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    scale = max(0.75, min(width / 1512, height / 666))
    font_size = max(18, round(25 * scale))
    font = load_font(font_size)

    badge_width = round(154 * scale)
    badge_height = round(58 * scale)
    center_x = width // 2
    center_y = height // 2
    left = center_x - badge_width // 2
    top = center_y - badge_height // 2
    right = left + badge_width
    bottom = top + badge_height
    radius = round(10 * scale)

    shadow_offset = round(3 * scale)
    draw.rounded_rectangle(
        (left + shadow_offset, top + shadow_offset, right + shadow_offset, bottom + shadow_offset),
        radius=radius,
        fill=(15, 23, 42, 45),
    )
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=radius,
        fill=(255, 255, 255, 242),
        outline=(71, 85, 105, 180),
        width=max(1, round(1.5 * scale)),
    )

    color = (15, 23, 42, 255)
    lf_box = draw.textbbox((0, 0), "LF", font=font)
    hf_box = draw.textbbox((0, 0), "HF", font=font)
    lf_width = lf_box[2] - lf_box[0]
    hf_width = hf_box[2] - hf_box[0]
    text_height = max(lf_box[3] - lf_box[1], hf_box[3] - hf_box[1])
    text_y = center_y - text_height // 2 - lf_box[1]

    padding = round(17 * scale)
    lf_x = left + padding
    hf_x = right - padding - hf_width
    draw.text((lf_x, text_y), "LF", font=font, fill=color)
    draw.text((hf_x, text_y), "HF", font=font, fill=color)

    arrow_start = lf_x + lf_width + round(10 * scale)
    arrow_end = hf_x - round(10 * scale)
    line_width = max(2, round(3 * scale))
    head_length = round(9 * scale)
    head_half_height = round(6 * scale)
    draw.line((arrow_start, center_y, arrow_end, center_y), fill=color, width=line_width)
    draw.polygon(
        (
            (arrow_end, center_y),
            (arrow_end - head_length, center_y - head_half_height),
            (arrow_end - head_length, center_y + head_half_height),
        ),
        fill=color,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(destination, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_transition_label(args.input, args.output)
    print(f"Saved: {args.output}")
