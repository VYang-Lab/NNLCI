"""Generate the public NNLCI training-token construction diagram."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "assets" / "nnlci-training-tokenization.png"

CANVAS = (1600, 1120)
INK = "#172033"
MUTED = "#5F6B7A"
GRID = "#C7CED8"
PANEL = "#F8FAFC"
BLUE = "#2563EB"
GREEN = "#159947"
RED = "#DC2626"
TEAL = "#087EA4"
AMBER = "#D97706"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, text_font, fill=INK) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2 - box[1]), text, font=text_font, fill=fill)


def rounded_box(draw: ImageDraw.ImageDraw, box, fill, outline=GRID, radius=16, width=2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start, end, color=INK, width=4, head=14) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - head * math.cos(angle) + head * 0.55 * math.sin(angle),
        end[1] - head * math.sin(angle) - head * 0.55 * math.cos(angle),
    )
    right = (
        end[0] - head * math.cos(angle) - head * 0.55 * math.sin(angle),
        end[1] - head * math.sin(angle) + head * 0.55 * math.cos(angle),
    )
    draw.polygon((end, left, right), fill=color)


def jittered_mesh(cols: int, rows: int, box, seed: int):
    left, top, right, bottom = box
    random.seed(seed)
    dx = (right - left) / (cols - 1)
    dy = (bottom - top) / (rows - 1)
    points = []
    for row in range(rows):
        current = []
        for col in range(cols):
            x = left + col * dx
            y = top + row * dy
            if 0 < col < cols - 1:
                x += random.uniform(-0.20, 0.20) * dx
            if 0 < row < rows - 1:
                y += random.uniform(-0.20, 0.20) * dy
            current.append((x, y))
        points.append(current)
    return points


def draw_mesh(draw: ImageDraw.ImageDraw, points, point_color, point_radius=2) -> None:
    rows = len(points)
    cols = len(points[0])
    for row in range(rows):
        for col in range(cols - 1):
            draw.line((points[row][col], points[row][col + 1]), fill=GRID, width=1)
    for row in range(rows - 1):
        for col in range(cols):
            draw.line((points[row][col], points[row + 1][col]), fill=GRID, width=1)
    for row in range(rows - 1):
        for col in range(cols - 1):
            if (row + col) % 2 == 0:
                draw.line((points[row][col], points[row + 1][col + 1]), fill=GRID, width=1)
            else:
                draw.line((points[row][col + 1], points[row + 1][col]), fill=GRID, width=1)
    for row in points:
        for x, y in row:
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)


def highlight_stencil(draw: ImageDraw.ImageDraw, points, row: int, col: int, color: str) -> tuple[float, float]:
    selected = []
    for rr in range(max(0, row - 1), min(len(points), row + 2)):
        for cc in range(max(0, col - 1), min(len(points[0]), col + 2)):
            selected.append(points[rr][cc])
    xs = [point[0] for point in selected]
    ys = [point[1] for point in selected]
    pad = 16
    draw.rounded_rectangle(
        (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad),
        radius=10,
        outline=color,
        width=4,
    )
    for x, y in selected:
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=INK, outline="white", width=2)
    center = points[row][col]
    draw.ellipse((center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10), fill=RED, outline="white", width=3)
    return center


def panel(draw, box, title, subtitle, cols, rows, seed, point_color, stencil=True):
    rounded_box(draw, box, fill=PANEL)
    left, top, right, bottom = box
    centered_text(draw, ((left + right) // 2, top + 34), title, font(27, bold=True))
    centered_text(draw, ((left + right) // 2, top + 68), subtitle, font(19), fill=MUTED)
    mesh_box = (left + 24, top + 96, right - 24, bottom - 24)
    points = jittered_mesh(cols, rows, mesh_box, seed)
    draw_mesh(draw, points, point_color)
    if stencil:
        center = highlight_stencil(draw, points, round((rows - 1) * 0.57), round((cols - 1) * 0.58), point_color)
    else:
        rr = round((rows - 1) * 0.57)
        cc = round((cols - 1) * 0.58)
        center = points[rr][cc]
        draw.ellipse((center[0] - 11, center[1] - 11, center[0] + 11, center[1] + 11), fill=RED, outline="white", width=3)
    return center


def build_diagram() -> Image.Image:
    image = Image.new("RGB", CANVAS, "white")
    draw = ImageDraw.Draw(image)

    centered_text(draw, (800, 48), "Constructing One NNLCI Training Sample", font(38, bold=True))
    centered_text(
        draw,
        (800, 91),
        "Low-fidelity neighborhoods form the input token; high fidelity supplies the supervisor.",
        font(22),
        fill=MUTED,
    )

    lf1_box = (65, 155, 505, 555)
    lf2_box = (580, 155, 1020, 555)
    hf_box = (1095, 155, 1535, 555)
    lf1_center = panel(draw, lf1_box, "LF Level 1", "Coarse, inexpensive field", 7, 6, 11, BLUE)
    lf2_center = panel(draw, lf2_box, "LF Level 2", "Richer low-cost field", 10, 8, 17, GREEN)
    hf_center = panel(draw, hf_box, "HF Supervisor", "Training only", 14, 11, 23, RED, stencil=False)

    centered_text(draw, (lf1_center[0], lf1_center[1] - 28), "same POI", font(16, bold=True), fill=RED)
    centered_text(draw, (lf2_center[0], lf2_center[1] - 28), "same POI", font(16, bold=True), fill=RED)
    centered_text(draw, (hf_center[0], hf_center[1] - 28), "y_i = U_HF(x_i)", font(16, bold=True), fill=RED)

    token_box = (300, 655, 860, 800)
    rounded_box(draw, token_box, fill="#EFF6FF", outline=BLUE, radius=18, width=3)
    centered_text(draw, (580, 688), "LF Input Token  X_i", font(28, bold=True), fill=BLUE)
    centered_text(draw, (580, 731), "[ LF1 local values  |  LF2 local values ]", font(23), fill=INK)
    centered_text(draw, (580, 768), "Point-centered physics features", font(18), fill=MUTED)

    arrow(draw, (lf1_center[0], 570), (445, 650), color=BLUE, width=4)
    arrow(draw, (lf2_center[0], 570), (710, 650), color=GREEN, width=4)

    model_box = (930, 655, 1260, 800)
    rounded_box(draw, model_box, fill="#F3F4F6", outline=INK, radius=18, width=3)
    centered_text(draw, (1095, 694), "Local Regressor", font(27, bold=True))
    centered_text(draw, (1095, 737), "f_theta(X_i)", font(25), fill=INK)
    centered_text(draw, (1095, 772), "MLP or residual MLP", font(18), fill=MUTED)
    arrow(draw, (865, 728), (925, 728), color=INK, width=4)

    pred_box = (1310, 675, 1515, 780)
    rounded_box(draw, pred_box, fill="#ECFDF5", outline=GREEN, radius=16, width=3)
    centered_text(draw, (1412, 711), "Prediction", font(22, bold=True), fill=GREEN)
    centered_text(draw, (1412, 753), "y_hat_i", font(27), fill=INK)
    arrow(draw, (1265, 728), (1305, 728), color=GREEN, width=4)

    loss_box = (1020, 920, 1450, 1035)
    rounded_box(draw, loss_box, fill="#FFF7ED", outline=AMBER, radius=18, width=3)
    centered_text(draw, (1235, 953), "Supervised Regression Loss", font(25, bold=True), fill=AMBER)
    centered_text(draw, (1235, 997), "MSE(y_hat_i, y_i)", font(25), fill=INK)

    arrow(draw, (1412, 785), (1325, 915), color=GREEN, width=4)
    draw.line((hf_center[0], 570, 1560, 570, 1560, 870, 1390, 915), fill=RED, width=4)
    arrow(draw, (1560, 870), (1390, 915), color=RED, width=4)
    centered_text(draw, (1510, 850), "HF label", font(18, bold=True), fill=RED)

    centered_text(
        draw,
        (620, 985),
        "Repeat across selected points and representative cases",
        font(23, bold=True),
        fill=TEAL,
    )

    return image


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    diagram = build_diagram()
    diagram.save(OUTPUT_PATH, optimize=True)
    print(f"Saved: {OUTPUT_PATH}")
