"""Generate static repository figures from canonical CSV result tables."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

BACKGROUND = "#07070d"
FOREGROUND = "#f2efff"
MUTED = "#aaa6bb"
GRID = "#2c2a38"
BLUE = "#89b8ff"
CYAN = "#43e0da"
PURPLE = "#9d78ff"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


TITLE = font(46, bold=True)
LABEL = font(23)
VALUE = font(21, bold=True)
AXIS = font(18)


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def benchmark_figure() -> None:
    rows = sorted(
        read_csv("benchmark_validation.csv"),
        key=lambda row: float(row["val_macro_f1"]),
    )
    image = Image.new("RGB", (1600, 1000), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((80, 50), "Validation Macro F1 by model", font=TITLE, fill=FOREGROUND)

    left, top, right, bottom = 430, 145, 1470, 920
    minimum, maximum = 0.45, 0.90
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9]:
        x = left + (tick - minimum) / (maximum - minimum) * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        text = f"{tick:.1f}"
        draw.text((x - draw.textlength(text, font=AXIS) / 2, bottom + 18), text, font=AXIS, fill=MUTED)

    row_height = (bottom - top) / len(rows)
    for index, row in enumerate(rows):
        value = float(row["val_macro_f1"])
        y0 = top + index * row_height + 9
        y1 = top + (index + 1) * row_height - 9
        width = (value - minimum) / (maximum - minimum) * (right - left)
        color = CYAN if row["model"] == "BiomedBERT" else BLUE
        draw.rounded_rectangle((left, y0, left + width, y1), radius=10, fill=color)
        label_y = (y0 + y1 - 24) / 2
        draw.text((80, label_y), row["model"], font=LABEL, fill=FOREGROUND)
        value_text = f"{value:.4f}"
        draw.text((left + width + 14, label_y + 1), value_text, font=VALUE, fill=FOREGROUND)

    draw.text((left + 320, 958), "Macro F1", font=AXIS, fill=MUTED)
    image.save(FIGURES / "results_macro_f1.png", optimize=True)


def per_class_figure() -> None:
    rows = read_csv("biomedbert_test_per_class.csv")
    names = ["Neoplasms", "Digestive", "Nervous", "Cardiovascular", "General"]
    values = [float(row["f1"]) for row in rows]

    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((80, 50), "BiomedBERT locked-test F1 by class", font=TITLE, fill=FOREGROUND)

    left, top, right, bottom = 150, 160, 1500, 750
    minimum, maximum = 0.70, 0.96
    for tick in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        y = bottom - (tick - minimum) / (maximum - minimum) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((70, y - 11), f"{tick:.2f}", font=AXIS, fill=MUTED)

    available = right - left
    bar_width = 170
    gap = (available - bar_width * len(values)) / (len(values) - 1)
    colors = [CYAN, "#5ddfd5", BLUE, "#64d4ef", PURPLE]
    for index, (name, value) in enumerate(zip(names, values)):
        x0 = left + index * (bar_width + gap)
        x1 = x0 + bar_width
        y0 = bottom - (value - minimum) / (maximum - minimum) * (bottom - top)
        draw.rounded_rectangle((x0, y0, x1, bottom), radius=12, fill=colors[index])
        value_text = f"{value:.4f}"
        draw.text(
            (x0 + (bar_width - draw.textlength(value_text, font=VALUE)) / 2, y0 - 38),
            value_text,
            font=VALUE,
            fill=FOREGROUND,
        )
        label_width = draw.textlength(name, font=LABEL)
        draw.text((x0 + (bar_width - label_width) / 2, bottom + 30), name, font=LABEL, fill=FOREGROUND)

    draw.text((28, 420), "F1", font=AXIS, fill=MUTED)
    image.save(FIGURES / "biomedbert_test_per_class.png", optimize=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    benchmark_figure()
    per_class_figure()
    print(f"Wrote figures to {FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
