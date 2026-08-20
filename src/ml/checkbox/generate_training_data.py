from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SEED = 20260820
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path("data/ml_training/checkbox")
IMAGE_ROOT = ROOT / "images"
MANIFEST_PATH = ROOT / "manifest.csv"

TRAIN_GROUPS = 240
VALIDATION_GROUPS = 30
TEST_GROUPS = 30
VARIATIONS_PER_GROUP = 8
IMAGE_SIZE = 32


def get_split(group_index: int) -> str:
    if group_index < TRAIN_GROUPS:
        return "train"

    if group_index < TRAIN_GROUPS + VALIDATION_GROUPS:
        return "validation"

    return "test"


def render_checkbox(
    checked: bool,
) -> Image.Image:
    background = random.randint(238, 255)

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=background,
    )

    draw = ImageDraw.Draw(image)

    margin = random.randint(5, 8)
    line_width = random.randint(1, 2)
    foreground = random.randint(0, 45)

    draw.rectangle(
        (
            margin,
            margin,
            IMAGE_SIZE - margin,
            IMAGE_SIZE - margin,
        ),
        outline=foreground,
        width=line_width,
    )

    if checked:
        mark_type = random.choice(
            ["tick", "cross", "filled"]
        )

        if mark_type == "tick":
            draw.line(
                (
                    margin + 3,
                    IMAGE_SIZE // 2,
                    IMAGE_SIZE // 2 - 1,
                    IMAGE_SIZE - margin - 3,
                ),
                fill=foreground,
                width=random.randint(2, 3),
            )

            draw.line(
                (
                    IMAGE_SIZE // 2 - 1,
                    IMAGE_SIZE - margin - 3,
                    IMAGE_SIZE - margin - 2,
                    margin + 3,
                ),
                fill=foreground,
                width=random.randint(2, 3),
            )

        elif mark_type == "cross":
            draw.line(
                (
                    margin + 3,
                    margin + 3,
                    IMAGE_SIZE - margin - 3,
                    IMAGE_SIZE - margin - 3,
                ),
                fill=foreground,
                width=random.randint(2, 3),
            )

            draw.line(
                (
                    IMAGE_SIZE - margin - 3,
                    margin + 3,
                    margin + 3,
                    IMAGE_SIZE - margin - 3,
                ),
                fill=foreground,
                width=random.randint(2, 3),
            )

        else:
            draw.rectangle(
                (
                    margin + 3,
                    margin + 3,
                    IMAGE_SIZE - margin - 3,
                    IMAGE_SIZE - margin - 3,
                ),
                fill=foreground,
            )

    if random.random() < 0.5:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.1, 0.6)
            )
        )

    array = np.asarray(image).astype(np.int16)

    noise = np.random.normal(
        0,
        random.uniform(0.0, 8.0),
        array.shape,
    )

    array = np.clip(
        array + noise,
        0,
        255,
    ).astype(np.uint8)

    image = Image.fromarray(array, mode="L")

    if random.random() < 0.4:
        image = image.rotate(
            random.uniform(-2.0, 2.0),
            resample=Image.Resampling.BILINEAR,
            fillcolor=background,
        )

    return image


def main() -> None:
    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_path in IMAGE_ROOT.glob("*.png"):
        image_path.unlink()

    total_groups = (
        TRAIN_GROUPS
        + VALIDATION_GROUPS
        + TEST_GROUPS
    )

    rows = []

    for group_index in range(total_groups):
        split = get_split(group_index)
        checked = bool(group_index % 2)
        label = 1 if checked else 0

        for variation_index in range(
            VARIATIONS_PER_GROUP
        ):
            image = render_checkbox(
                checked=checked
            )

            file_name = (
                f"{split}_"
                f"{group_index:04d}_"
                f"{variation_index:02d}.png"
            )

            image.save(
                IMAGE_ROOT / file_name,
                format="PNG",
            )

            rows.append(
                {
                    "image_path": (
                        Path("images") / file_name
                    ).as_posix(),
                    "label": label,
                    "class_name": (
                        "checked"
                        if checked
                        else "unchecked"
                    ),
                    "split": split,
                    "group_id": (
                        f"group_{group_index:04d}"
                    ),
                }
            )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "label",
                "class_name",
                "split",
                "group_id",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("CHECKBOX TRAINING DATA GENERATED")
    print("Samples:", len(rows))
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
