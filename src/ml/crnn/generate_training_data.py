from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SEED = 20260820
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_ROOT = Path("data/ml_training/text")
IMAGE_ROOT = OUTPUT_ROOT / "images"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
METADATA_PATH = OUTPUT_ROOT / "dataset_metadata.json"

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256

TRAIN_GROUPS = 240
VALIDATION_GROUPS = 30
TEST_GROUPS = 30
AUGMENTATIONS_PER_GROUP = 8

SUPPORTED_EXTENSIONS = {
    ".ttf",
    ".otf",
}


FIRST_NAMES = [
    "Aarav",
    "Aditya",
    "Ananya",
    "Arjun",
    "Divya",
    "Ishaan",
    "Kavya",
    "Meera",
    "Nikhil",
    "Priya",
    "Rahul",
    "Riya",
    "Sanjay",
    "Sneha",
    "Vikram",
]

LAST_NAMES = [
    "Reddy",
    "Sharma",
    "Gupta",
    "Kumar",
    "Patel",
    "Rao",
    "Singh",
    "Verma",
]

COUNTRIES = [
    "India",
    "Australia",
    "Brazil",
    "Canada",
    "France",
    "Germany",
    "Japan",
    "Singapore",
    "United Kingdom",
    "United States",
]

DRUGS = [
    "Amoxicillin",
    "Aspirin",
    "Atorvastatin",
    "Azithromycin",
    "Cefixime",
    "Cetirizine",
    "Crocin",
    "Ibuprofen",
    "Metformin",
    "Omeprazole",
    "Pantoprazole",
    "Paracetamol",
]

FORMS = [
    "capsules",
    "injection",
    "oral suspension",
    "solution",
    "tablets",
]

ROUTES = [
    "Oral",
    "Intramuscular",
    "Intravenous",
    "Subcutaneous",
    "Topical",
]

INDICATIONS = [
    "Allergic rhinitis",
    "Bacterial infection",
    "Fever and body pain",
    "Gastric reflux",
    "Hypertension",
    "Type 2 diabetes",
    "Upper respiratory infection",
]

REACTIONS = [
    "Abdominal discomfort",
    "Dizziness and nausea",
    "Facial swelling",
    "Generalised itching",
    "Headache and fatigue",
    "Mild skin rash",
    "Persistent vomiting",
    "Shortness of breath",
]

HISTORY = [
    "No relevant medical history",
    "History of asthma",
    "History of hypertension",
    "Known allergy to penicillin",
    "Type 2 diabetes for five years",
    "Previous episode of urticaria",
]

MANUFACTURERS = [
    "ABC Pharmaceuticals Ltd",
    "Global Health Laboratories",
    "Medicare Pharma Pvt Ltd",
    "Nova Therapeutics",
    "Sunrise Medicines Ltd",
]

SOURCE_TYPES = [
    "Health professional",
    "Literature",
    "Study",
]


def discover_fonts():
    candidate_roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
    ]

    preferred_names = {
        "arial.ttf",
        "arialbd.ttf",
        "calibri.ttf",
        "calibrib.ttf",
        "dejavusans.ttf",
        "dejavusans-bold.ttf",
        "liberationsans-regular.ttf",
        "liberationsans-bold.ttf",
    }

    preferred_fonts = []
    fallback_fonts = []

    for root in candidate_roots:
        if not root.exists():
            continue

        for font_path in root.rglob("*"):
            if not font_path.is_file():
                continue

            if font_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            fallback_fonts.append(font_path)

            if font_path.name.lower() in preferred_names:
                preferred_fonts.append(font_path)

    selected_fonts = preferred_fonts or fallback_fonts

    if not selected_fonts:
        raise RuntimeError(
            "No TrueType or OpenType fonts were found."
        )

    return sorted(set(selected_fonts))


def random_date() -> str:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1950, 2026)

    formats = [
        f"{day:02d}/{month:02d}/{year}",
        f"{day:02d}-{month:02d}-{year}",
        f"{year}-{month:02d}-{day:02d}",
    ]

    return random.choice(formats)


def random_initials() -> str:
    return (
        random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    )


def random_dose() -> str:
    amount = random.choice(
        [5, 10, 20, 40, 50, 75, 100, 250, 500, 650]
    )
    unit = random.choice(["mg", "mcg", "ml"])
    frequency = random.choice(
        [
            "once daily",
            "twice daily",
            "three times daily",
            "at bedtime",
            "every 8 hours",
            "as required",
        ]
    )

    return f"{amount} {unit} {frequency}"


def random_duration() -> str:
    amount = random.randint(1, 24)
    unit = random.choice(
        [
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "year",
            "years",
        ]
    )

    return f"{amount} {unit}"


def random_drug() -> str:
    drug = random.choice(DRUGS)
    amount = random.choice([10, 20, 40, 50, 75, 100, 250, 500])
    unit = random.choice(["mg", "mcg", "ml"])

    if random.random() < 0.65:
        form = random.choice(FORMS)
        return f"{drug} {amount} {unit} {form}"

    return f"{drug} {amount} {unit}"


def random_control_number() -> str:
    prefix = random.choice(["CIOMS", "PV", "SAE", "CASE"])
    number = random.randint(10000, 999999)
    year = random.randint(2020, 2026)

    return f"{prefix}-{year}-{number}"


def random_manufacturer_address() -> str:
    company = random.choice(MANUFACTURERS)
    city = random.choice(
        [
            "Bengaluru",
            "Chennai",
            "Hyderabad",
            "Mumbai",
            "New Delhi",
            "Pune",
        ]
    )

    return f"{company}, {city}"


def generate_text_value() -> str:
    generators = [
        random_initials,
        lambda: random.choice(COUNTRIES),
        random_date,
        lambda: str(random.randint(1, 95)),
        lambda: random.choice(REACTIONS),
        random_drug,
        random_dose,
        lambda: random.choice(ROUTES),
        lambda: random.choice(INDICATIONS),
        lambda: f"{random_date()} to {random_date()}",
        random_duration,
        lambda: random.choice(HISTORY),
        random_manufacturer_address,
        random_control_number,
        lambda: random.choice(SOURCE_TYPES),
        lambda: (
            f"{random.choice(FIRST_NAMES)} "
            f"{random.choice(LAST_NAMES)}"
        ),
    ]

    value = random.choice(generators)()
    value = " ".join(value.split())

    return value[:60]


def fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
) -> ImageFont.FreeTypeFont:
    for size in range(22, 9, -1):
        font = ImageFont.truetype(str(font_path), size=size)

        bounding_box = draw.textbbox(
            (0, 0),
            text,
            font=font,
        )

        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        if width <= IMAGE_WIDTH - 12 and height <= IMAGE_HEIGHT - 6:
            return font

    return ImageFont.truetype(str(font_path), size=10)


def add_noise(image: Image.Image) -> Image.Image:
    array = np.asarray(image).astype(np.int16)

    noise_level = random.uniform(0.0, 10.0)
    noise = np.random.normal(
        loc=0.0,
        scale=noise_level,
        size=array.shape,
    )

    array = np.clip(array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(array, mode="L")


def render_sample(
    text: str,
    font_path: Path,
) -> Image.Image:
    background = random.randint(238, 255)

    image = Image.new(
        "L",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        color=background,
    )

    draw = ImageDraw.Draw(image)
    font = fit_font(draw, text, font_path)

    bounding_box = draw.textbbox(
        (0, 0),
        text,
        font=font,
    )

    text_width = bounding_box[2] - bounding_box[0]
    text_height = bounding_box[3] - bounding_box[1]

    maximum_x = max(3, IMAGE_WIDTH - text_width - 3)
    maximum_y = max(2, IMAGE_HEIGHT - text_height - 2)

    x = random.randint(3, maximum_x)
    y = random.randint(1, maximum_y)

    foreground = random.randint(0, 40)

    draw.text(
        (x, y),
        text,
        font=font,
        fill=foreground,
    )

    if random.random() < 0.35:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.1, 0.7)
            )
        )

    if random.random() < 0.50:
        image = add_noise(image)

    if random.random() < 0.30:
        angle = random.uniform(-1.0, 1.0)
        image = image.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            fillcolor=background,
        )

    if random.random() < 0.20:
        draw = ImageDraw.Draw(image)
        line_y = random.choice(
            [
                1,
                IMAGE_HEIGHT - 2,
            ]
        )
        draw.line(
            (0, line_y, IMAGE_WIDTH, line_y),
            fill=random.randint(130, 220),
            width=1,
        )

    return image


def assign_split(group_index: int) -> str:
    if group_index < TRAIN_GROUPS:
        return "train"

    if group_index < TRAIN_GROUPS + VALIDATION_GROUPS:
        return "validation"

    return "test"


def main() -> None:
    fonts = discover_fonts()

    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    for existing_image in IMAGE_ROOT.glob("*.png"):
        existing_image.unlink()

    total_groups = (
        TRAIN_GROUPS
        + VALIDATION_GROUPS
        + TEST_GROUPS
    )

    rows = []
    used_texts = set()

    for group_index in range(total_groups):
        text = generate_text_value()

        while text in used_texts:
            text = generate_text_value()

        used_texts.add(text)
        split = assign_split(group_index)

        for augmentation_index in range(
            AUGMENTATIONS_PER_GROUP
        ):
            font_path = random.choice(fonts)

            image = render_sample(
                text=text,
                font_path=font_path,
            )

            file_name = (
                f"{split}_"
                f"{group_index:04d}_"
                f"{augmentation_index:02d}.png"
            )

            relative_image_path = Path("images") / file_name
            output_image_path = OUTPUT_ROOT / relative_image_path

            image.save(
                output_image_path,
                format="PNG",
                optimize=True,
            )

            rows.append(
                {
                    "image_path": relative_image_path.as_posix(),
                    "text": text,
                    "split": split,
                    "group_id": f"group_{group_index:04d}",
                    "font_name": font_path.name,
                }
            )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "text",
                "split",
                "group_id",
                "font_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}

    for row in rows:
        split = row["split"]
        split_counts[split] = split_counts.get(split, 0) + 1

    metadata = {
        "seed": SEED,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "font_count": len(fonts),
        "group_count": total_groups,
        "augmentations_per_group": AUGMENTATIONS_PER_GROUP,
        "sample_count": len(rows),
        "split_counts": split_counts,
        "leakage_control": (
            "All augmented images of the same transcription "
            "remain in one split."
        ),
        "purpose": (
            "Synthetic labelled crop images for offline CRNN "
            "text-recognition training. These are not runtime "
            "ground truth values."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(f"Manifest saved: {MANIFEST_PATH}")
    print(f"Images saved: {IMAGE_ROOT}")


if __name__ == "__main__":
    main()
