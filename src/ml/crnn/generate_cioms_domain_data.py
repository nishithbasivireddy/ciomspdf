from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from src.field_schema import FIELD_SCHEMA


SEED = 20260820
random.seed(SEED)

BLANK_TEMPLATE = Path("data/blank_cioms.pdf")
REAL_EVALUATION_PDF = Path("data/filled_form.pdf")

OUTPUT_ROOT = Path(
    "data/ml_training/cioms_rendered"
)

IMAGE_ROOT = OUTPUT_ROOT / "images"
TEMP_PDF_ROOT = OUTPUT_ROOT / "temporary_pdfs"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
METADATA_PATH = OUTPUT_ROOT / "metadata.json"

RENDER_ZOOM = 4.0

TRAIN_DOCUMENTS = 120
VALIDATION_DOCUMENTS = 15
TEST_DOCUMENTS = 15

SKIPPED_FIELDS = {
    "reaction_description",
}

COUNTRIES = [
    "India",
    "Canada",
    "Germany",
    "Japan",
    "Brazil",
    "France",
    "Australia",
    "Singapore",
]

DRUGS = [
    "Aspirin",
    "Amoxicillin",
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

ROUTES = [
    "Oral",
    "Topical",
    "Intravenous",
    "Intramuscular",
    "Subcutaneous",
]

INDICATIONS = [
    "after food",
    "for fever",
    "for infection",
    "for pain",
    "for allergy",
    "gastric reflux",
    "high blood pressure",
]

HISTORY_VALUES = [
    "lowerback pain",
    "history of asthma",
    "history of hypertension",
    "no relevant history",
    "penicillin allergy",
    "type 2 diabetes",
]

CITIES = [
    "Chennai",
    "Hyderabad",
    "Mumbai",
    "Bengaluru",
    "Pune",
    "New Delhi",
]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def random_date() -> str:
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1950, 2027)

    return random.choice(
        [
            f"{day:02d}/{month:02d}/{year}",
            f"{day:02d}-{month:02d}-{year}",
            f"{year}-{month:02d}-{day:02d}",
        ]
    )


def random_initials() -> str:
    length = random.choice([2, 3])

    return "".join(
        random.choice(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        for _ in range(length)
    )


def random_drug() -> str:
    drug = random.choice(DRUGS)

    if random.random() < 0.5:
        return drug

    strength = random.choice(
        [
            "5 mg",
            "20 mg",
            "40 mg",
            "75 mg",
            "100 mg",
            "250 mg",
            "500 mg",
        ]
    )

    return f"{drug} {strength}"


def random_dose() -> str:
    return random.choice(
        [
            "1",
            "2",
            "5 ml",
            "20 mg",
            "75 mg daily",
            "250 mg twice daily",
            "500 mg once daily",
        ]
    )


def random_duration() -> str:
    amount = random.randint(1, 24)
    unit = random.choice(
        [
            "days",
            "weeks",
            "months",
            "years",
        ]
    )

    return f"{amount} {unit}"


def generate_value(field: str) -> str:
    generators = {
        "patient_initials": random_initials,
        "country": lambda: random.choice(
            COUNTRIES
        ),
        "date_of_birth_day": lambda: (
            f"{random.randint(1, 28):02d}"
        ),
        "date_of_birth_month": lambda: (
            f"{random.randint(1, 12):02d}"
        ),
        "date_of_birth_year": lambda: str(
            random.randint(1950, 2010)
        ),
        "age": lambda: str(
            random.randint(1, 95)
        ),
        "sex": lambda: random.choice(
            ["M", "F"]
        ),
        "reaction_onset_day": lambda: (
            f"{random.randint(1, 28):02d}"
        ),
        "reaction_onset_month": lambda: (
            f"{random.randint(1, 12):02d}"
        ),
        "reaction_onset_year": lambda: str(
            random.randint(2020, 2027)
        ),
        "suspect_drugs": random_drug,
        "daily_doses": random_dose,
        "routes_of_administration": lambda: (
            random.choice(ROUTES)
        ),
        "indications": lambda: random.choice(
            INDICATIONS
        ),
        "therapy_dates": lambda: (
            f"{random_date()} to {random_date()}"
        ),
        "therapy_duration": random_duration,
        "concomitant_drugs": random_drug,
        "history": lambda: random.choice(
            HISTORY_VALUES
        ),
        "manufacturer_name_address": lambda: (
            f"{random.choice(CITIES)}"
        ),
        "mfr_control_no": lambda: str(
            random.randint(100000, 999999)
        ),
        "date_received": random_date,
        "report_date": random_date,
    }

    generator = generators.get(field)

    if generator is None:
        raise KeyError(
            f"No generator configured for: {field}"
        )

    return " ".join(
        str(generator()).split()
    )


def document_split(index: int) -> str:
    if index < TRAIN_DOCUMENTS:
        return "train"

    if (
        index
        < TRAIN_DOCUMENTS
        + VALIDATION_DOCUMENTS
    ):
        return "validation"

    return "test"


def render_page(
    page: fitz.Page,
) -> Image.Image:
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(
            RENDER_ZOOM,
            RENDER_ZOOM,
        ),
        alpha=False,
        annots=True,
    )

    return Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )


def crop_widget(
    page_image: Image.Image,
    rectangle: fitz.Rect,
) -> Image.Image:
    padding = 2

    left = max(
        0,
        int(round(
            rectangle.x0 * RENDER_ZOOM
        )) - padding,
    )

    top = max(
        0,
        int(round(
            rectangle.y0 * RENDER_ZOOM
        )) - padding,
    )

    right = min(
        page_image.width,
        int(round(
            rectangle.x1 * RENDER_ZOOM
        )) + padding,
    )

    bottom = min(
        page_image.height,
        int(round(
            rectangle.y1 * RENDER_ZOOM
        )) + padding,
    )

    return page_image.crop(
        (left, top, right, bottom)
    )


def fill_document(
    output_pdf: Path,
) -> dict[str, str]:
    document = fitz.open(BLANK_TEMPLATE)
    generated = {}

    reverse_map = {
        metadata["pdf_field"]: output_key
        for output_key, metadata
        in FIELD_SCHEMA.items()
    }

    try:
        for page in document:
            for widget in page.widgets() or []:
                pdf_name = str(
                    widget.field_name or ""
                ).strip()

                output_key = reverse_map.get(
                    pdf_name
                )

                if output_key is None:
                    continue

                if output_key in SKIPPED_FIELDS:
                    value = ""
                else:
                    value = generate_value(
                        output_key
                    )

                generated[output_key] = value
                widget.field_value = value
                widget.update()

        document.save(
            output_pdf,
            garbage=4,
            deflate=True,
        )

    finally:
        document.close()

    return generated


def extract_training_crops(
    pdf_path: Path,
    values: dict[str, str],
    document_id: str,
    split: str,
) -> list[dict]:
    rows = []

    reverse_map = {
        metadata["pdf_field"]: output_key
        for output_key, metadata
        in FIELD_SCHEMA.items()
    }

    document = fitz.open(pdf_path)

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            page_image = render_page(page)

            try:
                for widget in page.widgets() or []:
                    pdf_name = str(
                        widget.field_name or ""
                    ).strip()

                    output_key = reverse_map.get(
                        pdf_name
                    )

                    if output_key is None:
                        continue

                    if output_key in SKIPPED_FIELDS:
                        continue

                    label = values.get(
                        output_key,
                        "",
                    ).strip()

                    if not label:
                        continue

                    crop = crop_widget(
                        page_image,
                        widget.rect,
                    )

                    file_name = (
                        f"{split}_"
                        f"{document_id}_"
                        f"{output_key}.png"
                    )

                    image_path = (
                        IMAGE_ROOT / file_name
                    )

                    crop.save(image_path)
                    crop.close()

                    rows.append(
                        {
                            "document_id": (
                                document_id
                            ),
                            "split": split,
                            "field": output_key,
                            "pdf_field": pdf_name,
                            "image_path": (
                                Path("images")
                                / file_name
                            ).as_posix(),
                            "text": label,
                            "page": page_number,
                        }
                    )

            finally:
                page_image.close()

    finally:
        document.close()

    return rows


def main() -> None:
    if not BLANK_TEMPLATE.is_file():
        raise FileNotFoundError(
            f"Blank template is missing: "
            f"{BLANK_TEMPLATE}"
        )

    if not REAL_EVALUATION_PDF.is_file():
        raise FileNotFoundError(
            f"Real evaluation PDF is missing: "
            f"{REAL_EVALUATION_PDF}"
        )

    blank_hash = file_hash(
        BLANK_TEMPLATE
    )

    evaluation_hash = file_hash(
        REAL_EVALUATION_PDF
    )

    if blank_hash == evaluation_hash:
        raise RuntimeError(
            "Blank template and real evaluation "
            "PDF must not be the same file."
        )

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_PDF_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in IMAGE_ROOT.glob("*.png"):
        path.unlink()

    for path in TEMP_PDF_ROOT.glob("*.pdf"):
        path.unlink()

    total_documents = (
        TRAIN_DOCUMENTS
        + VALIDATION_DOCUMENTS
        + TEST_DOCUMENTS
    )

    rows = []

    for index in range(total_documents):
        split = document_split(index)

        document_id = (
            f"form_{index:04d}"
        )

        temporary_pdf = (
            TEMP_PDF_ROOT
            / f"{document_id}.pdf"
        )

        values = fill_document(
            temporary_pdf
        )

        rows.extend(
            extract_training_crops(
                pdf_path=temporary_pdf,
                values=values,
                document_id=document_id,
                split=split,
            )
        )

        temporary_pdf.unlink()

        if (
            index == 0
            or (index + 1) % 10 == 0
            or index + 1 == total_documents
        ):
            print(
                f"Generated forms: "
                f"{index + 1}/{total_documents}",
                flush=True,
            )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "document_id",
                "split",
                "field",
                "pdf_field",
                "image_path",
                "text",
                "page",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}

    for row in rows:
        split = row["split"]

        split_counts[split] = (
            split_counts.get(split, 0)
            + 1
        )

    metadata = {
        "seed": SEED,
        "template": str(
            BLANK_TEMPLATE
        ),
        "template_sha256": blank_hash,
        "excluded_real_evaluation_pdf": str(
            REAL_EVALUATION_PDF
        ),
        "excluded_real_evaluation_sha256": (
            evaluation_hash
        ),
        "train_documents": (
            TRAIN_DOCUMENTS
        ),
        "validation_documents": (
            VALIDATION_DOCUMENTS
        ),
        "test_documents": (
            TEST_DOCUMENTS
        ),
        "sample_count": len(rows),
        "split_counts": split_counts,
        "skipped_fields": sorted(
            SKIPPED_FIELDS
        ),
        "integrity_statement": (
            "The real evaluation PDF was not "
            "used to generate training labels "
            "or training crop images."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("CIOMS DOMAIN DATA GENERATED")
    print("---------------------------")
    print(
        "Documents:",
        total_documents,
    )
    print(
        "Crop samples:",
        len(rows),
    )
    print(
        "Split counts:",
        split_counts,
    )
    print(
        "Manifest:",
        MANIFEST_PATH,
    )
    print(
        "Metadata:",
        METADATA_PATH,
    )


if __name__ == "__main__":
    main()
