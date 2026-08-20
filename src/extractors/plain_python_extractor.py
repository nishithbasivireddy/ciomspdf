from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf as fitz

from src.field_schema import ALL_FIELDS, CHECKBOX_SCHEMA, FIELD_SCHEMA
from src.utils.json_utils import save_json


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def normalize_checkbox_value(value: Any) -> str:
    normalized = normalize_text_value(value).casefold()

    checked_values = {
        "yes",
        "on",
        "true",
        "1",
        "checked",
        "selected",
    }

    if normalized in checked_values:
        return "Yes"

    return "Off"


def collect_pdf_widgets(document: fitz.Document) -> tuple[dict[str, Any], list[dict]]:
    raw_fields: dict[str, Any] = {}
    widget_details: list[dict] = []

    for page_number, page in enumerate(document, start=1):
        for widget in page.widgets() or []:
            field_name = normalize_text_value(widget.field_name)

            if not field_name:
                continue

            raw_value = widget.field_value

            if field_name not in raw_fields:
                raw_fields[field_name] = raw_value

            widget_details.append(
                {
                    "page": page_number,
                    "field_name": field_name,
                    "field_type": widget.field_type_string or "",
                    "raw_value": normalize_text_value(raw_value),
                    "rectangle": [
                        float(widget.rect.x0),
                        float(widget.rect.y0),
                        float(widget.rect.x1),
                        float(widget.rect.y1),
                    ],
                }
            )

    return raw_fields, widget_details


def build_reference_output(raw_fields: dict[str, Any]) -> dict[str, str]:
    extracted = {field: "" for field in ALL_FIELDS}

    for output_key, metadata in FIELD_SCHEMA.items():
        pdf_field_name = metadata["pdf_field"]
        extracted[output_key] = normalize_text_value(
            raw_fields.get(pdf_field_name, "")
        )

    for output_key, pdf_field_name in CHECKBOX_SCHEMA.items():
        extracted[output_key] = normalize_checkbox_value(
            raw_fields.get(pdf_field_name, "")
        )

    return extracted


def validate_reference(
    raw_fields: dict[str, Any],
    extracted: dict[str, str],
) -> dict:
    expected_pdf_fields = {
        metadata["pdf_field"]
        for metadata in FIELD_SCHEMA.values()
    }

    expected_pdf_fields.update(CHECKBOX_SCHEMA.values())

    discovered_pdf_fields = set(raw_fields)

    mapped_fields_found = sorted(
        expected_pdf_fields.intersection(discovered_pdf_fields)
    )

    missing_pdf_fields = sorted(
        expected_pdf_fields.difference(discovered_pdf_fields)
    )

    unexpected_pdf_fields = sorted(
        discovered_pdf_fields.difference(expected_pdf_fields)
    )

    non_empty_text_fields = sum(
        1
        for key in FIELD_SCHEMA
        if normalize_text_value(extracted.get(key, ""))
    )

    validation_errors = []

    if not discovered_pdf_fields:
        validation_errors.append(
            "No PDF form widgets were found."
        )

    if not mapped_fields_found:
        validation_errors.append(
            "No expected CIOMS widget names were found."
        )

    if non_empty_text_fields == 0:
        validation_errors.append(
            "No mapped text fields contain values. The PDF may be blank, "
            "flattened, scanned, or incompatible with the expected template."
        )

    schema_complete = set(extracted) == set(ALL_FIELDS)

    if not schema_complete:
        validation_errors.append(
            "Reference output does not exactly match ALL_FIELDS."
        )

    return {
        "reference_valid": len(validation_errors) == 0,
        "reference_role": "Dynamic Python widget reference baseline",
        "total_widgets_found": len(discovered_pdf_fields),
        "expected_pdf_fields": len(expected_pdf_fields),
        "mapped_pdf_fields_found": len(mapped_fields_found),
        "non_empty_text_fields": non_empty_text_fields,
        "schema_complete": schema_complete,
        "missing_pdf_fields": missing_pdf_fields,
        "unexpected_pdf_fields": unexpected_pdf_fields,
        "validation_errors": validation_errors,
        "methodology_note": (
            "This reference is dynamically extracted from embedded PDF form "
            "widgets. It is not independently verified absolute ground truth."
        ),
    }


def extract_plain_python(
    pdf_path: str = "data/filled_form.pdf",
    output_path: str = "outputs/json/extracted_plain_python.json",
    validation_path: str = "outputs/json/plain_python_reference_validation.json",
) -> dict[str, str]:
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"Uploaded filled PDF was not found: {pdf_file}"
        )

    document = fitz.open(pdf_file)

    try:
        if document.page_count == 0:
            raise ValueError("The uploaded PDF contains no pages.")

        raw_fields, widget_details = collect_pdf_widgets(document)

    finally:
        document.close()

    extracted = build_reference_output(raw_fields)
    validation = validate_reference(raw_fields, extracted)
    validation["pdf_path"] = str(pdf_file)
    validation["page_count"] = len(
        {item["page"] for item in widget_details}
    )
    validation["widget_instances"] = len(widget_details)

    save_json(extracted, output_path)
    save_json(validation, validation_path)

    debug_path = Path("outputs/reports/plain_python_widget_debug.json")
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(
        json.dumps(widget_details, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not validation["reference_valid"]:
        errors = "; ".join(validation["validation_errors"])
        raise ValueError(f"Python reference baseline is invalid: {errors}")

    return extracted


if __name__ == "__main__":
    result = extract_plain_python()
    print("Plain Python reference extraction completed.")
    print(json.dumps(result, indent=2, ensure_ascii=False))
