from __future__ import annotations

import json
from pathlib import Path

from src.field_schema import ALL_FIELDS
from src.ml.checkbox.real_pdf_inference import main as run_checkbox_inference
from src.ml.crnn.real_pdf_inference import run_real_pdf_inference
from src.utils.json_utils import save_json


TEXT_RESULT_PATH = Path(
    "outputs/json/real_cioms_custom_ml_text_test.json"
)

CHECKBOX_RESULT_PATH = Path(
    "outputs/json/real_cioms_checkbox_ml_test.json"
)


def extract_custom_ml(
    pdf_path: str = "data/filled_form.pdf",
    output_path: str = "outputs/json/extracted_custom_ml.json",
) -> dict[str, str]:
    pdf_file = Path(pdf_path)

    if not pdf_file.is_file():
        result = {
            field: ""
            for field in ALL_FIELDS
        }

        result["custom_ml_status"] = (
            "Failed: uploaded PDF not found"
        )

        save_json(result, output_path)
        return result

    text_payload = run_real_pdf_inference(
        pdf_path=pdf_file,
        output_path=TEXT_RESULT_PATH,
    )

    run_checkbox_inference()

    checkbox_payload = json.loads(
        CHECKBOX_RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )

    result = {
        field: ""
        for field in ALL_FIELDS
    }

    for field, value in text_payload[
        "extracted_values"
    ].items():
        if field in result:
            result[field] = value

    for field, value in checkbox_payload[
        "extracted_values"
    ].items():
        if field in result:
            result[field] = value

    result["custom_ml_status"] = (
        "Success: custom fine-tuned CRNN and "
        "custom checkbox CNN processed rendered "
        "PDF pixels"
    )

    save_json(result, output_path)
    return result


if __name__ == "__main__":
    extracted = extract_custom_ml()

    print("CUSTOM ML EXTRACTION COMPLETED")
    print("------------------------------")

    for field, value in extracted.items():
        print(f"{field}: {value}")
