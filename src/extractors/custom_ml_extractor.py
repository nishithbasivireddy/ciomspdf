from pathlib import Path

from src.field_schema import ALL_FIELDS
from src.extractors.plain_python_extractor import extract_plain_python
from src.utils.json_utils import load_json, save_json


def extract_custom_ml(
    pdf_path="data/filled_cioms.pdf",
    plain_output_path="outputs/json/extracted_plain_python.json",
    output_path="outputs/json/extracted_custom_ml.json"
):
    """
    Custom ML extractor placeholder/baseline.

    Ground truth is NOT required for extraction.
    Ground truth is required only for final quality comparison.

    Current logic:
    1. If plain-python output exists, use it as extracted field candidates.
    2. If plain-python output does not exist, generate it from uploaded PDF.
    3. Save a normalized output with the expected field keys.

    Later this file can be upgraded into a trained custom ML model using synthetic labelled PDFs.
    """

    if not Path(pdf_path).exists():
        extracted = {field: "" for field in ALL_FIELDS}
        extracted["custom_ml_status"] = "Failed: uploaded PDF not found"
        save_json(extracted, output_path)
        return extracted

    if not Path(plain_output_path).exists():
        plain_output = extract_plain_python(pdf_path=pdf_path, output_path=plain_output_path)
    else:
        plain_output = load_json(plain_output_path)

    extracted = {}

    for field in ALL_FIELDS:
        extracted[field] = plain_output.get(field, "")

    extracted["custom_ml_status"] = "Success: baseline custom ML extraction completed without ground truth"

    save_json(extracted, output_path)
    return extracted


if __name__ == "__main__":
    result = extract_custom_ml()
    print("Custom ML extraction completed.")
    print(result)
