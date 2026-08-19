from pathlib import Path

import joblib
import pymupdf as fitz

from src.field_schema import ALL_FIELDS, CHECKBOX_SCHEMA
from src.utils.json_utils import save_json
from src.ml.train_field_classifier import train_and_save_model


MODEL_PATH = "models/field_classifier.joblib"


def load_or_train_model():
    if not Path(MODEL_PATH).exists():
        train_and_save_model()

    return joblib.load(MODEL_PATH)


def normalize_checkbox_value(value):
    value = str(value).strip().lower()

    if value == "yes":
        return "Yes"

    return "Off"


def extract_pdf_candidates(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]

    candidates = []

    for widget in page.widgets() or []:
        field_name = widget.field_name
        field_value = widget.field_value

        if field_value is None:
            field_value = ""

        field_value = str(field_value).strip()

        if field_value == "":
            continue

        candidate_text = f"{field_name} {field_value}"

        candidates.append(
            {
                "pdf_field_name": field_name,
                "value": field_value,
                "candidate_text": candidate_text
            }
        )

    doc.close()

    return candidates


def extract_custom_ml(
    pdf_path="data/filled_form.pdf",
    output_path="outputs/json/extracted_custom_ml.json"
):
    if not Path(pdf_path).exists():
        extracted = {field: "" for field in ALL_FIELDS}
        extracted["custom_ml_status"] = "Failed: uploaded PDF not found"
        save_json(extracted, output_path)
        return extracted

    model = load_or_train_model()
    candidates = extract_pdf_candidates(pdf_path)

    extracted = {}

    for field in ALL_FIELDS:
        if field in CHECKBOX_SCHEMA:
            extracted[field] = "Off"
        else:
            extracted[field] = ""

    confidence_tracker = {}

    for item in candidates:
        candidate_text = item["candidate_text"]
        raw_value = item["value"]

        predicted_field = model.predict([candidate_text])[0]

        confidence = 1.0

        try:
            probabilities = model.predict_proba([candidate_text])[0]
            confidence = max(probabilities)
        except Exception:
            confidence = 1.0

        current_confidence = confidence_tracker.get(predicted_field, -1)

        if confidence >= current_confidence:
            if predicted_field in CHECKBOX_SCHEMA:
                extracted[predicted_field] = normalize_checkbox_value(raw_value)
            else:
                extracted[predicted_field] = raw_value

            confidence_tracker[predicted_field] = confidence

    extracted["custom_ml_status"] = "Success: trained TF-IDF + Logistic Regression classifier used"

    save_json(extracted, output_path)

    return extracted


if __name__ == "__main__":
    result = extract_custom_ml()
    print("Custom ML extraction completed.")
    print(result)
