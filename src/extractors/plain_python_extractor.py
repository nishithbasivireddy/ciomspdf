import fitz
from src.field_schema import FIELD_SCHEMA, CHECKBOX_SCHEMA
from src.utils.json_utils import save_json


def extract_plain_python(pdf_path="data/filled_cioms.pdf", output_path="outputs/json/extracted_plain_python.json"):
    doc = fitz.open(pdf_path)
    page = doc[0]

    raw_fields = {}

    for widget in page.widgets() or []:
        raw_fields[widget.field_name] = widget.field_value

    extracted = {}

    for key, meta in FIELD_SCHEMA.items():
        extracted[key] = raw_fields.get(meta["pdf_field"], "")

    for key, checkbox_field in CHECKBOX_SCHEMA.items():
        value = raw_fields.get(checkbox_field, "")
        extracted[key] = "Yes" if str(value).lower() == "yes" else "Off"

    save_json(extracted, output_path)
    doc.close()

    return extracted


if __name__ == "__main__":
    result = extract_plain_python()
    print("Plain Python extraction completed.")
    print(result)
