import os
import json
import pymupdf as fitz
from dotenv import load_dotenv
from openai import OpenAI

from src.field_schema import ALL_FIELDS, FIELD_SCHEMA, CHECKBOX_SCHEMA
from src.utils.json_utils import save_json


def get_secret_value(key, default=""):
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass

    load_dotenv()
    return os.getenv(key, default).strip()


def extract_pdf_text(pdf_path="data/filled_form.pdf"):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    doc.close()
    return text


def extract_raw_form_fields(pdf_path="data/filled_form.pdf"):
    doc = fitz.open(pdf_path)
    page = doc[0]

    raw_fields = {}

    for widget in page.widgets() or []:
        raw_fields[widget.field_name] = widget.field_value

    doc.close()
    return raw_fields


def clean_llm_json_response(content):
    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()

    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content


def extract_llm(
    pdf_path="data/filled_form.pdf",
    output_path="outputs/json/extracted_llm.json"
):
    api_key = get_secret_value("GROQ_API_KEY")
    base_url = get_secret_value("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    model = get_secret_value("GROQ_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        extracted = {field: "" for field in ALL_FIELDS}
        extracted["llm_status"] = "Missing GROQ_API_KEY"
        save_json(extracted, output_path)
        return extracted

    pdf_text = extract_pdf_text(pdf_path)
    raw_form_fields = extract_raw_form_fields(pdf_path)

    if not pdf_text.strip() and not raw_form_fields:
        extracted = {field: "" for field in ALL_FIELDS}
        extracted["llm_status"] = "No text or form fields extracted from PDF"
        extracted["llm_model"] = model
        save_json(extracted, output_path)
        return extracted

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    prompt = f"""
You are extracting structured values from a filled PDF form.

Return only valid JSON.
Use exactly these output keys:
{ALL_FIELDS}

PDF form field mapping:
FIELD_SCHEMA = {FIELD_SCHEMA}
CHECKBOX_SCHEMA = {CHECKBOX_SCHEMA}

Raw PDF form fields extracted by Python:
{json.dumps(raw_form_fields, indent=2)}

Raw PDF text extracted by Python:
{pdf_text}

Rules:
1. Do not hallucinate.
2. Prefer Raw PDF form fields when available.
3. Use Raw PDF text only as supporting context.
4. If a value is missing or unclear, return an empty string.
5. Checkbox fields must be returned only as "Yes" or "Off".
6. Do not return numbers for checkbox fields.
7. Do not add extra keys.
8. Return only JSON, no explanation.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured JSON from filled PDF forms accurately and without hallucination."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content
        content = clean_llm_json_response(content)
        extracted = json.loads(content)

        final_output = {}

        for field in ALL_FIELDS:
            value = extracted.get(field, "")

            if field in CHECKBOX_SCHEMA:
                value = "Yes" if str(value).strip().lower() == "yes" else "Off"

            final_output[field] = value

        final_output["llm_status"] = "Success"
        final_output["llm_model"] = model

    except Exception as error:
        final_output = {field: "" for field in ALL_FIELDS}
        final_output["llm_status"] = f"LLM extraction failed: {str(error)}"
        final_output["llm_model"] = model

    save_json(final_output, output_path)
    return final_output


if __name__ == "__main__":
    result = extract_llm()
    print("LLM extraction completed.")
    print(result)
