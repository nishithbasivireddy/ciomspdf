from io import BytesIO
from pathlib import Path
import os

import numpy as np
import pymupdf as fitz
import requests
from PIL import Image, ImageOps, ImageEnhance
from dotenv import load_dotenv

from src.field_schema import FIELD_SCHEMA, CHECKBOX_SCHEMA, ALL_FIELDS
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


def clean_ocr_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = " ".join(text.split())

    return text.strip()


def preprocess_crop(image):
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(2.0)

    return image


def render_page_to_image(pdf_path, image_path="outputs/images/ocr_rendered_page.png", zoom=4):
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[0]

    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(image_path)

    doc.close()

    return image_path


def build_pdf_field_reverse_map():
    reverse_map = {}

    for output_key, metadata in FIELD_SCHEMA.items():
        reverse_map[metadata["pdf_field"]] = output_key

    for output_key, pdf_field_name in CHECKBOX_SCHEMA.items():
        reverse_map[pdf_field_name] = output_key

    return reverse_map


def crop_widget(page_image, rect, zoom=4, padding=12):
    left = max(int(rect.x0 * zoom) - padding, 0)
    top = max(int(rect.y0 * zoom) - padding, 0)
    right = min(int(rect.x1 * zoom) + padding, page_image.width)
    bottom = min(int(rect.y1 * zoom) + padding, page_image.height)

    return page_image.crop((left, top, right, bottom))


def detect_checkbox(crop):
    gray = crop.convert("L")
    width, height = gray.size

    left = int(width * 0.20)
    top = int(height * 0.20)
    right = int(width * 0.80)
    bottom = int(height * 0.80)

    inner = gray.crop((left, top, right, bottom))
    inner_array = np.array(inner)

    total_pixels = inner_array.size

    if total_pixels == 0:
        return "Off"

    dark_pixels = np.sum(inner_array < 100)
    dark_ratio = dark_pixels / total_pixels

    return "Yes" if dark_ratio > 0.04 else "Off"


def image_to_png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)

    return output


def call_ocr_space(image, api_key, api_url, engine):
    processed = preprocess_crop(image)
    image_bytes = image_to_png_bytes(processed)

    payload = {
        "apikey": api_key,
        "language": "eng",
        "isOverlayRequired": "false",
        "OCREngine": str(engine),
        "scale": "true"
    }

    files = {
        "file": ("field.png", image_bytes, "image/png")
    }

    response = requests.post(
        api_url,
        data=payload,
        files=files,
        timeout=90
    )

    response.raise_for_status()
    data = response.json()

    if data.get("IsErroredOnProcessing"):
        error_message = data.get("ErrorMessage") or data.get("ErrorDetails") or "OCR processing failed"

        if isinstance(error_message, list):
            error_message = " ".join(str(item) for item in error_message)

        raise RuntimeError(str(error_message))

    parsed_results = data.get("ParsedResults", [])

    if not parsed_results:
        return ""

    text_parts = []

    for item in parsed_results:
        text_parts.append(item.get("ParsedText", ""))

    return clean_ocr_text(" ".join(text_parts))


def extract_ocr_python(
    pdf_path="data/filled_form.pdf",
    output_path="outputs/json/extracted_ocr_python.json"
):
    default_url = "https://" + "api.ocr.space/parse/image"

    api_key = get_secret_value("OCR_SPACE_API_KEY")
    api_url = get_secret_value("OCR_SPACE_API_URL", default_url)
    engine = get_secret_value("OCR_SPACE_ENGINE", "2")

    if not api_key:
        extracted = {field: "" for field in ALL_FIELDS}
        extracted["ocr_status"] = "Missing OCR_SPACE_API_KEY. Add it in .env locally or Streamlit Secrets in cloud."
        save_json(extracted, output_path)
        return extracted

    if not Path(pdf_path).exists():
        extracted = {field: "" for field in ALL_FIELDS}
        extracted["ocr_status"] = "Failed: uploaded PDF not found"
        save_json(extracted, output_path)
        return extracted

    zoom = 4

    rendered_image_path = render_page_to_image(
        pdf_path=pdf_path,
        zoom=zoom
    )

    page_image = Image.open(rendered_image_path)

    doc = fitz.open(pdf_path)
    page = doc[0]

    reverse_map = build_pdf_field_reverse_map()
    extracted = {field: "" for field in ALL_FIELDS}
    debug_rows = []

    for widget in page.widgets() or []:
        pdf_field_name = widget.field_name

        if pdf_field_name not in reverse_map:
            continue

        output_key = reverse_map[pdf_field_name]
        crop = crop_widget(page_image, widget.rect, zoom=zoom)

        if output_key in CHECKBOX_SCHEMA:
            value = detect_checkbox(crop)
        else:
            try:
                value = call_ocr_space(crop, api_key, api_url, engine)
            except Exception as error:
                value = ""
                debug_rows.append(f"{output_key}: OCR failed: {str(error)}")

        extracted[output_key] = value
        debug_rows.append(f"{output_key}: {value}")

    doc.close()

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    with open("outputs/reports/ocr_field_debug.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(debug_rows))

    extracted["ocr_status"] = "Success"

    save_json(extracted, output_path)

    return extracted


if __name__ == "__main__":
    result = extract_ocr_python()
    print("OCR.space extraction completed.")
    print(result)
