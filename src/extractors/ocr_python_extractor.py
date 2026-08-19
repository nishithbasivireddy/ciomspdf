from pathlib import Path
import warnings

import easyocr
import numpy as np
import pymupdf as fitz
from PIL import Image, ImageOps, ImageEnhance

from src.field_schema import FIELD_SCHEMA, CHECKBOX_SCHEMA, ALL_FIELDS
from src.utils.json_utils import save_json

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_reader = None


def get_easyocr_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


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


def crop_widget(page_image, rect, zoom=4, padding=10):
    left = max(int(rect.x0 * zoom) - padding, 0)
    top = max(int(rect.y0 * zoom) - padding, 0)
    right = min(int(rect.x1 * zoom) + padding, page_image.width)
    bottom = min(int(rect.y1 * zoom) + padding, page_image.height)

    return page_image.crop((left, top, right, bottom))


def ocr_text_crop(crop):
    reader = get_easyocr_reader()
    processed = preprocess_crop(crop)

    image_array = np.array(processed)

    results = reader.readtext(
        image_array,
        detail=0,
        paragraph=True
    )

    if not results:
        return ""

    text = " ".join(results)
    return clean_ocr_text(text)


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


def extract_ocr_python(
    pdf_path="data/filled_form.pdf",
    output_path="outputs/json/extracted_ocr_python.json"
):
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
            value = ocr_text_crop(crop)

        extracted[output_key] = value
        debug_rows.append(f"{output_key}: {value}")

    doc.close()

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    with open("outputs/reports/ocr_field_debug.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(debug_rows))

    reader = get_easyocr_reader()
    full_page_results = reader.readtext(
        np.array(page_image),
        detail=0,
        paragraph=True
    )

    with open("outputs/reports/ocr_raw_text.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(full_page_results))

    extracted["ocr_status"] = "Success"

    save_json(extracted, output_path)

    return extracted


if __name__ == "__main__":
    result = extract_ocr_python()
    print("OCR extraction completed.")
    print(result)
