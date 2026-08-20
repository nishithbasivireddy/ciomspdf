from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf as fitz
import torch
from PIL import Image

from src.field_schema import FIELD_SCHEMA
from src.ml.crnn.decoder import greedy_ctc_decode
from src.ml.crnn.model import CRNNTextRecognizer
from src.ml.crnn.vocabulary import VOCABULARY


MODEL_PATH = Path(
    "models/custom_ml/crnn_cioms_finetuned.pt"
)

DEFAULT_PDF_PATH = Path(
    "data/filled_form.pdf"
)

DEFAULT_OUTPUT_PATH = Path(
    "outputs/json/real_cioms_custom_ml_text_test.json"
)

DEFAULT_DEBUG_DIRECTORY = Path(
    "outputs/custom_ml_debug/real_cioms_crops"
)

INPUT_HEIGHT = 32
INPUT_WIDTH = 512
TRAINING_CANVAS_WIDTH = 256
TRAINING_CANVAS_HEIGHT = 32
RENDER_ZOOM = 4.0

MULTILINE_FIELDS = {
    "reaction_description",
    "concomitant_drugs",
    "history",
    "manufacturer_name_address",
}


def load_trained_model() -> CRNNTextRecognizer:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model is missing: {MODEL_PATH}"
        )

    model = CRNNTextRecognizer(
        number_of_classes=len(VOCABULARY),
        hidden_size=128,
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    return model


def build_reverse_field_map() -> dict[str, str]:
    reverse_map = {}

    for output_key, metadata in FIELD_SCHEMA.items():
        pdf_name = metadata["pdf_field"]
        reverse_map[pdf_name] = output_key

    return reverse_map


def render_page(
    page: fitz.Page,
    zoom: float,
) -> Image.Image:
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
        annots=True,
    )

    return Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )


def crop_rectangle(
    page_image: Image.Image,
    rectangle: fitz.Rect,
    zoom: float,
    padding: int = 2,
) -> Image.Image:
    left = max(
        0,
        int(round(rectangle.x0 * zoom)) - padding,
    )

    top = max(
        0,
        int(round(rectangle.y0 * zoom)) - padding,
    )

    right = min(
        page_image.width,
        int(round(rectangle.x1 * zoom)) + padding,
    )

    bottom = min(
        page_image.height,
        int(round(rectangle.y1 * zoom)) + padding,
    )

    if right <= left or bottom <= top:
        raise ValueError(
            f"Invalid rendered rectangle: {rectangle}"
        )

    return page_image.crop(
        (left, top, right, bottom)
    )


def remove_edge_region(
    grayscale: Image.Image,
) -> Image.Image:
    width, height = grayscale.size

    horizontal_margin = max(
        1,
        int(width * 0.02),
    )

    vertical_margin = max(
        1,
        int(height * 0.08),
    )

    if (
        width <= horizontal_margin * 2
        or height <= vertical_margin * 2
    ):
        return grayscale

    return grayscale.crop(
        (
            horizontal_margin,
            vertical_margin,
            width - horizontal_margin,
            height - vertical_margin,
        )
    )


def visible_ink_count(
    image: Image.Image,
) -> int:
    grayscale = image.convert("L")
    interior = remove_edge_region(grayscale)

    array = np.asarray(
        interior,
        dtype=np.uint8,
    )

    return int(
        np.count_nonzero(array < 190)
    )


def crop_contains_text(
    image: Image.Image,
) -> bool:
    grayscale = image.convert("L")
    interior = remove_edge_region(grayscale)

    area = interior.width * interior.height

    minimum_pixels = max(
        8,
        int(area * 0.001),
    )

    return visible_ink_count(image) >= minimum_pixels


def find_line_ranges(
    crop: Image.Image,
) -> list[tuple[int, int]]:
    grayscale = crop.convert("L")
    array = np.asarray(
        grayscale,
        dtype=np.uint8,
    )

    height, width = array.shape

    x_margin = max(
        1,
        int(width * 0.02),
    )

    if width > x_margin * 2:
        array = array[:, x_margin:-x_margin]

    dark_mask = array < 190

    row_counts = dark_mask.sum(axis=1)

    minimum_row_pixels = max(
        2,
        int(array.shape[1] * 0.003),
    )

    active_rows = row_counts >= minimum_row_pixels

    if height >= 4:
        active_rows[0] = False
        active_rows[-1] = False

    ranges = []
    start = None

    for row_index, is_active in enumerate(active_rows):
        if is_active and start is None:
            start = row_index

        if not is_active and start is not None:
            ranges.append(
                (start, row_index)
            )
            start = None

    if start is not None:
        ranges.append(
            (start, height)
        )

    merged_ranges = []

    for start, end in ranges:
        if end - start < 2:
            continue

        if (
            merged_ranges
            and start - merged_ranges[-1][1] <= 3
        ):
            previous_start, _ = merged_ranges[-1]

            merged_ranges[-1] = (
                previous_start,
                end,
            )

        else:
            merged_ranges.append(
                (start, end)
            )

    padded_ranges = []

    for start, end in merged_ranges:
        padded_start = max(
            0,
            start - 2,
        )

        padded_end = min(
            height,
            end + 2,
        )

        if padded_end - padded_start >= 3:
            padded_ranges.append(
                (
                    padded_start,
                    padded_end,
                )
            )

    return padded_ranges


def split_into_text_lines(
    crop: Image.Image,
) -> list[Image.Image]:
    if not crop_contains_text(crop):
        return []

    line_ranges = find_line_ranges(crop)

    if not line_ranges:
        return [crop.copy()]

    if len(line_ranges) == 1:
        return [crop.copy()]

    line_images = []

    for start, end in line_ranges:
        line_crop = crop.crop(
            (
                0,
                start,
                crop.width,
                end,
            )
        )

        if crop_contains_text(line_crop):
            line_images.append(line_crop)

    return line_images or [crop.copy()]


def find_text_bounds(
    grayscale: Image.Image,
) -> tuple[int, int, int, int] | None:
    array = np.asarray(
        grayscale,
        dtype=np.uint8,
    )

    height, width = array.shape

    if height < 3 or width < 3:
        return None

    edge_x = max(1, int(width * 0.02))
    edge_y = max(1, int(height * 0.08))

    working = array.copy()

    working[:edge_y, :] = 255
    working[-edge_y:, :] = 255
    working[:, :edge_x] = 255
    working[:, -edge_x:] = 255

    dark_rows, dark_columns = np.where(
        working < 190
    )

    if len(dark_rows) < 4:
        return None

    left = max(
        0,
        int(dark_columns.min()) - 2,
    )

    top = max(
        0,
        int(dark_rows.min()) - 2,
    )

    right = min(
        width,
        int(dark_columns.max()) + 3,
    )

    bottom = min(
        height,
        int(dark_rows.max()) + 3,
    )

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def preprocess_line(
    line_image: Image.Image,
) -> torch.Tensor:
    grayscale = line_image.convert("L")
    bounds = find_text_bounds(grayscale)

    if bounds is not None:
        grayscale = grayscale.crop(bounds)

    available_width = TRAINING_CANVAS_WIDTH - 8
    available_height = TRAINING_CANVAS_HEIGHT - 6

    scale = min(
        available_width / max(1, grayscale.width),
        available_height / max(1, grayscale.height),
    )

    resized_width = max(
        1,
        int(round(grayscale.width * scale)),
    )

    resized_height = max(
        1,
        int(round(grayscale.height * scale)),
    )

    resized = grayscale.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    training_canvas = Image.new(
        "L",
        (
            TRAINING_CANVAS_WIDTH,
            TRAINING_CANVAS_HEIGHT,
        ),
        color=255,
    )

    horizontal_offset = 4
    vertical_offset = (
        TRAINING_CANVAS_HEIGHT - resized_height
    ) // 2

    training_canvas.paste(
        resized,
        (
            horizontal_offset,
            vertical_offset,
        ),
    )

    model_image = training_canvas.resize(
        (INPUT_WIDTH, INPUT_HEIGHT),
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        model_image,
        dtype=np.float32,
    )

    image_array = image_array / 255.0
    image_array = 1.0 - image_array

    return torch.from_numpy(
        image_array
    ).unsqueeze(0).unsqueeze(0)


def predict_line(
    model: CRNNTextRecognizer,
    line_image: Image.Image,
) -> str:
    tensor = preprocess_line(line_image)

    with torch.inference_mode():
        logits = model(tensor)

    prediction = greedy_ctc_decode(
        logits
    )[0]

    return prediction.strip()


def predict_field_crop(
    model: CRNNTextRecognizer,
    crop: Image.Image,
    line_debug_prefix: Path,
) -> tuple[str, list[dict]]:
    line_images = split_into_text_lines(crop)

    if not line_images:
        return "", []

    predictions = []
    line_evidence = []

    for line_number, line_image in enumerate(
        line_images,
        start=1,
    ):
        line_path = Path(
            f"{line_debug_prefix}_line_{line_number}.png"
        )

        line_image.save(line_path)

        prediction = predict_line(
            model=model,
            line_image=line_image,
        )

        if prediction:
            predictions.append(prediction)

        line_evidence.append(
            {
                "line_number": line_number,
                "line_crop": str(line_path),
                "prediction": prediction,
            }
        )

        line_image.close()

    return "\n".join(predictions), line_evidence


def run_real_pdf_inference(
    pdf_path: Path = DEFAULT_PDF_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    debug_directory: Path = DEFAULT_DEBUG_DIRECTORY,
) -> dict:
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF is missing: {pdf_path}"
        )

    model = load_trained_model()
    reverse_map = build_reverse_field_map()

    debug_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for existing_file in debug_directory.glob("*.png"):
        existing_file.unlink()

    predictions = {
        output_key: ""
        for output_key in FIELD_SCHEMA
    }

    evidence = []

    document = fitz.open(pdf_path)

    try:
        total_widget_count = 0
        matched_text_widget_count = 0

        for page_index, page in enumerate(
            document,
            start=1,
        ):
            page_image = render_page(
                page=page,
                zoom=RENDER_ZOOM,
            )

            try:
                for widget in page.widgets() or []:
                    total_widget_count += 1

                    pdf_name = str(
                        widget.field_name or ""
                    ).strip()

                    if pdf_name not in reverse_map:
                        continue

                    matched_text_widget_count += 1

                    output_key = reverse_map[pdf_name]

                    crop = crop_rectangle(
                        page_image=page_image,
                        rectangle=widget.rect,
                        zoom=RENDER_ZOOM,
                    )

                    field_crop_path = (
                        debug_directory
                        / f"{page_index}_{output_key}.png"
                    )

                    crop.save(field_crop_path)

                    line_prefix = (
                        debug_directory
                        / f"{page_index}_{output_key}"
                    )

                    if output_key in MULTILINE_FIELDS:
                        prediction, line_evidence = (
                            predict_field_crop(
                                model=model,
                                crop=crop,
                                line_debug_prefix=line_prefix,
                            )
                        )
                    elif not crop_contains_text(crop):
                        prediction = ""
                        line_evidence = []
                    else:
                        prediction = predict_line(
                            model=model,
                            line_image=crop,
                        )

                        single_line_path = Path(
                            f"{line_prefix}_line_1.png"
                        )

                        crop.save(single_line_path)

                        line_evidence = [
                            {
                                "line_number": 1,
                                "line_crop": str(
                                    single_line_path
                                ),
                                "prediction": prediction,
                            }
                        ]

                    predictions[output_key] = prediction

                    evidence.append(
                        {
                            "page": page_index,
                            "output_key": output_key,
                            "pdf_name": pdf_name,
                            "field_crop": str(
                                field_crop_path
                            ),
                            "line_count": len(
                                line_evidence
                            ),
                            "prediction": prediction,
                            "line_evidence": line_evidence,
                            "rectangle": [
                                float(widget.rect.x0),
                                float(widget.rect.y0),
                                float(widget.rect.x1),
                                float(widget.rect.y1),
                            ],
                            "recognition_source": (
                                "Rendered PDF pixels processed "
                                "by the trained CRNN"
                            ),
                        }
                    )

                    crop.close()

            finally:
                page_image.close()

    finally:
        document.close()

    if total_widget_count == 0:
        raise ValueError(
            "No PDF form widgets were found."
        )

    if matched_text_widget_count == 0:
        raise ValueError(
            "No text widget names matched FIELD_SCHEMA."
        )

    payload = {
        "method": (
            "Custom CRNN applied to rendered "
            "CIOMS field pixels"
        ),
        "pdf_path": str(pdf_path),
        "model_path": str(MODEL_PATH),
        "total_widget_count": total_widget_count,
        "matched_text_widget_count": (
            matched_text_widget_count
        ),
        "metadata_usage": (
            "Widget name identifies the schema field. "
            "Widget rectangle identifies the pixel crop."
        ),
        "recognition_input": (
            "Rendered field-crop pixels only"
        ),
        "extracted_values": predictions,
        "evidence": evidence,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return payload


if __name__ == "__main__":
    result = run_real_pdf_inference()

    print("REAL CIOMS CRNN INFERENCE COMPLETED")
    print("-----------------------------------")
    print(
        "Total widgets:",
        result["total_widget_count"],
    )
    print(
        "Matched text widgets:",
        result["matched_text_widget_count"],
    )
    print()

    for field, prediction in result[
        "extracted_values"
    ].items():
        print(f"{field}: {prediction}")
