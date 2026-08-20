from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf as fitz
import torch
from PIL import Image

from src.field_schema import CHECKBOX_SCHEMA
from src.ml.checkbox.model import CheckboxCNN


MODEL_PATH = Path(
    "models/custom_ml/checkbox_cnn.pt"
)

PDF_PATH = Path(
    "data/filled_form.pdf"
)

OUTPUT_PATH = Path(
    "outputs/json/real_cioms_checkbox_ml_test.json"
)

DEBUG_ROOT = Path(
    "outputs/custom_ml_debug/real_checkbox_crops"
)

RENDER_ZOOM = 8.0


def load_model() -> CheckboxCNN:
    model = CheckboxCNN()

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    return model


def preprocess(
    crop: Image.Image,
) -> torch.Tensor:
    grayscale = crop.convert("L").resize(
        (32, 32),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(
        grayscale,
        dtype=np.float32,
    )

    array = 1.0 - (array / 255.0)

    return torch.from_numpy(
        array
    ).unsqueeze(0).unsqueeze(0)


def main() -> None:
    model = load_model()

    reverse_map = {pdf_name: output_key for output_key, pdf_name in CHECKBOX_SCHEMA.items()}

    DEBUG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = fitz.open(PDF_PATH)

    results = {}
    evidence = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(
                    RENDER_ZOOM,
                    RENDER_ZOOM,
                ),
                alpha=False,
                annots=True,
            )

            page_image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

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

                    if str(
                        widget.field_type_string
                    ).casefold() != "checkbox":
                        continue

                    rectangle = widget.rect
                    padding = 8

                    left = max(
                        0,
                        int(rectangle.x0 * RENDER_ZOOM)
                        - padding,
                    )

                    top = max(
                        0,
                        int(rectangle.y0 * RENDER_ZOOM)
                        - padding,
                    )

                    right = min(
                        page_image.width,
                        int(rectangle.x1 * RENDER_ZOOM)
                        + padding,
                    )

                    bottom = min(
                        page_image.height,
                        int(rectangle.y1 * RENDER_ZOOM)
                        + padding,
                    )

                    crop = page_image.crop(
                        (left, top, right, bottom)
                    )

                    crop_path = (
                        DEBUG_ROOT
                        / f"{page_number}_{output_key}.png"
                    )

                    crop.save(crop_path)

                    tensor = preprocess(crop)

                    with torch.inference_mode():
                        logits = model(tensor)
                        probabilities = torch.softmax(
                            logits,
                            dim=1,
                        )[0]

                    prediction = int(
                        probabilities.argmax().item()
                    )

                    result_value = (
                        "Yes"
                        if prediction == 1
                        else "Off"
                    )

                    results[output_key] = result_value

                    evidence.append(
                        {
                            "field": output_key,
                            "pdf_name": pdf_name,
                            "prediction": result_value,
                            "unchecked_probability": float(
                                probabilities[0].item()
                            ),
                            "checked_probability": float(
                                probabilities[1].item()
                            ),
                            "crop_path": str(crop_path),
                        }
                    )

                    crop.close()

            finally:
                page_image.close()

    finally:
        document.close()

    payload = {
        "method": (
            "Custom checkbox CNN applied to "
            "rendered PDF checkbox pixels"
        ),
        "pdf_path": str(PDF_PATH),
        "model_path": str(MODEL_PATH),
        "extracted_values": results,
        "evidence": evidence,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("REAL CHECKBOX CNN INFERENCE COMPLETED")

    for field, value in results.items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
