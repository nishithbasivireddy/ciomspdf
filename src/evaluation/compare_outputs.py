from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.metrics import (
    evaluate_extraction,
)
from src.field_schema import ALL_FIELDS
from src.utils.json_utils import (
    load_json,
    save_json,
)


META_KEYS = {
    "llm_status",
    "llm_model",
    "custom_ml_status",
    "ocr_status",
    "ocr_raw_text_available",
    "pdf_text_available",
    "raw_llm_response",
}


EVALUATION_FILE_NAMES = {
    "Plain Python": (
        "plain_python_evaluation.json"
    ),
    "OCR + Python": (
        "ocr_plus_python_evaluation.json"
    ),
    "Custom ML + Python": (
        "custom_ml_plus_python_evaluation.json"
    ),
    "LLM Schema Mapping": (
        "llm_schema_mapping_evaluation.json"
    ),
}


APPROACH_FILES = {
    "Plain Python": (
        "outputs/json/"
        "extracted_plain_python.json"
    ),
    "OCR + Python": (
        "outputs/json/"
        "extracted_ocr_python.json"
    ),
    "Custom ML + Python": (
        "outputs/json/"
        "extracted_custom_ml.json"
    ),
    "LLM Schema Mapping": (
        "outputs/json/"
        "extracted_llm.json"
    ),
}


def clean_prediction(
    data: dict,
) -> dict[str, str]:
    return {
        field: data.get(field, "")
        for field in ALL_FIELDS
        if field not in META_KEYS
    }


def get_run_status(
    raw_data: dict,
) -> str:
    for key in [
        "ocr_status",
        "custom_ml_status",
        "llm_status",
    ]:
        value = raw_data.get(key)

        if not value:
            continue

        normalized = str(
            value
        ).strip().casefold()

        if normalized == "success":
            return "Completed"

        if normalized.startswith(
            "success"
        ):
            return "Completed"

        return str(value)

    return "Completed"


def compare_all_outputs(
    expected_values_path: str = (
        "outputs/json/expected_values.json"
    ),
    expected_source: str = (
        "User-provided Expected Values"
    ),
    output_csv_path: str = (
        "outputs/reports/"
        "comparison_report.csv"
    ),
) -> pd.DataFrame:
    Path(
        "outputs/json"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        "outputs/reports"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_path = Path(
        expected_values_path
    )

    if not expected_path.is_file():
        raise FileNotFoundError(
            "Expected Values were not found. "
            "Upload a verified Expected JSON "
            "or save the manual CIOMS form "
            "before running comparison."
        )

    raw_expected = load_json(
        expected_path
    )

    reference = {
        field: raw_expected.get(
            field,
            "",
        )
        for field in ALL_FIELDS
    }

    rows = []

    for (
        approach_name,
        prediction_path,
    ) in APPROACH_FILES.items():
        prediction_file = Path(
            prediction_path
        )

        if not prediction_file.is_file():
            rows.append(
                {
                    "Approach": approach_name,
                    "Status": "Not Run",
                    "Field Agreement": None,
                    "Precision": None,
                    "Recall": None,
                    "F1 Score": None,
                    "TP": 0,
                    "FP": 0,
                    "FN": 0,
                    "TN": 0,
                    "Total Fields": len(
                        reference
                    ),
                    "Matched Fields": 0,
                    "Incorrect Values": 0,
                    "Missing Values": 0,
                    "Unexpected Values": 0,
                    "Wrong/Missing Fields": (
                        len(reference)
                    ),
                }
            )

            continue

        raw_prediction = load_json(
            prediction_file
        )

        prediction = clean_prediction(
            raw_prediction
        )

        run_status = get_run_status(
            raw_prediction
        )

        metrics, field_results = (
            evaluate_extraction(
                reference,
                prediction,
            )
        )

        rows.append(
            {
                "Approach": approach_name,
                "Status": run_status,
                "Field Agreement": metrics[
                    "field_agreement"
                ],
                "Precision": metrics[
                    "precision"
                ],
                "Recall": metrics[
                    "recall"
                ],
                "F1 Score": metrics[
                    "f1_score"
                ],
                "TP": metrics[
                    "true_positive"
                ],
                "FP": metrics[
                    "false_positive"
                ],
                "FN": metrics[
                    "false_negative"
                ],
                "TN": metrics[
                    "true_negative"
                ],
                "Total Fields": metrics[
                    "total_fields"
                ],
                "Matched Fields": metrics[
                    "matched_fields"
                ],
                "Incorrect Values": metrics[
                    "incorrect_values"
                ],
                "Missing Values": metrics[
                    "missing_values"
                ],
                "Unexpected Values": metrics[
                    "unexpected_values"
                ],
                "Wrong/Missing Fields": (
                    metrics[
                        "missed_or_wrong_fields"
                    ]
                ),
            }
        )

        evaluation_file = (
            EVALUATION_FILE_NAMES[
                approach_name
            ]
        )

        save_json(
            {
                "approach": approach_name,
                "expected_source": (
                    expected_source
                ),
                "metrics": metrics,
                "field_results": (
                    field_results
                ),
            },
            str(
                Path("outputs/json")
                / evaluation_file
            ),
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        output_csv_path,
        index=False,
    )

    return dataframe


if __name__ == "__main__":
    result_dataframe = (
        compare_all_outputs()
    )

    print(
        "Comparison completed."
    )

    print(
        result_dataframe
    )
