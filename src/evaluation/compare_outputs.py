import re
import pandas as pd
from pathlib import Path

from src.utils.json_utils import load_json, save_json
from src.evaluation.metrics import evaluate_extraction


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
    "Python Reference": "plain_python_reference_evaluation.json",
    "OCR + Python": "ocr_plus_python_evaluation.json",
    "Custom ML + Python": "custom_ml_plus_python_evaluation.json",
    "LLM + Python": "llm_plus_python_evaluation.json",
}


def clean_prediction(data):
    return {
        key: value
        for key, value in data.items()
        if key not in META_KEYS
    }


def get_run_status(raw_data):
    for key in ["ocr_status", "custom_ml_status", "llm_status"]:
        value = raw_data.get(key)
        if value:
            if str(value).lower() == "success":
                return "Completed"
            if str(value).lower().startswith("success"):
                return "Completed"
            return str(value)

    return "Completed"


def compare_all_outputs(
    plain_output_path="outputs/json/extracted_plain_python.json",
    output_csv_path="outputs/reports/comparison_report.csv",
):
    Path("outputs/json").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    if not Path(plain_output_path).exists():
        raise FileNotFoundError(
            "Python Reference output not found. Run Python extraction first."
        )

    reference = clean_prediction(load_json(plain_output_path))

    approach_files = {
        "Python Reference": "outputs/json/extracted_plain_python.json",
        "OCR + Python": "outputs/json/extracted_ocr_python.json",
        "Custom ML + Python": "outputs/json/extracted_custom_ml.json",
        "LLM + Python": "outputs/json/extracted_llm.json",
    }

    rows = []

    for approach_name, prediction_path in approach_files.items():
        if not Path(prediction_path).exists():
            rows.append(
                {
                    "Approach": approach_name,
                    "Status": "Not Run",
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "F1 Score": None,
                    "Total Fields": len(reference),
                    "Matched Fields": 0,
                    "Wrong/Missing Fields": len(reference),
                }
            )
            continue

        raw_prediction = load_json(prediction_path)
        prediction = clean_prediction(raw_prediction)
        run_status = get_run_status(raw_prediction)

        metrics, field_results = evaluate_extraction(reference, prediction)

        rows.append(
            {
                "Approach": approach_name,
                "Status": run_status,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "Total Fields": metrics["total_fields"],
                "Matched Fields": metrics["matched_fields"],
                "Wrong/Missing Fields": metrics["missed_or_wrong_fields"],
            }
        )

        eval_file = EVALUATION_FILE_NAMES[approach_name]

        save_json(
            {
                "approach": approach_name,
                "reference_source": "Python extraction baseline",
                "metrics": metrics,
                "field_results": field_results,
            },
            f"outputs/json/{eval_file}",
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    return df


if __name__ == "__main__":
    result_df = compare_all_outputs()
    print("Comparison completed.")
    print(result_df)
