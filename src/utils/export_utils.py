from pathlib import Path
import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.utils.json_utils import load_json


EVALUATION_FILES = {
    "Plain Python Reference": "outputs/json/plain_python_reference_evaluation.json",
    "OCR + Python": "outputs/json/ocr_plus_python_evaluation.json",
    "Custom ML + Python": "outputs/json/custom_ml_plus_python_evaluation.json",
    "LLM + Python": "outputs/json/llm_plus_python_evaluation.json",
}


def build_field_level_dataframe():
    rows = []

    for approach_name, evaluation_path in EVALUATION_FILES.items():
        if not Path(evaluation_path).exists():
            continue

        evaluation = load_json(evaluation_path)

        for item in evaluation.get("field_results", []):
            rows.append(
                {
                    "Approach": approach_name,
                    "Field": item.get("field", ""),
                    "Expected": item.get("expected", ""),
                    "Actual": item.get("actual", ""),
                    "Status": item.get("status", ""),
                    "Match": item.get("match", ""),
                }
            )

    return pd.DataFrame(rows)


def build_confusion_dataframe():
    rows = []

    for approach_name, evaluation_path in EVALUATION_FILES.items():
        if not Path(evaluation_path).exists():
            continue

        evaluation = load_json(evaluation_path)
        matrix = evaluation["metrics"]["confusion_matrix"]

        rows.append(
            {
                "Approach": approach_name,
                "Correctly Extracted": matrix[0][0],
                "Wrong or Missing": matrix[0][1],
                "Actual Incorrect / Predicted Correct": matrix[1][0],
                "Actual Incorrect / Predicted Incorrect": matrix[1][1],
            }
        )

    return pd.DataFrame(rows)


def create_csv_exports():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    field_df = build_field_level_dataframe()
    confusion_df = build_confusion_dataframe()

    field_path = Path("outputs/reports/field_level_report.csv")
    confusion_path = Path("outputs/reports/confusion_summary.csv")

    if not field_df.empty:
        field_df.to_csv(field_path, index=False)

    if not confusion_df.empty:
        confusion_df.to_csv(confusion_path, index=False)

    return field_path, confusion_path


def create_excel_export():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    excel_path = Path("outputs/reports/extraction_quality_report.xlsx")

    summary_path = Path("outputs/reports/comparison_report.csv")
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    field_df = build_field_level_dataframe()
    confusion_df = build_confusion_dataframe()

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        if not field_df.empty:
            field_df.to_excel(writer, sheet_name="Field Level Report", index=False)

        if not confusion_df.empty:
            confusion_df.to_excel(writer, sheet_name="Confusion Summary", index=False)

    return excel_path


def _add_table_page(pdf, title, df, max_rows=30):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=12)
    else:
        display_df = df.head(max_rows).copy()
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_pdf_export():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    pdf_path = Path("outputs/reports/extraction_quality_report.pdf")

    summary_path = Path("outputs/reports/comparison_report.csv")
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    field_df = build_field_level_dataframe()
    confusion_df = build_confusion_dataframe()

    with PdfPages(pdf_path) as pdf:
        _add_table_page(pdf, "Extraction Quality Summary", summary_df, max_rows=20)
        _add_table_page(pdf, "Confusion Summary", confusion_df, max_rows=20)
        _add_table_page(pdf, "Field-Level Report Preview", field_df, max_rows=25)

    return pdf_path
