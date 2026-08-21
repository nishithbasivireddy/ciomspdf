import io
import json
import base64
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
import pymupdf as fitz
import streamlit.components.v1 as st_components

from src.field_schema import ALL_FIELDS, CHECKBOX_SCHEMA, FIELD_SCHEMA
from src.components.cioms_expected_form.component import (
    blank_expected_values,
    normalize_expected_values,
    render_cioms_expected_form,
    validate_expected_values,
)
from src.extractors.plain_python_extractor import extract_plain_python
from src.extractors.ocr_python_extractor import extract_ocr_python
from src.extractors.custom_ml_extractor import extract_custom_ml
from src.extractors.llm_extractor import extract_llm
from src.evaluation.compare_outputs import compare_all_outputs
from src.utils.json_utils import load_json, save_json

st.set_page_config(
    page_title="PDF Form Extraction Quality Comparison",
    layout="wide"
)

st.title("PDF Form Extraction Quality Comparison")

Path("data").mkdir(parents=True, exist_ok=True)
Path("outputs/json").mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/images").mkdir(parents=True, exist_ok=True)


if "expected_values" not in st.session_state:
    st.session_state["expected_values"] = (
        blank_expected_values()
    )

if "expected_source" not in st.session_state:
    st.session_state["expected_source"] = (
        "Not provided"
    )

if "comparison_completed" not in st.session_state:
    st.session_state["comparison_completed"] = False



def clear_previous_outputs():
    for folder in ["outputs/json", "outputs/reports", "outputs/images"]:
        for file_path in Path(folder).glob("*"):
            if file_path.is_file():
                file_path.unlink(missing_ok=True)


def clean_display_output(data):
    hidden_keys = {
        "llm_status",
        "llm_model",
        "custom_ml_status",
        "ocr_status",
        "ocr_raw_text_available",
        "pdf_text_available",
        "raw_llm_response",
        "tesseract_path",
    }

    return {
        key: value
        for key, value in data.items()
        if key not in hidden_keys
    }


def run_with_progress(label, function_call):
    progress_bar = st.progress(0)
    status_box = st.empty()

    status_box.info(f"{label} started...")
    progress_bar.progress(20)

    with st.spinner(f"{label} in progress..."):
        progress_bar.progress(60)
        result = function_call()
        progress_bar.progress(90)

    progress_bar.progress(100)
    status_box.success(f"{label} completed.")

    return result


def run_plain_python():
    return extract_plain_python(
        pdf_path="data/filled_form.pdf",
        output_path="outputs/json/extracted_plain_python.json",
        validation_path="outputs/json/plain_python_reference_validation.json",
    )


def run_ocr_python():
    try:
        return extract_ocr_python(
            pdf_path="data/filled_form.pdf",
            output_path="outputs/json/extracted_ocr_python.json"
        )
    except Exception as error:
        fail_output = {field: "" for field in ALL_FIELDS}
        fail_output["ocr_status"] = f"Failed: {str(error)}"
        save_json(fail_output, "outputs/json/extracted_ocr_python.json")
        return fail_output


def run_custom_ml():
    return extract_custom_ml(
        pdf_path="data/filled_form.pdf",
        output_path="outputs/json/extracted_custom_ml.json"
    )


def run_llm():
    return extract_llm(
        pdf_path="data/filled_form.pdf",
        output_path="outputs/json/extracted_llm.json"
    )


CHECKBOX_FIELD_LABELS = {
    "reaction_patient_died": (
        "7. Seriousness - Patient Died"
    ),
    "reaction_hospitalisation": (
        "7. Seriousness - Hospitalisation"
    ),
    "reaction_disability": (
        "7. Seriousness - Disability or Incapacity"
    ),
    "reaction_life_threatening": (
        "7. Seriousness - Life Threatening"
    ),
    "reaction_abated_yes": (
        "20. Reaction Abated - Yes"
    ),
    "reaction_abated_no": (
        "20. Reaction Abated - No"
    ),
    "reaction_abated_na": (
        "20. Reaction Abated - Not Applicable"
    ),
    "reaction_reappeared_yes": (
        "21. Reaction Reappeared - Yes"
    ),
    "reaction_reappeared_no": (
        "21. Reaction Reappeared - No"
    ),
    "reaction_reappeared_na": (
        "21. Reaction Reappeared - Not Applicable"
    ),
    "report_source_study": (
        "24d. Report Source - Study"
    ),
    "report_source_literature": (
        "24d. Report Source - Literature"
    ),
    "report_type_initial": (
        "25a. Report Type - Initial"
    ),
    "report_type_followup": (
        "25a. Report Type - Follow-up"
    ),
    "report_source_health_professional": (
        "24d. Report Source - Health Professional"
    ),
}


def get_field_display_label(
    field: str,
) -> str:
    if field in FIELD_SCHEMA:
        return FIELD_SCHEMA[field].get(
            "label",
            field,
        )

    return CHECKBOX_FIELD_LABELS.get(
        field,
        field.replace("_", " ").title(),
    )


def build_field_level_dataframe():
    expected_path = Path(
        "outputs/json/expected_values.json"
    )

    if not expected_path.is_file():
        return pd.DataFrame()

    expected = load_json(expected_path)

    evaluation_files = {
        "Plain Python": (
            "outputs/json/"
            "plain_python_evaluation.json"
        ),
        "OCR + Python": (
            "outputs/json/"
            "ocr_plus_python_evaluation.json"
        ),
        "Custom ML + Python": (
            "outputs/json/"
            "custom_ml_plus_python_evaluation.json"
        ),
        "LLM Schema Mapping": (
            "outputs/json/"
            "llm_schema_mapping_evaluation.json"
        ),
    }

    results_by_approach = {}

    for approach, evaluation_path in (
        evaluation_files.items()
    ):
        if not Path(evaluation_path).is_file():
            results_by_approach[approach] = {}
            continue

        evaluation = load_json(
            evaluation_path
        )

        results_by_approach[approach] = {
            item.get("field", ""): item
            for item in evaluation.get(
                "field_results",
                [],
            )
        }

    rows = []

    for field in ALL_FIELDS:
        row = {
            "Field Key": field,
            "Field": get_field_display_label(
                field
            ),
            "Expected": expected.get(
                field,
                "",
            ),
        }

        for approach in evaluation_files:
            item = results_by_approach[
                approach
            ].get(
                field,
                {},
            )

            row[approach] = item.get(
                "actual",
                "",
            )

            row[
                f"{approach} Status"
            ] = item.get(
                "status",
                "Not Run",
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_confusion_dataframe():
    evaluation_files = {
        "Plain Python": (
            "outputs/json/"
            "plain_python_evaluation.json"
        ),
        "OCR + Python": (
            "outputs/json/"
            "ocr_plus_python_evaluation.json"
        ),
        "Custom ML + Python": (
            "outputs/json/"
            "custom_ml_plus_python_evaluation.json"
        ),
        "LLM Schema Mapping": (
            "outputs/json/"
            "llm_schema_mapping_evaluation.json"
        ),
    }

    rows = []

    for approach, evaluation_path in (
        evaluation_files.items()
    ):
        if not Path(evaluation_path).is_file():
            continue

        evaluation = load_json(
            evaluation_path
        )

        metrics = evaluation.get(
            "metrics",
            {},
        )

        rows.append(
            {
                "Approach": approach,
                "TP": metrics.get(
                    "true_positive",
                    0,
                ),
                "FP": metrics.get(
                    "false_positive",
                    0,
                ),
                "FN": metrics.get(
                    "false_negative",
                    0,
                ),
                "TN": metrics.get(
                    "true_negative",
                    0,
                ),
                "Exact Matches": metrics.get(
                    "matched_fields",
                    0,
                ),
                "Incorrect": metrics.get(
                    "incorrect_values",
                    0,
                ),
                "Missing": metrics.get(
                    "missing_values",
                    0,
                ),
                "Unexpected": metrics.get(
                    "unexpected_values",
                    0,
                ),
            }
        )

    return pd.DataFrame(rows)


def style_field_level_report(
    dataframe: pd.DataFrame,
):
    approach_columns = [
        "Plain Python",
        "OCR + Python",
        "Custom ML + Python",
        "LLM Schema Mapping",
    ]

    status_columns = {
        approach: f"{approach} Status"
        for approach in approach_columns
    }

    visible_columns = [
        "Field",
        "Expected",
        *approach_columns,
    ]

    display_dataframe = dataframe[
        visible_columns
    ].copy()

    def style_row(row):
        styles = [
            ""
            for _ in display_dataframe.columns
        ]

        original_row = dataframe.loc[
            row.name
        ]

        for column_index, column_name in enumerate(
            display_dataframe.columns
        ):
            if column_name == "Field":
                styles[column_index] = (
                    "font-weight: 600; "
                    "background-color: #f3f4f6;"
                )

            elif column_name == "Expected":
                styles[column_index] = (
                    "font-weight: 600; "
                    "background-color: #e5e7eb; "
                    "color: #111827;"
                )

            elif column_name in approach_columns:
                status = str(
                    original_row.get(
                        status_columns[column_name],
                        "",
                    )
                )

                if status in {
                    "Correct",
                    "Correct Blank",
                }:
                    styles[column_index] = (
                        "background-color: #dcfce7; "
                        "color: #166534;"
                    )

                elif status in {
                    "Incorrect",
                    "Missing",
                    "Unexpected",
                }:
                    styles[column_index] = (
                        "background-color: #fee2e2; "
                        "color: #991b1b;"
                    )

                else:
                    styles[column_index] = (
                        "background-color: #fef3c7; "
                        "color: #92400e;"
                    )

        return styles

    return (
        display_dataframe.style
        .apply(
            style_row,
            axis=1,
        )
        .set_properties(
            **{
                "white-space": "pre-wrap",
                "vertical-align": "top",
            }
        )
    )


def create_excel_bytes(
    summary_df,
    field_df,
    confusion_df,
):
    from openpyxl.styles import (
        Alignment,
        Font,
        PatternFill,
    )
    from openpyxl.utils import (
        get_column_letter,
    )

    output = io.BytesIO()

    approach_columns = [
        "Plain Python",
        "OCR + Python",
        "Custom ML + Python",
        "LLM Schema Mapping",
    ]

    status_columns = {
        approach: f"{approach} Status"
        for approach in approach_columns
    }

    visible_field_columns = [
        "Field",
        "Expected",
        *approach_columns,
    ]

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name=(
                "Approach Comparison Summary"
            ),
            index=False,
        )

        field_df[
            visible_field_columns
        ].to_excel(
            writer,
            sheet_name="Field Level Report",
            index=False,
        )

        workbook = writer.book

        summary_sheet = workbook[
            "Approach Comparison Summary"
        ]

        field_sheet = workbook[
            "Field Level Report"
        ]

        percentage_headers = {
            "Field Agreement",
            "Precision",
            "Recall",
            "F1 Score",
        }

        summary_headers = {
            cell.value: cell.column
            for cell in summary_sheet[1]
        }

        for header in percentage_headers:
            column_index = summary_headers.get(
                header
            )

            if column_index is None:
                continue

            for row_index in range(
                2,
                summary_sheet.max_row + 1,
            ):
                summary_sheet.cell(
                    row=row_index,
                    column=column_index,
                ).number_format = "0.00%"

        summary_sheet.row_dimensions[1].height = 42
        field_sheet.row_dimensions[1].height = 38

        for row_index in range(
            2,
            field_sheet.max_row + 1,
        ):
            values = [
                field_sheet.cell(
                    row=row_index,
                    column=column_index,
                ).value
                for column_index in range(
                    1,
                    field_sheet.max_column + 1,
                )
            ]

            maximum_line_count = max(
                (
                    str(value).count("\n") + 1
                    if value not in {None, ""}
                    else 1
                )
                for value in values
            )

            field_sheet.row_dimensions[
                row_index
            ].height = max(
                24,
                min(
                    90,
                    18 * maximum_line_count,
                ),
            )

        header_fill = PatternFill(
            fill_type="solid",
            fgColor="1F2937",
        )

        header_font = Font(
            color="FFFFFF",
            bold=True,
        )

        expected_fill = PatternFill(
            fill_type="solid",
            fgColor="E5E7EB",
        )

        field_fill = PatternFill(
            fill_type="solid",
            fgColor="F3F4F6",
        )

        correct_fill = PatternFill(
            fill_type="solid",
            fgColor="DCFCE7",
        )

        incorrect_fill = PatternFill(
            fill_type="solid",
            fgColor="FEE2E2",
        )

        pending_fill = PatternFill(
            fill_type="solid",
            fgColor="FEF3C7",
        )

        for worksheet in [
            summary_sheet,
            field_sheet,
        ]:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                    wrap_text=True,
                )

        for row_index in range(
            2,
            field_sheet.max_row + 1,
        ):
            field_sheet.cell(
                row=row_index,
                column=1,
            ).fill = field_fill

            field_sheet.cell(
                row=row_index,
                column=1,
            ).font = Font(
                bold=True
            )

            field_sheet.cell(
                row=row_index,
                column=2,
            ).fill = expected_fill

            field_sheet.cell(
                row=row_index,
                column=2,
            ).font = Font(
                bold=True
            )

            source_row = field_df.iloc[
                row_index - 2
            ]

            for approach_index, approach in (
                enumerate(
                    approach_columns,
                    start=3,
                )
            ):
                status = str(
                    source_row.get(
                        status_columns[approach],
                        "Not Run",
                    )
                )

                target_cell = field_sheet.cell(
                    row=row_index,
                    column=approach_index,
                )

                if status in {
                    "Correct",
                    "Correct Blank",
                }:
                    target_cell.fill = (
                        correct_fill
                    )

                elif status in {
                    "Incorrect",
                    "Missing",
                    "Unexpected",
                }:
                    target_cell.fill = (
                        incorrect_fill
                    )

                else:
                    target_cell.fill = (
                        pending_fill
                    )

            for column_index in range(
                1,
                field_sheet.max_column + 1,
            ):
                field_sheet.cell(
                    row=row_index,
                    column=column_index,
                ).alignment = Alignment(
                    vertical="top",
                    wrap_text=True,
                )

        summary_widths = {
            1: 24,
            2: 18,
        }

        for column_index in range(
            3,
            summary_sheet.max_column + 1,
        ):
            summary_widths[
                column_index
            ] = 18

        for column_index, width in (
            summary_widths.items()
        ):
            summary_sheet.column_dimensions[
                get_column_letter(
                    column_index
                )
            ].width = width

        field_widths = {
            1: 34,
            2: 28,
            3: 28,
            4: 28,
            5: 28,
            6: 28,
        }

        for column_index, width in (
            field_widths.items()
        ):
            field_sheet.column_dimensions[
                get_column_letter(
                    column_index
                )
            ].width = width

        field_sheet.sheet_view.showGridLines = (
            False
        )

        summary_sheet.sheet_view.showGridLines = (
            False
        )

    output.seek(0)
    return output


def create_pdf_bytes(
    summary_df,
    field_df,
    confusion_df,
):
    from matplotlib.backends.backend_pdf import (
        PdfPages,
    )
    import matplotlib.pyplot as plt
    import textwrap

    output = io.BytesIO()

    approach_columns = [
        "Plain Python",
        "OCR + Python",
        "Custom ML + Python",
        "LLM Schema Mapping",
    ]

    status_columns = {
        approach: f"{approach} Status"
        for approach in approach_columns
    }

    visible_field_columns = [
        "Field",
        "Expected",
        *approach_columns,
    ]

    colors = {
        "header": "#1f2937",
        "header_text": "#ffffff",
        "field": "#f3f4f6",
        "expected": "#e5e7eb",
        "correct": "#dcfce7",
        "incorrect": "#fee2e2",
        "pending": "#fef3c7",
    }

    def display_text(
        value,
        width=24,
    ):
        if pd.isna(value):
            return ""

        text = str(value)

        return "\n".join(
            textwrap.wrap(
                text,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )

    def percentage_text(value):
        if pd.isna(value) or value == "":
            return ""

        return f"{float(value) * 100:.2f}%"

    def style_header(
        table,
        column_count,
    ):
        for column_index in range(
            column_count
        ):
            cell = table[
                0,
                column_index,
            ]

            cell.set_facecolor(
                colors["header"]
            )

            cell.get_text().set_color(
                colors["header_text"]
            )

            cell.get_text().set_weight(
                "bold"
            )

            cell.get_text().set_wrap(
                True
            )

    def add_summary_page(pdf):
        figure = plt.figure(
            figsize=(16.5, 11.7)
        )

        figure.subplots_adjust(
            left=0.035,
            right=0.965,
            top=0.94,
            bottom=0.05,
            hspace=0.38,
        )

        figure.suptitle(
            "CIOMS Extraction Quality Report",
            fontsize=20,
            fontweight="bold",
            y=0.975,
        )

        figure.text(
            0.5,
            0.94,
            (
                "Expected Values Source: "
                f"{st.session_state['expected_source']}"
            ),
            ha="center",
            fontsize=10,
        )

        figure.text(
            0.5,
            0.918,
            (
                "Evaluation Method: Normalized "
                "Exact-Match Field-Level Comparison"
            ),
            ha="center",
            fontsize=10,
        )

        metrics_axis = figure.add_subplot(
            2,
            1,
            1,
        )

        outcomes_axis = figure.add_subplot(
            2,
            1,
            2,
        )

        metrics_axis.axis("off")
        outcomes_axis.axis("off")

        metrics_axis.set_title(
            "Performance Metrics",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        outcomes_axis.set_title(
            "Field Outcome Counts",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        metrics_columns = [
            "Approach",
            "Status",
            "Field Agreement",
            "Precision",
            "Recall",
            "F1 Score",
        ]

        metrics_display = summary_df[
            [
                column
                for column in metrics_columns
                if column in summary_df.columns
            ]
        ].copy()

        for column in [
            "Field Agreement",
            "Precision",
            "Recall",
            "F1 Score",
        ]:
            if column in metrics_display.columns:
                metrics_display[column] = (
                    metrics_display[column]
                    .apply(percentage_text)
                )

        metrics_labels = [
            "Approach",
            "Status",
            "Field\nAgreement",
            "Precision",
            "Recall",
            "F1 Score",
        ][:len(metrics_display.columns)]

        metrics_table = metrics_axis.table(
            cellText=metrics_display.fillna("").values,
            colLabels=metrics_labels,
            cellLoc="center",
            loc="center",
            bbox=[0.01, 0.08, 0.98, 0.78],
        )

        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(9)

        style_header(
            metrics_table,
            len(metrics_display.columns),
        )

        count_columns = [
            "Approach",
            "True Positive (TP)",
            "False Positive (FP)",
            "False Negative (FN)",
            "True Negative (TN)",
            "Total Fields",
            "Matched Fields",
            "Incorrect Values",
            "Missing Values",
            "Unexpected Values",
        ]

        counts_display = summary_df[
            [
                column
                for column in count_columns
                if column in summary_df.columns
            ]
        ].copy()

        count_labels = {
            "Approach": "Approach",
            "True Positive (TP)": "True Positive\n(TP)",
            "False Positive (FP)": "False Positive\n(FP)",
            "False Negative (FN)": "False Negative\n(FN)",
            "True Negative (TN)": "True Negative\n(TN)",
            "Total Fields": "Total\nFields",
            "Matched Fields": "Exact\nMatches",
            "Incorrect Values": "Incorrect\nValues",
            "Missing Values": "Missing\nValues",
            "Unexpected Values": "Unexpected\nValues",
        }

        counts_table = outcomes_axis.table(
            cellText=counts_display.fillna("").values,
            colLabels=[
                count_labels.get(
                    column,
                    column,
                )
                for column in counts_display.columns
            ],
            cellLoc="center",
            loc="center",
            bbox=[0.01, 0.08, 0.98, 0.78],
        )

        counts_table.auto_set_font_size(False)
        counts_table.set_fontsize(8)

        style_header(
            counts_table,
            len(counts_display.columns),
        )

        pdf.savefig(
            figure,
        )

        plt.close(figure)

    def add_field_page(
        pdf,
        page_dataframe,
        page_number,
        total_pages,
        row_offset,
    ):
        figure, axis = plt.subplots(
            figsize=(16.5, 11.7)
        )

        figure.subplots_adjust(
            left=0.025,
            right=0.975,
            top=0.90,
            bottom=0.055,
        )

        axis.axis("off")

        figure.suptitle(
            (
                "Field-Level Report "
                f"({page_number} of {total_pages})"
            ),
            fontsize=18,
            fontweight="bold",
            y=0.965,
        )

        figure.text(
            0.5,
            0.925,
            (
                "Green = exact match | "
                "Red = incorrect, missing, or unexpected | "
                "Gray = Expected | Amber = not run"
            ),
            ha="center",
            fontsize=9,
        )

        visible_dataframe = page_dataframe[
            visible_field_columns
        ].copy()

        for column in visible_dataframe.columns:
            wrap_width = (
                28
                if column == "Field"
                else 22
            )

            visible_dataframe[column] = (
                visible_dataframe[column]
                .apply(
                    lambda value: display_text(
                        value,
                        wrap_width,
                    )
                )
            )

        column_labels = [
            "Field",
            "Expected",
            "Plain\nPython",
            "OCR +\nPython",
            "Custom ML +\nPython",
            "LLM Schema\nMapping",
        ]

        table = axis.table(
            cellText=visible_dataframe.values,
            colLabels=column_labels,
            cellLoc="left",
            loc="center",
            bbox=[0.005, 0.03, 0.99, 0.85],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(7.5)

        style_header(
            table,
            len(column_labels),
        )

        column_widths = [
            0.20,
            0.16,
            0.16,
            0.16,
            0.16,
            0.16,
        ]

        for (
            row_index,
            column_index,
        ), cell in table.get_celld().items():
            cell.set_width(
                column_widths[column_index]
            )

            cell.get_text().set_wrap(
                True
            )

            cell.get_text().set_verticalalignment(
                "center"
            )

            if row_index > 0:
                cell.set_height(
                    0.095
                )

        for local_row_index in range(
            len(visible_dataframe)
        ):
            table_row = local_row_index + 1

            source_row = field_df.iloc[
                row_offset + local_row_index
            ]

            field_cell = table[
                table_row,
                0,
            ]

            field_cell.set_facecolor(
                colors["field"]
            )

            field_cell.get_text().set_weight(
                "bold"
            )

            expected_cell = table[
                table_row,
                1,
            ]

            expected_cell.set_facecolor(
                colors["expected"]
            )

            expected_cell.get_text().set_weight(
                "bold"
            )

            for approach_column, approach in (
                enumerate(
                    approach_columns,
                    start=2,
                )
            ):
                status = str(
                    source_row.get(
                        status_columns[approach],
                        "Not Run",
                    )
                )

                target_cell = table[
                    table_row,
                    approach_column,
                ]

                if status in {
                    "Correct",
                    "Correct Blank",
                }:
                    target_cell.set_facecolor(
                        colors["correct"]
                    )

                elif status in {
                    "Incorrect",
                    "Missing",
                    "Unexpected",
                }:
                    target_cell.set_facecolor(
                        colors["incorrect"]
                    )

                else:
                    target_cell.set_facecolor(
                        colors["pending"]
                    )

        pdf.savefig(
            figure,
        )

        plt.close(figure)

    with PdfPages(output) as pdf:
        add_summary_page(pdf)

        rows_per_page = 8

        total_pages = max(
            1,
            (
                len(field_df)
                + rows_per_page
                - 1
            )
            // rows_per_page,
        )

        for page_index in range(
            total_pages
        ):
            start_row = (
                page_index * rows_per_page
            )

            end_row = min(
                start_row + rows_per_page,
                len(field_df),
            )

            add_field_page(
                pdf,
                field_df.iloc[
                    start_row:end_row
                ],
                page_index + 1,
                total_pages,
                start_row,
            )

    output.seek(0)
    return output



st.header("Implemented Extraction Approaches")

approach_columns = st.columns(4)

with approach_columns[0]:
    st.markdown("#### Plain Python")
    st.write(
        "Reads embedded PDF form-widget values "
        "using PyMuPDF and maps them into the "
        "38-field CIOMS schema."
    )

with approach_columns[1]:
    st.markdown("#### OCR + Python")
    st.write(
        "Renders the PDF visually, applies OCR, "
        "and maps recognized content into the "
        "38-field CIOMS schema."
    )

with approach_columns[2]:
    st.markdown("#### Custom ML + Python")
    st.write(
        "Uses a fine-tuned CRNN for text fields "
        "and a custom CNN for checkbox fields "
        "from rendered PDF pixels."
    )

with approach_columns[3]:
    st.markdown("#### LLM + Python")
    st.write(
        "Uses an LLM to map Python-extracted PDF "
        "form-field values into the required "
        "38-field JSON schema."
    )

st.divider()

st.header("Filled CIOMS PDF Preview")

demo_pdf_path = Path(
    "demo/cioms-form2.pdf"
)

demo_expected_path = Path(
    "demo/cioms_manual_expected_values.json"
)

runtime_pdf_path = Path(
    "data/filled_form.pdf"
)

if not demo_pdf_path.is_file():
    st.error(
        "Bundled demo PDF is missing: "
        f"{demo_pdf_path}"
    )

    st.stop()

if not demo_expected_path.is_file():
    st.error(
        "Bundled Expected JSON is missing: "
        f"{demo_expected_path}"
    )

    st.stop()

demo_pdf_bytes = demo_pdf_path.read_bytes()

runtime_pdf_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

runtime_pdf_path.write_bytes(
    demo_pdf_bytes
)

demo_expected_values = load_json(
    demo_expected_path
)

expected_validation = (
    validate_expected_values(
        demo_expected_values
    )
)

if any(
    expected_validation.values()
):
    st.error(
        "The bundled Expected JSON does not "
        "match the required 38-field schema."
    )

    st.json(
        expected_validation
    )

    st.stop()

st.session_state[
    "expected_values"
] = normalize_expected_values(
    demo_expected_values
)

st.session_state[
    "expected_source"
] = "Bundled Verified Expected JSON"

try:
    preview_document = fitz.open(
        stream=demo_pdf_bytes,
        filetype="pdf",
    )

    for page_number in range(
        preview_document.page_count
    ):
        preview_page = preview_document[
            page_number
        ]

        preview_pixmap = (
            preview_page.get_pixmap(
                matrix=fitz.Matrix(
                    1.5,
                    1.5,
                ),
                alpha=False,
                annots=True,
            )
        )

        preview_png = (
            preview_pixmap.tobytes(
                "png"
            )
        )

        st.image(
            preview_png,
            caption=(
                f"Filled CIOMS PDF - Page "
                f"{page_number + 1}"
            ),
            width="stretch",
        )

    preview_document.close()

except Exception as preview_error:
    st.error(
        "The bundled PDF could not be "
        "previewed: "
        f"{preview_error}"
    )

    st.stop()

pdf_ready = True
expected_ready = True
extract_ready = True

st.divider()

st.header("Extract and Compare")

run_extraction = st.button(
    "Run",
    type="primary",
    disabled=not extract_ready,
    use_container_width=True,
)

if run_extraction:
    active_expected = (
        normalize_expected_values(
            st.session_state[
                "expected_values"
            ]
        )
    )

    save_json(
        active_expected,
        "outputs/json/expected_values.json",
    )

    status_steps = [
        {
            "name": "Plain Python",
            "icon": "🐍",
            "function": run_plain_python,
        },
        {
            "name": "OCR + Python",
            "icon": "👁",
            "function": run_ocr_python,
        },
        {
            "name": "Custom ML",
            "icon": "🧠",
            "function": run_custom_ml,
        },
        {
            "name": "LLM + Python",
            "icon": "✨",
            "function": run_llm,
        },
        {
            "name": "Comparison",
            "icon": "📊",
            "function": lambda: compare_all_outputs(
                expected_values_path=(
                    "outputs/json/"
                    "expected_values.json"
                ),
                expected_source=(
                    st.session_state[
                        "expected_source"
                    ]
                ),
            ),
        },
    ]

    status_columns = st.columns(5)

    status_placeholders = []

    for column in status_columns:
        with column:
            status_placeholders.append(
                st.empty()
            )

    def render_status_card(
        placeholder,
        icon,
        name,
        state,
    ):
        state_styles = {
            "Pending": {
                "symbol": "○",
                "background": "#f3f4f6",
                "border": "#9ca3af",
                "color": "#4b5563",
            },
            "Running": {
                "symbol": "◌",
                "background": "#eff6ff",
                "border": "#2563eb",
                "color": "#1d4ed8",
            },
            "Completed": {
                "symbol": "✓",
                "background": "#ecfdf5",
                "border": "#16a34a",
                "color": "#15803d",
            },
            "Failed": {
                "symbol": "✕",
                "background": "#fef2f2",
                "border": "#dc2626",
                "color": "#b91c1c",
            },
        }

        style = state_styles[state]

        placeholder.markdown(
            f"""
            <div style="
                min-height: 112px;
                padding: 14px 10px;
                border: 2px solid {style["border"]};
                border-radius: 12px;
                background: {style["background"]};
                text-align: center;
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 5px;
            ">
                <div style="font-size: 25px;">
                    {icon}
                </div>
                <div style="
                    font-size: 14px;
                    font-weight: 700;
                    color: #111827;
                ">
                    {name}
                </div>
                <div style="
                    font-size: 13px;
                    font-weight: 700;
                    color: {style["color"]};
                ">
                    {style["symbol"]} {state}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for index, step in enumerate(
        status_steps
    ):
        render_status_card(
            status_placeholders[index],
            step["icon"],
            step["name"],
            "Pending",
        )

    failed_step = None

    try:
        for index, step in enumerate(
            status_steps
        ):
            render_status_card(
                status_placeholders[index],
                step["icon"],
                step["name"],
                "Running",
            )

            try:
                step["function"]()

                render_status_card(
                    status_placeholders[index],
                    step["icon"],
                    step["name"],
                    "Completed",
                )

            except Exception:
                failed_step = step["name"]

                render_status_card(
                    status_placeholders[index],
                    step["icon"],
                    step["name"],
                    "Failed",
                )

                raise

        st.session_state[
            "comparison_completed"
        ] = True

        st.toast(
            "Extraction and comparison completed.",
            icon="✅",
        )

    except Exception as error:
        st.session_state[
            "comparison_completed"
        ] = False

        st.error(
            f"{failed_step or 'Processing'} failed: "
            f"{error}"
        )

report_ready = (
    st.session_state.get(
        "comparison_completed",
        False,
    )
    and Path(
        "outputs/reports/comparison_report.csv"
    ).is_file()
    and Path(
        "outputs/json/expected_values.json"
    ).is_file()
)

if report_ready:
    st.divider()

    st.header("Field-Level Comparison")

    field_df = (
        build_field_level_dataframe()
    )

    st.caption(
        "Green indicates an exact normalized "
        "match. Red indicates an incorrect, "
        "missing, or unexpected value. Gray "
        "contains the verified Expected value."
    )

    st.dataframe(
        style_field_level_report(
            field_df
        ),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.header(
        "Overall Extraction Quality Matrices"
    )

    st.caption(
        "Each matrix summarizes mutually exclusive "
        "field-level outcomes across all 38 CIOMS "
        "fields. The four cells always total 38. "
        "Exact-value correctness is evaluated against "
        "the verified Expected JSON."
    )

    evaluation_files = {
        "Plain Python": (
            "outputs/json/"
            "plain_python_evaluation.json"
        ),
        "OCR + Python": (
            "outputs/json/"
            "ocr_plus_python_evaluation.json"
        ),
        "Custom ML + Python": (
            "outputs/json/"
            "custom_ml_plus_python_evaluation.json"
        ),
        "LLM + Python": (
            "outputs/json/"
            "llm_schema_mapping_evaluation.json"
        ),
    }

    matrix_columns = st.columns(2)

    for matrix_index, (
        approach,
        evaluation_path,
    ) in enumerate(
        evaluation_files.items()
    ):
        if not Path(
            evaluation_path
        ).is_file():
            continue

        evaluation = load_json(
            evaluation_path
        )

        metrics = evaluation.get(
            "metrics",
            {},
        )

        correct_non_empty = int(
            metrics.get(
                "true_positive",
                0,
            )
        )

        correct_blank = int(
            metrics.get(
                "true_negative",
                0,
            )
        )

        incorrect_value = int(
            metrics.get(
                "incorrect_values",
                0,
            )
        )

        missing_value = int(
            metrics.get(
                "missing_values",
                0,
            )
        )

        unexpected_value = int(
            metrics.get(
                "unexpected_values",
                0,
            )
        )

        missing_or_unexpected = (
            missing_value
            + unexpected_value
        )

        total_fields = int(
            metrics.get(
                "total_fields",
                0,
            )
        )

        outcome_total = (
            correct_non_empty
            + incorrect_value
            + correct_blank
            + missing_or_unexpected
        )

        if outcome_total != total_fields:
            with matrix_columns[
                matrix_index % 2
            ]:
                st.error(
                    f"{approach}: outcome total "
                    f"{outcome_total} does not match "
                    f"the expected {total_fields} "
                    "fields."
                )

            continue

        matrix = [
            [
                correct_non_empty,
                incorrect_value,
            ],
            [
                correct_blank,
                missing_or_unexpected,
            ],
        ]

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        figure, axis = plt.subplots(
            figsize=(5.8, 4.4)
        )

        axis.set_xlim(0, 2)
        axis.set_ylim(0, 2)
        axis.axis("off")

        axis.set_title(
            approach,
            fontweight="bold",
            fontsize=15,
            pad=16,
        )

        outcome_cells = [
            {
                "x": 0,
                "y": 1,
                "label": "Correct Non-Empty",
                "value": correct_non_empty,
                "background": "#dcfce7",
                "border": "#16a34a",
                "text": "#166534",
            },
            {
                "x": 1,
                "y": 1,
                "label": "Incorrect Value",
                "value": incorrect_value,
                "background": "#fee2e2",
                "border": "#dc2626",
                "text": "#991b1b",
            },
            {
                "x": 0,
                "y": 0,
                "label": "Correct Blank",
                "value": correct_blank,
                "background": "#e5e7eb",
                "border": "#6b7280",
                "text": "#374151",
            },
            {
                "x": 1,
                "y": 0,
                "label": "Missing / Unexpected",
                "value": missing_or_unexpected,
                "background": "#fef3c7",
                "border": "#d97706",
                "text": "#92400e",
            },
        ]

        for cell in outcome_cells:
            axis.add_patch(
                Rectangle(
                    (
                        cell["x"],
                        cell["y"],
                    ),
                    1,
                    1,
                    facecolor=cell[
                        "background"
                    ],
                    edgecolor=cell[
                        "border"
                    ],
                    linewidth=2.2,
                )
            )

            axis.text(
                cell["x"] + 0.5,
                cell["y"] + 0.68,
                cell["label"],
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=cell["text"],
                wrap=True,
            )

            axis.text(
                cell["x"] + 0.5,
                cell["y"] + 0.35,
                str(cell["value"]),
                ha="center",
                va="center",
                fontsize=24,
                fontweight="bold",
                color=cell["text"],
            )

        figure.text(
            0.5,
            0.025,
            (
                f"Total evaluated fields: "
                f"{outcome_total}"
            ),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#111827",
        )

        figure.subplots_adjust(
            left=0.04,
            right=0.96,
            top=0.86,
            bottom=0.12,
        )

        with matrix_columns[
            matrix_index % 2
        ]:
            st.pyplot(
                figure,
                use_container_width=True,
            )

            st.caption(
                f"{approach}: "
                f"{correct_non_empty} correct "
                "non-empty, "
                f"{correct_blank} correct blank, "
                f"{incorrect_value} incorrect, "
                f"{missing_value} missing, and "
                f"{unexpected_value} unexpected."
            )

        plt.close(figure)
