import io
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.field_schema import ALL_FIELDS
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

st.warning(
    "Demo app for synthetic/sample PDFs only. Do not upload confidential, personal, medical, client, or production data."
)

st.write(
    "Upload a filled PDF form, run extraction using different approaches, compare outputs, and export the quality report."
)

Path("data").mkdir(parents=True, exist_ok=True)
Path("outputs/json").mkdir(parents=True, exist_ok=True)
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/images").mkdir(parents=True, exist_ok=True)


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
        output_path="outputs/json/extracted_plain_python.json"
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


def build_field_level_dataframe():
    evaluation_files = {
        "Python Reference": "outputs/json/plain_python_reference_evaluation.json",
        "OCR + Python": "outputs/json/ocr_plus_python_evaluation.json",
        "Custom ML + Python": "outputs/json/custom_ml_plus_python_evaluation.json",
        "LLM + Python": "outputs/json/llm_plus_python_evaluation.json",
    }

    rows = []

    for approach_name, evaluation_path in evaluation_files.items():
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
    evaluation_files = {
        "Python Reference": "outputs/json/plain_python_reference_evaluation.json",
        "OCR + Python": "outputs/json/ocr_plus_python_evaluation.json",
        "Custom ML + Python": "outputs/json/custom_ml_plus_python_evaluation.json",
        "LLM + Python": "outputs/json/llm_plus_python_evaluation.json",
    }

    rows = []

    for approach_name, evaluation_path in evaluation_files.items():
        if not Path(evaluation_path).exists():
            continue

        evaluation = load_json(evaluation_path)
        matrix = evaluation["metrics"]["confusion_matrix"]

        rows.append(
            {
                "Approach": approach_name,
                "Correctly Extracted": matrix[0][0],
                "Wrong or Missing": matrix[0][1],
            }
        )

    return pd.DataFrame(rows)


def create_excel_bytes(summary_df, field_df, confusion_df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        if not field_df.empty:
            field_df.to_excel(writer, sheet_name="Field Level Report", index=False)

        if not confusion_df.empty:
            confusion_df.to_excel(writer, sheet_name="Confusion Summary", index=False)

    output.seek(0)
    return output


def create_pdf_bytes(summary_df, field_df, confusion_df):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    output = io.BytesIO()

    def add_table_page(pdf, title, df, max_rows=25):
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(title, fontsize=16, pad=20)

        if df.empty:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                fontsize=12
            )
        else:
            display_df = df.head(max_rows)

            table = ax.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                loc="center",
                cellLoc="left"
            )

            table.auto_set_font_size(False)
            table.set_fontsize(7)
            table.scale(1, 1.3)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    with PdfPages(output) as pdf:
        add_table_page(pdf, "Extraction Quality Summary", summary_df)
        add_table_page(pdf, "Confusion Summary", confusion_df)
        add_table_page(pdf, "Field-Level Report Preview", field_df)

    output.seek(0)
    return output


upload_tab, extract_tab, compare_tab = st.tabs(
    [
        "1. Upload PDF",
        "2. Run Extraction",
        "3. Compare & Export",
    ]
)


with upload_tab:
    st.header("Upload Filled PDF Form")

    uploaded_pdf = st.file_uploader(
        "Upload filled PDF form",
        type=["pdf"],
        key="filled_pdf_upload"
    )

    if uploaded_pdf is not None:
        uploaded_bytes = uploaded_pdf.getvalue()
        uploaded_hash = hashlib.md5(uploaded_bytes).hexdigest()

        if st.session_state.get("uploaded_pdf_hash") != uploaded_hash:
            Path("data/filled_form.pdf").write_bytes(uploaded_bytes)
            clear_previous_outputs()
            st.session_state["uploaded_pdf_hash"] = uploaded_hash
            st.success("PDF uploaded successfully. Previous extraction outputs were cleared.")
        else:
            st.info("Same uploaded PDF is already loaded.")

    if Path("data/filled_form.pdf").exists():
        st.info("PDF is ready for extraction.")

        with open("data/filled_form.pdf", "rb") as file:
            st.download_button(
                "Download Uploaded PDF",
                file,
                file_name="uploaded_filled_form.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("No PDF uploaded yet.")


with extract_tab:
    st.header("Run Extraction")

    plain_tab, ocr_tab, ml_tab, llm_tab = st.tabs(
        [
            "Python",
            "OCR + Python",
            "Custom ML + Python",
            "LLM + Python",
        ]
    )

    with plain_tab:
        st.subheader("Python Extraction")

        if st.button("Run Python Extraction"):
            if not Path("data/filled_form.pdf").exists():
                st.error("Please upload the filled PDF form first.")
            else:
                result = run_with_progress(
                    "Python Extraction",
                    run_plain_python
                )
                st.json(clean_display_output(result))

    with ocr_tab:
        st.subheader("OCR + Python Extraction")

        if st.button("Run OCR Extraction"):
            if not Path("data/filled_form.pdf").exists():
                st.error("Please upload the filled PDF form first.")
            else:
                result = run_with_progress(
                    "OCR extraction",
                    run_ocr_python
                )

                if str(result.get("ocr_status", "")).lower().startswith("failed"):
                    st.warning(result.get("ocr_status"))

                st.json(clean_display_output(result))

        if Path("outputs/reports/ocr_raw_text.txt").exists():
            with st.expander("View OCR Raw Text"):
                st.text(
                    Path("outputs/reports/ocr_raw_text.txt").read_text(
                        encoding="utf-8"
                    )
                )

    with ml_tab:
        st.subheader("Custom ML + Python Extraction")

        if st.button("Run Custom ML Extraction"):
            if not Path("data/filled_form.pdf").exists():
                st.error("Please upload the filled PDF form first.")
            else:
                result = run_with_progress(
                    "Custom ML extraction",
                    run_custom_ml
                )
                st.json(clean_display_output(result))

    with llm_tab:
        st.subheader("LLM + Python Extraction")

        if st.button("Run LLM Extraction"):
            if not Path("data/filled_form.pdf").exists():
                st.error("Please upload the filled PDF form first.")
            else:
                result = run_with_progress(
                    "LLM extraction",
                    run_llm
                )

                if result.get("llm_status", "").lower() != "success":
                    st.warning(
                        result.get(
                            "llm_status",
                            "LLM extraction completed with warning."
                        )
                    )

                st.json(clean_display_output(result))


with compare_tab:
    st.header("Compare & Export Results")

    st.write(
        "Comparison uses Python output as the reference baseline for this demo."
    )

    if st.button("Run Full Comparison"):
        if not Path("data/filled_form.pdf").exists():
            st.error("Please upload the filled PDF form first.")
        else:
            run_with_progress(
                "Python Reference extraction",
                run_plain_python
            )

            run_with_progress(
                "OCR + Python extraction",
                run_ocr_python
            )

            run_with_progress(
                "Custom ML + Python extraction",
                run_custom_ml
            )

            run_with_progress(
                "LLM + Python extraction",
                run_llm
            )

            run_with_progress(
                "Quality comparison",
                compare_all_outputs
            )

            field_df = build_field_level_dataframe()
            confusion_df = build_confusion_dataframe()

            field_df.to_csv("outputs/reports/field_level_report.csv", index=False)
            confusion_df.to_csv("outputs/reports/confusion_summary.csv", index=False)

            st.success("Comparison results are ready below.")

    if Path("outputs/reports/comparison_report.csv").exists():
        summary_df = pd.read_csv("outputs/reports/comparison_report.csv")
        field_df = build_field_level_dataframe()
        confusion_df = build_confusion_dataframe()

        st.subheader("Comparison Summary")
        st.dataframe(summary_df, width="stretch")

        st.subheader("Confusion Summary")
        st.dataframe(confusion_df, width="stretch")

        with st.expander("Field-Level Report"):
            st.dataframe(field_df, width="stretch")

        st.subheader("Export Results")

        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        excel_bytes = create_excel_bytes(summary_df, field_df, confusion_df)
        pdf_bytes = create_pdf_bytes(summary_df, field_df, confusion_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "Download CSV",
                csv_bytes,
                file_name="comparison_report.csv",
                mime="text/csv"
            )

        with col2:
            st.download_button(
                "Download Excel",
                excel_bytes,
                file_name="extraction_quality_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col3:
            st.download_button(
                "Download PDF",
                pdf_bytes,
                file_name="extraction_quality_report.pdf",
                mime="application/pdf"
            )
