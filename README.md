# CIOMS PDF Extraction Quality Comparison

## Project Goal
This is one end-to-end project for CIOMS PDF automation.

## Requirement
1. Fill CIOMS PDF with synthetic random relevant sample data.
2. Extract values from the filled PDF using four approaches.
3. Compare output quality using confusion matrix and metrics.

## Four Approaches
- Plain Python
- OCR + Python
- Custom ML + Python
- LLM + Python

## Important Design Decision
This is one single project, not four separate projects.
All four approaches will use the same filled PDF and same ground truth JSON.

## Pipeline
1. Load blank CIOMS PDF.
2. Generate synthetic sample data.
3. Fill CIOMS PDF.
4. Save filled PDF.
5. Save ground truth JSON.
6. Run Plain Python extractor.
7. Run OCR Python extractor.
8. Run Custom ML extractor.
9. Run LLM extractor.
10. Compare outputs using confusion matrix, accuracy, precision, recall, and F1 score.

## Project Structure
OCR/
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .env.example
|-- cioms-form.pdf
|-- data/
|   |-- blank_cioms.pdf
|-- outputs/
|   |-- confusion_matrices/
|   |-- reports/
|   |-- json/
|   |-- images/
|-- src/
|   |-- config.py
|   |-- field_schema.py
|   |-- synthetic_data.py
|   |-- fill_pdf.py
|   |-- extractors/
|   |   |-- plain_python_extractor.py
|   |   |-- ocr_python_extractor.py
|   |   |-- custom_ml_extractor.py
|   |   |-- llm_extractor.py
|   |-- evaluation/
|   |   |-- metrics.py
|   |   |-- compare_outputs.py
|   |-- utils/
|   |   |-- json_utils.py
|   |   |-- pdf_utils.py
|   |   |-- text_utils.py
|   |-- ml/
|   |-- llm/

## Run
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
