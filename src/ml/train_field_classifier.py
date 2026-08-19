from pathlib import Path
import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.field_schema import FIELD_SCHEMA, CHECKBOX_SCHEMA, ALL_FIELDS


MODEL_PATH = "models/field_classifier.joblib"
LABELS_PATH = "models/field_classifier_labels.json"


TEXT_FIELD_EXAMPLES = {
    "patient_initials": [
        "Initials AB",
        "Initials RK",
        "Patient initials AB",
        "Patient initials RK",
        "AB",
        "RK"
    ],
    "country": [
        "Country India",
        "Country United States",
        "Country Germany",
        "India",
        "United States"
    ],
    "date_of_birth_day": [
        "Date of birth day 12",
        "DOB day 08",
        "Day 12",
        "Birth day 21"
    ],
    "date_of_birth_month": [
        "Date of birth month 05",
        "DOB month 08",
        "Month 05",
        "Birth month 11"
    ],
    "date_of_birth_year": [
        "Date of birth year 1985",
        "DOB year 1991",
        "Year 1985",
        "Birth year 1978"
    ],
    "age": [
        "Age 41",
        "Age 55",
        "Patient age 41",
        "Age 29"
    ],
    "sex": [
        "Sex M",
        "Sex F",
        "Patient sex M",
        "Patient sex F"
    ],
    "reaction_onset_day": [
        "Reaction onset day 14",
        "Onset day 14",
        "Day2 14"
    ],
    "reaction_onset_month": [
        "Reaction onset month 08",
        "Onset month 08",
        "Month2 08"
    ],
    "reaction_onset_year": [
        "Reaction onset year 2026",
        "Onset year 2026",
        "Year2 2026"
    ],
    "reaction_description": [
        "Patient developed skin rash and fever after starting therapy",
        "Mild maculopapular rash and low-grade fever",
        "Describe reaction rash fever itching",
        "Relevant tests eosinophils mildly elevated"
    ],
    "suspect_drugs": [
        "Suspect drug Amoxicillin 500 mg capsules",
        "Suspect drugs Aspirin Crocin Pantoprazole",
        "Amoxicillin 500 mg capsules",
        "Aspirin, Crocin, Pantoprazole"
    ],
    "daily_doses": [
        "Daily dose 500 mg twice daily",
        "Daily doses Aspirin 75 mg once daily",
        "500 mg twice daily",
        "75 mg OD"
    ],
    "routes_of_administration": [
        "Route Oral",
        "Routes of administration Oral",
        "Oral",
        "Intravenous"
    ],
    "indications": [
        "Indication acute respiratory tract infection",
        "Indications fever body pains",
        "Acute respiratory tract infection",
        "Fever and body pains"
    ],
    "therapy_dates": [
        "Therapy dates 01/08/2026 to 10/08/2026",
        "Treatment dates 01/09/2025 to 31/07/2026",
        "01/08/2026 to 10/08/2026"
    ],
    "therapy_duration": [
        "Therapy duration 10 days",
        "Duration 11 months",
        "10 days",
        "11 months"
    ],
    "concomitant_drugs": [
        "Concomitant drugs Paracetamol Cetirizine",
        "Paracetamol 500 mg as needed",
        "Cetirizine 10 mg once daily"
    ],
    "history": [
        "No known drug allergies",
        "No relevant chronic medical history",
        "Non-smoker",
        "Other relevant history no allergies"
    ],
    "manufacturer_name_address": [
        "Manufacturer name and address ABC Pharma Pvt Ltd Mumbai India",
        "ABC Pharma Pvt. Ltd. Mumbai Maharashtra India",
        "Manufacturer address Mumbai India"
    ],
    "mfr_control_no": [
        "MFR control number MFR-2026-4587",
        "Control MFR-2026-4587",
        "Manufacturer control number MFR-2026-4587"
    ],
    "date_received": [
        "Date received 15/08/2026",
        "Date received by manufacturer 15/08/2026",
        "15/08/2026"
    ],
    "report_date": [
        "Report date 19/08/2026",
        "Date of this report 19/08/2026",
        "19/08/2026"
    ],
}


CHECKBOX_EXAMPLES = {
    "reaction_patient_died": [
        "Check1 Yes patient died",
        "Check1 Off patient died",
        "reaction patient died yes",
        "reaction patient died off"
    ],
    "reaction_hospitalisation": [
        "Check2 Yes hospitalisation",
        "Check2 Off hospitalisation",
        "reaction hospitalization yes",
        "reaction hospitalization off"
    ],
    "reaction_disability": [
        "Check3 Yes disability",
        "Check3 Off disability",
        "reaction disability yes",
        "reaction disability off"
    ],
    "reaction_life_threatening": [
        "Check4 Yes life threatening",
        "Check4 Off life threatening",
        "life threatening yes",
        "life threatening off"
    ],
    "reaction_abated_yes": [
        "Check5 Yes reaction abated yes",
        "reaction abated yes",
        "dechallenge yes"
    ],
    "reaction_abated_no": [
        "Check6 Yes reaction abated no",
        "reaction abated no",
        "dechallenge no"
    ],
    "reaction_abated_na": [
        "Check7 Yes reaction abated not applicable",
        "reaction abated na",
        "dechallenge not applicable"
    ],
    "reaction_reappeared_yes": [
        "Check8 Yes reaction reappeared yes",
        "reaction reappeared yes",
        "rechallenge yes"
    ],
    "reaction_reappeared_no": [
        "Check9 Yes reaction reappeared no",
        "reaction reappeared no",
        "rechallenge no"
    ],
    "reaction_reappeared_na": [
        "Check10 Yes reaction reappeared not applicable",
        "reaction reappeared na",
        "rechallenge not applicable"
    ],
    "report_source_study": [
        "Check11 Yes report source study",
        "report source study yes",
        "study report"
    ],
    "report_source_literature": [
        "Check12 Yes report source literature",
        "report source literature yes",
        "literature report"
    ],
    "report_type_initial": [
        "Check13 Yes initial report",
        "report type initial yes",
        "initial report"
    ],
    "report_type_followup": [
        "Check14 Yes follow up report",
        "report type followup yes",
        "follow up report"
    ],
    "report_source_health_professional": [
        "Check15 Yes health professional",
        "report source health professional yes",
        "health professional report"
    ],
}


def build_training_data():
    rows = []
    labels = []

    for field_name, examples in TEXT_FIELD_EXAMPLES.items():
        pdf_field = FIELD_SCHEMA.get(field_name, {}).get("pdf_field", "")
        label_text = FIELD_SCHEMA.get(field_name, {}).get("label", "")

        for example in examples:
            rows.append(example)
            labels.append(field_name)

            rows.append(f"{pdf_field} {example}")
            labels.append(field_name)

            rows.append(f"{label_text} {example}")
            labels.append(field_name)

    for field_name, examples in CHECKBOX_EXAMPLES.items():
        pdf_field = CHECKBOX_SCHEMA.get(field_name, "")

        for example in examples:
            rows.append(example)
            labels.append(field_name)

            rows.append(f"{pdf_field} {example}")
            labels.append(field_name)

    return rows, labels


def train_and_save_model():
    Path("models").mkdir(parents=True, exist_ok=True)

    train_x, train_y = build_training_data()

    model = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    lowercase=True
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced"
                )
            )
        ]
    )

    model.fit(train_x, train_y)

    joblib.dump(model, MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(ALL_FIELDS, file, indent=4)

    return MODEL_PATH


if __name__ == "__main__":
    path = train_and_save_model()
    print(f"Custom ML model trained and saved: {path}")
