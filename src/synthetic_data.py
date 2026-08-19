import json
from pathlib import Path


def generate_synthetic_cioms_data():
    return {
        "patient_initials": "RK",
        "country": "India",
        "date_of_birth_day": "12",
        "date_of_birth_month": "05",
        "date_of_birth_year": "1985",
        "age": "41",
        "sex": "M",
        "reaction_onset_day": "14",
        "reaction_onset_month": "08",
        "reaction_onset_year": "2026",
        "reaction_description": "Patient developed skin rash, mild fever, and itching after starting therapy. No airway symptoms were reported. Eosinophils were mildly elevated. Symptoms improved after stopping the suspect drugs.",
        "reaction_patient_died": "Off",
        "reaction_hospitalisation": "Off",
        "reaction_disability": "Off",
        "reaction_life_threatening": "Off",
        "suspect_drugs": "Aspirin, Crocin, Pantoprazole",
        "daily_doses": "Aspirin 75 mg OD, Crocin 500 mg SOS, Pantoprazole 40 mg OD",
        "routes_of_administration": "Oral",
        "indications": "Fever, body pains, and gastric protection",
        "therapy_dates": "01/09/2025 to 31/07/2026",
        "therapy_duration": "11 months",
        "reaction_abated_yes": "Yes",
        "reaction_abated_no": "Off",
        "reaction_abated_na": "Off",
        "reaction_reappeared_yes": "Off",
        "reaction_reappeared_no": "Yes",
        "reaction_reappeared_na": "Off",
        "concomitant_drugs": "Vitamin D3 weekly, Cetirizine 10 mg as needed",
        "history": "No known drug allergies. No significant chronic medical history reported.",
        "manufacturer_name_address": "ABC Pharma Pvt. Ltd., Mumbai, Maharashtra, India",
        "mfr_control_no": "MFR-2026-4587",
        "date_received": "15/08/2026",
        "report_source_study": "Off",
        "report_source_literature": "Off",
        "report_source_health_professional": "Yes",
        "report_date": "19/08/2026",
        "report_type_initial": "Yes",
        "report_type_followup": "Off"
    }


def save_ground_truth(output_path="data/ground_truth.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_cioms_data()
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    return data


if __name__ == "__main__":
    saved_data = save_ground_truth()
    print("Ground truth JSON created successfully.")
    print(saved_data)
