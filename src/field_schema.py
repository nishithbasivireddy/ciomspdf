FIELD_SCHEMA = {
    "patient_initials": {"pdf_field": "Initials", "label": "1. Patient Initials"},
    "country": {"pdf_field": "Country", "label": "1a. Country"},
    "date_of_birth_day": {"pdf_field": "Day", "label": "2. Date of Birth - Day"},
    "date_of_birth_month": {"pdf_field": "Month", "label": "2. Date of Birth - Month"},
    "date_of_birth_year": {"pdf_field": "Year", "label": "2. Date of Birth - Year"},
    "age": {"pdf_field": "Age", "label": "2a. Age"},
    "sex": {"pdf_field": "Sex", "label": "3. Sex"},
    "reaction_onset_day": {"pdf_field": "Day2", "label": "4-6 Reaction Onset - Day"},
    "reaction_onset_month": {"pdf_field": "Month2", "label": "4-6 Reaction Onset - Month"},
    "reaction_onset_year": {"pdf_field": "Year2", "label": "4-6 Reaction Onset - Year"},
    "reaction_description": {"pdf_field": "Description", "label": "7 + 13 Describe Reaction"},
    "suspect_drugs": {"pdf_field": "Suspect_Drugs", "label": "14. Suspect Drug(s)"},
    "daily_doses": {"pdf_field": "Daily_Doses", "label": "15. Daily Dose(s)"},
    "routes_of_administration": {"pdf_field": "Routes_of_Admin", "label": "16. Route(s) of Administration"},
    "indications": {"pdf_field": "Indications", "label": "17. Indication(s) for Use"},
    "therapy_dates": {"pdf_field": "Therapy", "label": "18. Therapy Dates"},
    "therapy_duration": {"pdf_field": "Duration", "label": "19. Therapy Duration"},
    "concomitant_drugs": {"pdf_field": "Concomitant", "label": "22. Concomitant Drug(s)"},
    "history": {"pdf_field": "History", "label": "23. Other Relevant History"},
    "manufacturer_name_address": {"pdf_field": "Manu_Name-Add", "label": "24a. Manufacturer Name and Address"},
    "mfr_control_no": {"pdf_field": "Control", "label": "24b. MFR Control No."},
    "date_received": {"pdf_field": "Date_Rec", "label": "24c. Date Received by Manufacturer"},
    "report_date": {"pdf_field": "Report_Date", "label": "Date of This Report"}
}

CHECKBOX_SCHEMA = {
    "reaction_patient_died": "Check1",
    "reaction_hospitalisation": "Check2",
    "reaction_disability": "Check3",
    "reaction_life_threatening": "Check4",
    "reaction_abated_yes": "Check5",
    "reaction_abated_no": "Check6",
    "reaction_abated_na": "Check7",
    "reaction_reappeared_yes": "Check8",
    "reaction_reappeared_no": "Check9",
    "reaction_reappeared_na": "Check10",
    "report_source_study": "Check11",
    "report_source_literature": "Check12",
    "report_type_initial": "Check13",
    "report_type_followup": "Check14",
    "report_source_health_professional": "Check15"
}

ALL_FIELDS = list(FIELD_SCHEMA.keys()) + list(CHECKBOX_SCHEMA.keys())
