"""The canonical VitalHealth demo-patient roster.

Pure data — no app imports, safe to import from any venv (or none at all).
Each character carries one scenario per app, shaped exactly to that app's
real input schema:

  triage  -> apps/Jace/api.py `PredictRequest` (age/sex/arrival_mode/
             complaints/vitals) — the same `fields` dict shape
             `core.predict_from_fields()` expects.
  stroke  -> apps/Jeslyn/app.py `parse_patient_form()` output (the 13
             `feature_cols` keys, already typed).
  emc     -> apps/YS/app.py `collect_metadata()`/`collect_features()`
             shape, except `symptoms` is a short list of the FEATURE_LAYOUT
             columns to set to 1 (compute_emc.py expands it to the full
             132-key 0/1 dict — see collect_features(), which likewise
             defaults every unset column to 0).

`external_id` doubles as the shared-DB `patients.external_id` AND YS's
`patient_id` (both need `[A-Za-z0-9-]{3,32}`), so the same identity is
reusable end to end. All eight are prefixed `demo-` for easy cleanup
(`seed_db.py --reset`).

Ages are chosen so each character is valid for Jeslyn (0-120) and YS
(0-130), while character 5 (Timothy Ng, 15) is deliberately *outside*
Jace's valid 18-102 range to demonstrate its graceful `model_refused`
behavior.
"""

CHARACTERS = [
    {
        "external_id": "demo-nur-aini-yusof",
        "display_name": "Nur Aini binte Yusof",
        "note": "Low acuity / low stroke risk / simple EMC — healthy young adult with a headache.",
        "clinician_note": "19yo woman walked in, headache since this morning",
        "triage": {
            "age": 19,
            "sex": "Female",
            "arrival_mode": "walk_in",
            "complaints": [{"token": "headache", "evidence": None}],
            "vitals": {"hr": 72, "sbp": 112, "dbp": 72, "rr": 16, "o2": 99,
                       "o2_device": None, "temp": 36.8, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 19, "Gender": "Female", "BMI": 21.4,
            "Blood_Pressure_Systolic": 112, "Blood_Pressure_Diastolic": 72,
            "Cholesterol": 160, "Glucose_Level": 85,
            "Smoking": 0, "Alcohol_Intake": 0, "Physical_Activity": 1,
            "Family_History": 0, "Heart_Disease": 0, "Diabetes": 0,
        },
        "emc": {
            "metadata": {
                "patient_name": "Nur Aini binte Yusof", "patient_age": 19,
                "clinic_name": "Ang Mo Kio Polyclinic",
                "clinic_address": "21 Ang Mo Kio Central 2, Singapore 569666",
                "attending_clinician_name": "Dr Alex Lee",
                "clinician_registration_no": "M12345A",
                "authorized_medical_leave_days": 2,
                "diagnosis_disclosure_consent": False,
            },
            "symptoms": ["headache", "fatigue", "mild_fever"],
        },
    },
    {
        "external_id": "demo-david-tan",
        "display_name": "David Tan",
        "note": "Red-flag triage (P1, strokealert) / high stroke risk / complex EMC — deliberately coherent across apps.",
        "clinician_note": "74yo man brought in by ambulance, sudden slurred speech and "
                           "one-sided weakness, family suspects stroke",
        "triage": {
            "age": 74,
            "sex": "Male",
            "arrival_mode": "ambulance",
            "complaints": [{"token": "strokealert", "evidence": None}],
            "vitals": {"hr": 112, "sbp": 188, "dbp": 102, "rr": 24, "o2": 92,
                       "o2_device": None, "temp": 37.4, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 74, "Gender": "Male", "BMI": 29.5,
            "Blood_Pressure_Systolic": 188, "Blood_Pressure_Diastolic": 102,
            "Cholesterol": 265, "Glucose_Level": 150,
            "Smoking": 1, "Alcohol_Intake": 1, "Physical_Activity": 0,
            "Family_History": 1, "Heart_Disease": 1, "Diabetes": 1,
        },
        "emc": {
            "metadata": {
                "patient_name": "David Tan", "patient_age": 74,
                "clinic_name": "Tampines Polyclinic",
                "clinic_address": "1 Tampines Street 41, Singapore 529204",
                "attending_clinician_name": "Dr Priya Nathan",
                "clinician_registration_no": "M45678K",
                "authorized_medical_leave_days": 14,
                "diagnosis_disclosure_consent": True,
            },
            "symptoms": ["slurred_speech", "weakness_of_one_body_side", "dizziness",
                         "headache", "loss_of_balance"],
        },
    },
    {
        "external_id": "demo-grace-lim",
        "display_name": "Grace Lim",
        "note": "Medium acuity (abdominal pain) / medium stroke risk / moderate EMC.",
        "clinician_note": "52yo woman, drove herself in, abdominal pain since last night",
        "triage": {
            "age": 52,
            "sex": "Female",
            "arrival_mode": "car",
            "complaints": [{"token": "abdominalpain", "evidence": None}],
            "vitals": {"hr": 90, "sbp": 142, "dbp": 88, "rr": 18, "o2": 97,
                       "o2_device": None, "temp": 37.6, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 52, "Gender": "Female", "BMI": 27.0,
            "Blood_Pressure_Systolic": 142, "Blood_Pressure_Diastolic": 88,
            "Cholesterol": 215, "Glucose_Level": 130,
            "Smoking": 0, "Alcohol_Intake": 1, "Physical_Activity": 0,
            "Family_History": 1, "Heart_Disease": 0, "Diabetes": 0,
        },
        "emc": {
            "metadata": {
                "patient_name": "Grace Lim", "patient_age": 52,
                "clinic_name": "Bedok Polyclinic",
                "clinic_address": "11 Bedok North Street 1, Singapore 469662",
                "attending_clinician_name": "Dr Alex Lee",
                "clinician_registration_no": "M12345A",
                "authorized_medical_leave_days": 5,
                "diagnosis_disclosure_consent": True,
            },
            "symptoms": ["abdominal_pain", "nausea", "vomiting", "indigestion"],
        },
    },
    {
        "external_id": "demo-marcus-wong",
        "display_name": "Marcus Wong",
        "note": "High acuity (breathing difficulty) / high stroke risk / complex EMC.",
        "clinician_note": "81yo man, ambulance brought him in, severe difficulty breathing, "
                           "started about an hour ago",
        "triage": {
            "age": 81,
            "sex": "Male",
            "arrival_mode": "ambulance",
            "complaints": [{"token": "breathingdifficulty", "evidence": None}],
            "vitals": {"hr": 118, "sbp": 160, "dbp": 95, "rr": 28, "o2": 89,
                       "o2_device": None, "temp": 38.1, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 81, "Gender": "Male", "BMI": 31.0,
            "Blood_Pressure_Systolic": 160, "Blood_Pressure_Diastolic": 95,
            "Cholesterol": 240, "Glucose_Level": 160,
            "Smoking": 1, "Alcohol_Intake": 0, "Physical_Activity": 0,
            "Family_History": 1, "Heart_Disease": 1, "Diabetes": 1,
        },
        "emc": {
            "metadata": {
                "patient_name": "Marcus Wong", "patient_age": 81,
                "clinic_name": "Tampines Polyclinic",
                "clinic_address": "1 Tampines Street 41, Singapore 529204",
                "attending_clinician_name": "Dr Priya Nathan",
                "clinician_registration_no": "M45678K",
                "authorized_medical_leave_days": 10,
                "diagnosis_disclosure_consent": True,
            },
            "symptoms": ["breathlessness", "chest_pain", "fatigue", "cough", "fast_heart_rate"],
        },
    },
    {
        "external_id": "demo-timothy-ng",
        "display_name": "Timothy Ng",
        "note": "Edge case: age 15 is outside Jace's valid 18-102 range (expect model_refused), "
                "while Jeslyn (0-120) and YS (0-130) still process the same character normally.",
        "clinician_note": "15yo boy walked in with his mother, fell while skateboarding, hurt his wrist",
        "triage": {
            "age": 15,
            "sex": "Male",
            "arrival_mode": "walk_in",
            "complaints": [{"token": "fall", "evidence": None}],
            "vitals": {"hr": 88, "sbp": 108, "dbp": 70, "rr": 18, "o2": 98,
                       "o2_device": None, "temp": 37.0, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 15, "Gender": "Male", "BMI": 19.0,
            "Blood_Pressure_Systolic": 108, "Blood_Pressure_Diastolic": 70,
            "Cholesterol": 150, "Glucose_Level": 80,
            "Smoking": 0, "Alcohol_Intake": 0, "Physical_Activity": 1,
            "Family_History": 0, "Heart_Disease": 0, "Diabetes": 0,
        },
        "emc": {
            "metadata": {
                "patient_name": "Timothy Ng", "patient_age": 15,
                "clinic_name": "Bedok Polyclinic",
                "clinic_address": "11 Bedok North Street 1, Singapore 469662",
                "attending_clinician_name": "Dr Alex Lee",
                "clinician_registration_no": "M12345A",
                "authorized_medical_leave_days": 3,
                "diagnosis_disclosure_consent": False,
            },
            "symptoms": ["joint_pain", "muscle_weakness"],
        },
    },
    {
        "external_id": "demo-ethan-koh",
        "display_name": "Ethan Koh",
        "note": "Medium-high acuity (chest pain) / medium stroke risk / moderate EMC.",
        "clinician_note": "45yo man drove himself in, chest pain that started this morning",
        "triage": {
            "age": 45,
            "sex": "Male",
            "arrival_mode": "car",
            "complaints": [{"token": "chestpain", "evidence": None}],
            "vitals": {"hr": 102, "sbp": 150, "dbp": 92, "rr": 20, "o2": 95,
                       "o2_device": None, "temp": 37.0, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 45, "Gender": "Male", "BMI": 26.5,
            "Blood_Pressure_Systolic": 150, "Blood_Pressure_Diastolic": 92,
            "Cholesterol": 230, "Glucose_Level": 110,
            "Smoking": 1, "Alcohol_Intake": 0, "Physical_Activity": 0,
            "Family_History": 0, "Heart_Disease": 0, "Diabetes": 0,
        },
        "emc": {
            "metadata": {
                "patient_name": "Ethan Koh", "patient_age": 45,
                "clinic_name": "Ang Mo Kio Polyclinic",
                "clinic_address": "21 Ang Mo Kio Central 2, Singapore 569666",
                "attending_clinician_name": "Dr Priya Nathan",
                "clinician_registration_no": "M45678K",
                "authorized_medical_leave_days": 7,
                "diagnosis_disclosure_consent": True,
            },
            "symptoms": ["chest_pain", "palpitations", "sweating", "fatigue"],
        },
    },
    {
        "external_id": "demo-rosa-fernandez",
        "display_name": "Rosa Fernandez",
        "note": "Low-medium acuity (fever) / low-medium stroke risk / simple EMC.",
        "clinician_note": "68yo woman took the bus in, fever since yesterday",
        "triage": {
            "age": 68,
            "sex": "Female",
            "arrival_mode": "public_transport",
            "complaints": [{"token": "fever", "evidence": None}],
            "vitals": {"hr": 96, "sbp": 128, "dbp": 80, "rr": 18, "o2": 96,
                       "o2_device": None, "temp": 38.6, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 68, "Gender": "Female", "BMI": 24.0,
            "Blood_Pressure_Systolic": 128, "Blood_Pressure_Diastolic": 80,
            "Cholesterol": 190, "Glucose_Level": 105,
            "Smoking": 0, "Alcohol_Intake": 0, "Physical_Activity": 1,
            "Family_History": 0, "Heart_Disease": 0, "Diabetes": 0,
        },
        "emc": {
            "metadata": {
                "patient_name": "Rosa Fernandez", "patient_age": 68,
                "clinic_name": "Ang Mo Kio Polyclinic",
                "clinic_address": "21 Ang Mo Kio Central 2, Singapore 569666",
                "attending_clinician_name": "Dr Alex Lee",
                "clinician_registration_no": "M12345A",
                "authorized_medical_leave_days": 3,
                "diagnosis_disclosure_consent": False,
            },
            "symptoms": ["high_fever", "chills", "headache", "fatigue"],
        },
    },
    {
        "external_id": "demo-balvinder-singh",
        "display_name": "Balvinder Singh",
        "note": "Medium acuity (dizziness/fall risk) / medium-high stroke risk / moderate EMC.",
        "clinician_note": "58yo man came in a wheelchair, feeling dizzy since this morning",
        "triage": {
            "age": 58,
            "sex": "Male",
            "arrival_mode": "wheelchair",
            "complaints": [{"token": "dizziness", "evidence": None}],
            "vitals": {"hr": 84, "sbp": 136, "dbp": 84, "rr": 16, "o2": 97,
                       "o2_device": None, "temp": 36.9, "temp_unit": "C"},
        },
        "stroke": {
            "Age": 58, "Gender": "Male", "BMI": 28.0,
            "Blood_Pressure_Systolic": 136, "Blood_Pressure_Diastolic": 84,
            "Cholesterol": 205, "Glucose_Level": 120,
            "Smoking": 1, "Alcohol_Intake": 1, "Physical_Activity": 0,
            "Family_History": 0, "Heart_Disease": 0, "Diabetes": 1,
        },
        "emc": {
            "metadata": {
                "patient_name": "Balvinder Singh", "patient_age": 58,
                "clinic_name": "Tampines Polyclinic",
                "clinic_address": "1 Tampines Street 41, Singapore 529204",
                "attending_clinician_name": "Dr Priya Nathan",
                "clinician_registration_no": "M45678K",
                "authorized_medical_leave_days": 4,
                "diagnosis_disclosure_consent": True,
            },
            "symptoms": ["dizziness", "loss_of_balance", "unsteadiness", "fatigue"],
        },
    },
]


def by_external_id():
    return {c["external_id"]: c for c in CHARACTERS}
