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
        # Triage showcase: routine basis. Complete note, normal vitals, nothing escalating —
        # the justification has to say plainly that no escalating factor was recorded.
        "clinician_note": "19yo woman walked in on her own. Headache since this morning, "
                          "pain 4/10, no vomiting and no visual changes. Not on any regular "
                          "medication. NKDA.",
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
        # Triage showcase: red_flag basis — the level is floored to P1 by rule, not by the
        # model, which is the justification's hardest thing to say honestly. Richest Background
        # of the roster (onset, three history mentions, two medications, a stated allergy), so
        # it also exercises the handover's Pathway line.
        "clinician_note": "74yo man brought in by ambulance. Sudden slurred speech and "
                          "one-sided weakness starting 40 minutes ago, family suspects "
                          "stroke. Hx hypertension, atrial fibrillation, previous TIA two "
                          "years ago. On warfarin and amlodipine. Allergic to penicillin.",
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
        # Triage showcase: complaint basis with a stated pain score and a prior diagnosis the
        # note quotes but the model never codes — the handover Background carries it verbatim
        # while the justification must stay silent about it.
        "clinician_note": "52yo woman drove herself in. Abdominal pain since last night, "
                          "worse over the right side, pain 7/10. Vomited twice this morning. "
                          "Hx gallstones. Takes omeprazole. NKDA.",
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
        # Triage showcase: physiology basis — RR 28, SpO2 89 and T 38.1 are all out of range,
        # the one character where the vitals genuinely drive the level. The handover's Monitor
        # line has something real to attach a trigger to.
        "clinician_note": "81yo man, ambulance brought him in. Severe difficulty breathing "
                          "that started about an hour ago, worse lying flat. Hx COPD and "
                          "heart failure, admitted twice last year. On salbutamol inhaler "
                          "and furosemide. No known drug allergies.",
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
        # Triage showcase: age refusal. The note is deliberately GOOD — full onset, pain score,
        # negatives, allergy status — so the refusal is visibly about the 18-102 range and not
        # about a thin note.
        "clinician_note": "15yo boy walked in with his mother. Fell while skateboarding about "
                          "an hour ago and hurt his right wrist, pain 6/10. No head injury and "
                          "did not black out. NKDA, no regular medication.",
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
        # Triage showcase: the highest-base-rate complaint in the roster, so the justification
        # has a real historical frequency to lead with, and the handover has an obvious
        # protocol-standard Obtain line (ECG / troponin).
        "clinician_note": "45yo man drove himself in. Central chest pain since this morning, "
                          "pain 6/10, radiating to the left arm, feels clammy. Hx high "
                          "cholesterol, father had a heart attack in his fifties. On "
                          "atorvastatin. NKDA.",
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
        "note": "Low-medium acuity (fever) / low-medium stroke risk / simple EMC. Deliberately "
                "the THIN note of the roster — see clinician_note.",
        # Triage showcase: incompleteness. Kept sparse on purpose and given a reason for it, so
        # the handover's Information gaps line and Complete: step have something to report and
        # the justification has to state its uncertainty rather than pad.
        "clinician_note": "68yo woman took the bus in. Fever since yesterday. Came alone, no "
                          "family with her, unable to give any further history.",
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
        # Triage showcase: the only wheelchair arrival, plus a partially-unknown medication the
        # note quotes honestly rather than guessing — span-or-silence made visible in the
        # handover Background.
        "clinician_note": "58yo man brought in by his daughter in a wheelchair. Dizzy since "
                          "this morning, worse on standing. Hx type 2 diabetes and "
                          "hypertension. On metformin and a blood pressure tablet he cannot "
                          "name. Allergic to sulfa drugs.",
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


# --- Demo logins -------------------------------------------------------------
#
# seed_db.py creates one patient account per character plus the clinician below,
# so both dashboards have content the moment you log in. Emails are derived from
# external_id rather than stored per character: there is exactly one login per
# patient, and deriving it keeps the roster above as the single source of truth.
#
# .local is reserved by RFC 6762 and cannot resolve on the public internet, so
# these addresses can never accidentally reach a real mailbox.

DEMO_PASSWORD = "demo1234"  # 8 chars: the gateway's registration minimum
DEMO_EMAIL_DOMAIN = "demo.vitalhealth.local"

DEMO_CLINICIAN = {
    # Matches the attending_clinician_name already used across the EMC scenarios.
    "display_name": "Dr Alex Lee",
    "email": f"dr.alex.lee@{DEMO_EMAIL_DOMAIN}",
    "password": DEMO_PASSWORD,
}


def demo_email(external_id: str) -> str:
    """demo-grace-lim -> grace.lim@demo.vitalhealth.local"""
    local_part = external_id.removeprefix("demo-").replace("-", ".")
    return f"{local_part}@{DEMO_EMAIL_DOMAIN}"


def demo_logins():
    """(email, password, display_name, external_id) for every demo patient."""
    return [
        (demo_email(c["external_id"]), DEMO_PASSWORD, c["display_name"], c["external_id"])
        for c in CHARACTERS
    ]
