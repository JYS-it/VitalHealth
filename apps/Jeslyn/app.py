from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import os
import re
from threading import Lock

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for
from dotenv import load_dotenv
import joblib
import pandas as pd
from pypdf import PdfReader
from vitalhealth_storage import get_store, identity, load_shared_env, missing_shared_keys

try:
    from google import genai
except Exception:
    genai = None


# Load environment variables from .env
load_dotenv()
# SESSION_SECRET and DATABASE_URL are shared with the gateway, not local
# config. This app's own .env ships them blank, and a blank value here means
# every gateway-signed cookie fails verification and every record write is a
# no-op — so fill them from the shared source before anything reads them.
load_shared_env()
for _key in missing_shared_keys():
    print(f"[Jeslyn] WARNING: {_key} is not set - sessions and saved records will not work.")
SHARED_STORE = get_store()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-in-production")


@app.before_request
def require_clinician_role():
    """Reject patient sessions from clinician-only stroke workflows.

    Exception: submit_form/submit/submitted are the patient-facing
    self-submission routes — a patient may reach exactly these, nothing else.

    The gateway always requires authentication. Keeping direct launches
    usable without a cookie preserves the documented local development path.
    """
    # "static" is Flask's built-in static-file endpoint (app.static_url_path),
    # not a route this app defines — without it here, the CSS/JS a patient's
    # allowed pages link to 403s even though the pages themselves load fine.
    if request.endpoint in {"submit_form", "submit", "submitted", "resume_submitted", "submission_status", "static"}:
        return
    if request.cookies.get(identity.COOKIE_NAME):
        actor = identity.actor_from_cookies(request.cookies)
        if actor is None:
            abort(401)
        if not actor.is_clinician:
            abort(403)


def _prefixed_url_for(endpoint: str, **values) -> str:
    """Build URLs that keep the gateway sub-path when proxied."""
    built = url_for(endpoint, **values)
    prefix = (
        request.headers.get("X-Forwarded-Prefix")
        or request.environ.get("HTTP_X_FORWARDED_PREFIX")
        or request.environ.get("SCRIPT_NAME")
        or ""
    ).rstrip("/")
    if not prefix:
        return built
    if built == prefix or built.startswith(prefix + "/"):
        return built
    if built.startswith("/"):
        return f"{prefix}{built}"
    return built


app.jinja_env.globals["url_for"] = _prefixed_url_for


@app.context_processor
def _inject_prefixed_url_for():
    return {"purl": _prefixed_url_for}


class PrefixMiddleware:
    """Makes url_for() emit paths prefixed with X-Forwarded-Prefix when this
    app is proxied behind the gateway under a sub-path (e.g. /stroke). A
    no-op when the header is absent, so standalone runs are unaffected."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        prefix = environ.get("HTTP_X_FORWARDED_PREFIX", "")
        if prefix:
            environ["SCRIPT_NAME"] = prefix
        return self.wsgi_app(environ, start_response)


app.wsgi_app = PrefixMiddleware(app.wsgi_app)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def _resolve_rag_source_folder() -> Path:
    """Find the rag_sources folder using common project paths."""
    candidates = [
        BASE_DIR / "data" / "rag_sources",
        BASE_DIR / "Data" / "rag_sources",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


# Load ML artifacts once when app starts.
model = joblib.load(MODELS_DIR / "stroke_logistic_model.pkl")
scaler = joblib.load(MODELS_DIR / "stroke_scaler.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
num_cols = joblib.load(MODELS_DIR / "num_cols.pkl")
risk_threshold = float(joblib.load(MODELS_DIR / "risk_threshold.pkl"))


RAG_PDF_SOURCES = {
    "stroke_signs": {
        "file_name": "stroke_signs_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Recognising The Signs Of A Stroke",
        "topic": "Stroke signs, F.A.S.T response, stroke risk factors",
    },
    "blood_pressure": {
        "file_name": "high_blood_pressure_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "High Blood Pressure: Healthy Eating Guide",
        "topic": "High blood pressure, hypertension, healthy eating",
    },
    "cholesterol": {
        "file_name": "cholesterol_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Cholesterol and Heart Disease",
        "topic": "Cholesterol and heart disease",
    },
    "physical_activity": {
        "file_name": "physical_activity_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Health Benefits of Exercise and Physical Activity",
        "topic": "Exercise and physical activity",
    },
    "diabetes_eating": {
        "file_name": "diabetes_eating_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Diabetes Healthy Eating Guidance",
        "topic": "Diabetes, glucose, low GI food, healthier eating",
    },
    "stroke_prevention": {
        "file_name": "stroke_prevention_rag.pdf",
        "source_name": "HealthHub Singapore",
        "source_title": "Stroke Prevention: Medications and Lifestyle Changes to Help Reduce Stroke Risk",
        "topic": "Stroke prevention and lifestyle changes",
    },
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from all pages in one PDF."""
    reader = PdfReader(str(pdf_path))
    extracted_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            extracted_text += f"\n--- Page {page_number} ---\n"
            extracted_text += page_text

    return extracted_text


def load_rag_documents() -> list:
    """Load all configured RAG PDFs into memory."""
    rag_documents = []
    rag_folder = _resolve_rag_source_folder()

    for source_key, source_info in RAG_PDF_SOURCES.items():
        pdf_path = rag_folder / source_info["file_name"]

        if not pdf_path.exists():
            continue

        pdf_text = extract_text_from_pdf(pdf_path)

        rag_documents.append(
            {
                "source_key": source_key,
                "source_name": source_info["source_name"],
                "source_title": source_info["source_title"],
                "topic": source_info["topic"],
                "source_file": source_info["file_name"],
                "text": pdf_text,
            }
        )

    return rag_documents


# PDF text extraction is expensive enough to delay server startup, especially
# from a synced directory. Load it on the first care-plan request instead.
_rag_documents = None
_rag_documents_lock = Lock()


def get_rag_documents() -> list:
    global _rag_documents
    if _rag_documents is None:
        with _rag_documents_lock:
            if _rag_documents is None:
                _rag_documents = load_rag_documents()
    return _rag_documents


def retrieve_relevant_pdf_guidance(patient_data: dict) -> list:
    """Select relevant PDF guidance based on patient risk factors."""
    selected_source_keys = ["stroke_signs", "stroke_prevention"]

    if (
        patient_data["Blood_Pressure_Systolic"] >= 140
        or patient_data["Blood_Pressure_Diastolic"] >= 90
    ):
        selected_source_keys.append("blood_pressure")

    if patient_data["Cholesterol"] >= 200:
        selected_source_keys.append("cholesterol")

    if patient_data["Physical_Activity"] == 0:
        selected_source_keys.append("physical_activity")

    if patient_data["Glucose_Level"] >= 140 or patient_data["Diabetes"] == 1:
        selected_source_keys.append("diabetes_eating")

    if patient_data["BMI"] >= 25 and "diabetes_eating" not in selected_source_keys:
        selected_source_keys.append("diabetes_eating")

    selected_source_keys = list(dict.fromkeys(selected_source_keys))

    retrieved_guidance = []

    for document in get_rag_documents():
        if document["source_key"] in selected_source_keys:
            retrieved_guidance.append(
                {
                    "source_key": document["source_key"],
                    "source_name": document["source_name"],
                    "source_title": document["source_title"],
                    "source_file": document["source_file"],
                    "topic": document["topic"],
                    "retrieved_text": document["text"][:2500],
                }
            )

    return retrieved_guidance


def clean_generated_care_plan(text: str) -> str:
    """
    Clean Gemini output so markdown symbols do not appear on the website.
    """

    if not text:
        return ""

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        # Remove markdown headings such as ### Title
        line = re.sub(r"^#{1,6}\s*", "", line)

        # Remove markdown bold, italic, and code formatting
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = line.replace("`", "")

        # Remove horizontal rules like ***, ---, or ___
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Convert markdown bullet stars to simple dash bullets
        line = re.sub(r"^\*\s+", "- ", line)

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Remove too many blank lines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def build_rag_prompt(patient_data: dict, prediction_result: dict, retrieved_guidance: list) -> str:
    """Build Gemini prompt with patient profile, prediction output, and retrieved context."""
    retrieved_context_sections = []

    for item in retrieved_guidance:
        retrieved_context_sections.append(
            f"""
Source Title: {item['source_title']}
Source Name: {item['source_name']}
Source File: {item['source_file']}
Topic: {item['topic']}

Retrieved Text:
{item['retrieved_text']}
"""
        )

    prompt = f"""
You are a healthcare education assistant for a student AI project.

Your task:
Generate a personalized stroke risk care plan using the machine learning result and retrieved healthcare source context below.

Important safety rules:
- Do not diagnose stroke.
- Do not prescribe medication or dosage.
- Advise consulting a healthcare professional.
- Mention emergency stroke warning signs clearly.
- End with an educational disclaimer.
- Use simple patient-friendly language.
- Use plain text only.
- Do not use markdown symbols such as #, *, **, ---, or backticks.
- Use clear numbered section titles.
- Use simple dash bullet points only.

Binary values:
- 1 means Yes
- 0 means No

Machine Learning Model Output:
- Model used: Logistic Regression
- Predicted risk category: {prediction_result['risk_category']}
- Model-estimated risk probability: {prediction_result['risk_probability_percent']}%

Patient Health Profile:
- Age: {patient_data['Age']}
- Gender: {patient_data['Gender']}
- BMI: {patient_data['BMI']}
- Systolic Blood Pressure: {patient_data['Blood_Pressure_Systolic']}
- Diastolic Blood Pressure: {patient_data['Blood_Pressure_Diastolic']}
- Cholesterol: {patient_data['Cholesterol']}
- Glucose Level: {patient_data['Glucose_Level']}
- Smoking: {patient_data['Smoking']}
- Alcohol Intake: {patient_data['Alcohol_Intake']}
- Physical Activity: {patient_data['Physical_Activity']}
- Family History of Stroke: {patient_data['Family_History']}
- Heart Disease: {patient_data['Heart_Disease']}
- Diabetes: {patient_data['Diabetes']}

Retrieved PDF Source Context:
{chr(10).join(retrieved_context_sections)}

Generate the care plan with this exact structure:

1. Risk Summary
2. Sources Retrieved
3. Key Health Concerns
4. Recommended Daily Goals
5. Lifestyle Guidance
6. Monitoring Advice
7. When to Seek Medical Help
8. Disclaimer

Emergency warning signs to mention:
face drooping, arm weakness, speech difficulty, sudden confusion, sudden vision problems, sudden severe headache.
"""

    return prompt


def generate_care_plan_with_gemini(prompt: str) -> str:
    """Generate care plan using Gemini model and API key from .env."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return (
            "Gemini API key is not configured. Please add GEMINI_API_KEY in .env.\n\n"
            "Educational disclaimer: This output is for educational support only and does not replace advice from a qualified healthcare professional."
        )

    if genai is None:
        return (
            "google-generativeai package is not available. Install dependencies from requirements.txt.\n\n"
            "Educational disclaimer: This output is for educational support only and does not replace advice from a qualified healthcare professional."
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return clean_generated_care_plan(response.text)

    except Exception as exc:
        return (
            "Care plan generation is currently unavailable.\n"
            f"Technical note: {exc}\n\n"
            "Educational disclaimer: This output is for educational support only and does not replace advice from a qualified healthcare professional."
        )


def build_daily_goals(patient_data: dict) -> list:
    """Create rule-based daily goals from patient risk factors."""
    goals = []

    if (
        patient_data["Blood_Pressure_Systolic"] >= 140
        or patient_data["Blood_Pressure_Diastolic"] >= 90
    ):
        goals.append("Check blood pressure and choose lower-salt meals.")

    if patient_data["Glucose_Level"] >= 140 or patient_data["Diabetes"] == 1:
        goals.append("Follow glucose-friendly meals and reduce high-sugar foods.")

    if patient_data["Cholesterol"] >= 200:
        goals.append("Choose heart-healthier meals with less fried or fatty food.")

    if patient_data["Smoking"] == 1:
        goals.append("Reduce smoking triggers and seek support to quit.")

    if patient_data["Alcohol_Intake"] == 1:
        goals.append("Limit alcohol intake and increase water intake.")

    if patient_data["Physical_Activity"] == 0:
        goals.append("Do light walking or gentle movement for 10 to 20 minutes.")

    if patient_data["BMI"] >= 25:
        goals.append("Practice portion control and include regular gentle movement.")

    if patient_data["Heart_Disease"] == 1 or patient_data["Family_History"] == 1:
        goals.append("Plan or maintain regular medical follow-up.")

    if not goals:
        goals.append("Maintain healthy habits and continue regular monitoring.")

    return goals


def build_7_day_care_calendar(patient_data: dict) -> list:
    """Build a simple 7-day care calendar table for display."""
    goals = build_daily_goals(patient_data)
    today = datetime.today()
    calendar_rows = []

    for i in range(7):
        current_date = today + timedelta(days=i)
        main_goal = goals[i % len(goals)]

        calendar_rows.append(
            {
                "Date": current_date.strftime("%Y-%m-%d"),
                "Day": current_date.strftime("%A"),
                "Main Focus": main_goal,
                "Morning Action": "Review today's health goal.",
                "Afternoon Action": main_goal,
                "Evening Action": "Reflect on progress and prepare for tomorrow.",
            }
        )

    return calendar_rows


def parse_patient_form(form_data) -> dict:
    """Convert submitted form values into a typed patient_data dictionary."""
    return {
        "Age": int(form_data["Age"]),
        "Gender": form_data["Gender"],
        "BMI": float(form_data["BMI"]),
        "Blood_Pressure_Systolic": float(form_data["Blood_Pressure_Systolic"]),
        "Blood_Pressure_Diastolic": float(form_data["Blood_Pressure_Diastolic"]),
        "Cholesterol": float(form_data["Cholesterol"]),
        "Glucose_Level": float(form_data["Glucose_Level"]),
        "Smoking": int(form_data["Smoking"]),
        "Alcohol_Intake": int(form_data["Alcohol_Intake"]),
        "Physical_Activity": int(form_data["Physical_Activity"]),
        "Family_History": int(form_data["Family_History"]),
        "Heart_Disease": int(form_data["Heart_Disease"]),
        "Diabetes": int(form_data["Diabetes"]),
    }


def predict_stroke_risk(patient_data: dict) -> dict:
    """The model call only — sub-millisecond, no network. Shared by the
    clinician-instant /prediction route and the patient /submit route."""
    model_input_df = pd.DataFrame([patient_data])[feature_cols].copy()
    model_input_df["Gender"] = model_input_df["Gender"].map({"Male": 1, "Female": 0})
    model_input_df[num_cols] = scaler.transform(model_input_df[num_cols])
    model_input_df = model_input_df.apply(pd.to_numeric, errors="coerce")

    risk_probability = float(model.predict_proba(model_input_df)[:, 1][0])
    risk_category = "High risk" if risk_probability >= risk_threshold else "Low risk"

    return {
        "risk_category": risk_category,
        "risk_probability_percent": round(risk_probability * 100, 2),
        "threshold_used": float(risk_threshold),
    }


def generate_stroke_care_plan(patient_data: dict, prediction_result: dict) -> tuple[str, list, list]:
    """RAG retrieval + the one real LLM call + the 7-day calendar. Shared by
    the clinician-instant /care-plan route and the patient /submit route,
    which generates this eagerly so a reviewing clinician has a complete
    draft rather than an extra "generate now" step."""
    retrieved_guidance = retrieve_relevant_pdf_guidance(patient_data)
    retrieved_source_titles = [item["source_title"] for item in retrieved_guidance]
    rag_prompt = build_rag_prompt(patient_data, prediction_result, retrieved_guidance)
    generated_care_plan = generate_care_plan_with_gemini(rag_prompt)
    care_calendar = build_7_day_care_calendar(patient_data)
    return generated_care_plan, retrieved_source_titles, care_calendar


def patient_data_fingerprint(patient_data) -> str:
    """Identifies one particular set of answers, so a second assessment in the
    same browser session is recognised as new rather than as an edit."""
    return hashlib.sha256(
        json.dumps(patient_data, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def persist_stroke_record(patient_data, prediction_result, care_plan=None, care_calendar=None):
    """Store the assessment independently of the browser session when enabled."""
    output = {"prediction": prediction_result}
    if care_plan is not None:
        output["care_plan"] = care_plan
    if care_calendar is not None:
        output["care_calendar"] = care_calendar

    # Who is logged in, and whose assessment this is. Read from the signed
    # session cookie the gateway forwards rather than any X-Vitalhealth-*
    # header, which anything on this host could set.
    actor = identity.actor_from_cookies(request.cookies)
    subject_ref, subject_name = identity.resolve_subject(request.cookies, actor)

    fingerprint = patient_data_fingerprint(patient_data)
    stored = session.get("database_record") or {}

    # Only continue an existing record when it is the *same* assessment being
    # enriched with a care plan. Keying on the session alone meant a second
    # assessment silently overwrote the first, which is invisible today but
    # destroys history the moment a dashboard lists it.
    if stored.get("id") and stored.get("fingerprint") == fingerprint:
        record_id = stored["id"]
        SHARED_STORE.safe_update_record(
            record_id,
            status="CARE_PLAN_READY" if care_plan else "ASSESSED",
            output_payload=output,
            owner_user_id=actor.user_id if actor else None,
        )
    else:
        record_id = SHARED_STORE.safe_create_record(
            source_app="stroke",
            record_type="stroke_risk_assessment",
            status="CARE_PLAN_READY" if care_plan else "ASSESSED",
            input_payload=patient_data,
            output_payload=output,
            model_version="stroke_logistic_model",
            patient_external_id=subject_ref,
            patient_name=subject_name,
            owner_user_id=actor.user_id if actor else None,
        )
        if record_id:
            session["database_record"] = {"id": record_id, "fingerprint": fingerprint}

    SHARED_STORE.safe_append_audit_event(
        source_app="stroke",
        event_type="stroke_care_plan_generated" if care_plan else "stroke_assessed",
        record_id=record_id,
        actor_reference=actor.email if actor else None,
        payload={"risk_category": prediction_result["risk_category"]},
    )
    return record_id


def persist_stroke_record_for_review(patient_data, prediction_result, care_plan, care_calendar):
    """Patient self-submission, always a new record. Deliberately does not
    touch Flask session at all (unlike persist_stroke_record) — that
    mechanism links /prediction to /care-plan within one browser session,
    which doesn't apply here: this route computes both eagerly in one
    request, and review happens in a completely different browser/session
    later."""
    output = {"prediction": prediction_result, "care_plan": care_plan, "care_calendar": care_calendar}
    actor = identity.actor_from_cookies(request.cookies)
    subject_ref, subject_name = identity.resolve_subject(request.cookies, actor)

    # Care-plan generation can take long enough for a patient to double-click
    # Submit. Do not create two open clinician tasks for identical answers;
    # resume the already-pending request instead.
    if actor and SHARED_STORE.enabled:
        try:
            for record in SHARED_STORE.list_records(
                owner_user_id=actor.user_id,
                source_app="stroke",
                status="PENDING_REVIEW",
                limit=50,
            ):
                if record.get("input_payload") == patient_data:
                    return record["id"]
        except Exception:
            # Keep the existing availability behaviour if a database read is
            # temporarily unavailable; safe_create_record handles its own
            # write failures below.
            pass

    record_id = SHARED_STORE.safe_create_record(
        source_app="stroke", record_type="stroke_risk_assessment", status="PENDING_REVIEW",
        input_payload=patient_data, output_payload=output, model_version="stroke_logistic_model",
        patient_external_id=subject_ref, patient_name=subject_name,
        owner_user_id=actor.user_id if actor else None,
    )
    SHARED_STORE.safe_append_audit_event(
        source_app="stroke", event_type="stroke_submitted_for_review", record_id=record_id,
        actor_reference=actor.email if actor else None,
        payload={"risk_category": prediction_result["risk_category"]},
    )
    return record_id


@app.get("/submit")
def submit_form():
    actor = identity.actor_from_cookies(request.cookies)
    if actor is None or not actor.is_patient:
        abort(403)
    return render_template("submit.html", page_title="Request Stroke Risk Assessment", error_message=None)


@app.post("/submit")
def submit():
    """Patient self-submission. Never renders a result back to the caller —
    the response is only a generic "submitted" confirmation (see
    /submitted/<record_id>)."""
    actor = identity.actor_from_cookies(request.cookies)
    if actor is None or not actor.is_patient:
        abort(403)
    try:
        patient_data = parse_patient_form(request.form)
        prediction_result = predict_stroke_risk(patient_data)
    except Exception as exc:
        return render_template(
            "submit.html", page_title="Request Stroke Risk Assessment",
            error_message=f"Could not process submission: {exc}",
        )
    generated_care_plan, _source_titles, care_calendar = generate_stroke_care_plan(patient_data, prediction_result)
    record_id = persist_stroke_record_for_review(patient_data, prediction_result, generated_care_plan, care_calendar)
    return redirect(_prefixed_url_for("submitted", record_id=record_id))


@app.get("/submitted/<record_id>")
def submitted(record_id):
    """The patient's waiting page. Renders no result itself — it polls
    /status/<record_id> and reveals the outcome in place once a clinician has
    approved it, so the patient can simply wait here instead of being sent
    away to the dashboard."""
    # The waiting page is intentionally generic, but it is still a patient
    # workflow resource.  Require the same ownership check as its polling
    # endpoint so a copied or guessed id cannot be used to open it.
    actor = identity.actor_from_cookies(request.cookies)
    record = SHARED_STORE.get_record(record_id) if SHARED_STORE.enabled else None
    if (
        actor is None
        or not actor.is_patient
        or record is None
        or record.get("source_app") != "stroke"
        or not _patient_owns_record(actor, record)
    ):
        abort(404)
    return render_template("submitted.html", page_title="Submitted", record_id=record_id)


def _patient_owns_record(actor, record):
    """A patient may only ever poll their own submission."""
    if actor is None or record is None:
        return False
    if record.get("owner_user_id") and record["owner_user_id"] == actor.user_id:
        return True
    if not record.get("patient_id") or not actor.patient_external_id:
        return False
    patient = SHARED_STORE.get_patient(record["patient_id"])
    return bool(patient and patient["external_id"] == actor.patient_external_id)


@app.get("/submit/<record_id>")
def resume_submitted(record_id):
    """Support the common copied `/submit/<id>` URL without exposing data."""
    actor = identity.actor_from_cookies(request.cookies)
    record = SHARED_STORE.get_record(record_id) if SHARED_STORE.enabled else None
    if (
        actor is None
        or not actor.is_patient
        or record is None
        or record.get("source_app") != "stroke"
        or not _patient_owns_record(actor, record)
    ):
        abort(404)
    return redirect(_prefixed_url_for("submitted", record_id=record_id))


@app.get("/status/<record_id>")
def submission_status(record_id):
    """Patient-facing poll target for the waiting page.

    Returns the clinician-approved result and nothing else. While a record is
    pending there is deliberately no risk category, probability, or care-plan
    text in the response at all — not hidden in the payload for the frontend
    to filter, simply absent — so a patient watching the network tab still
    cannot read a result their clinician has not released.
    """
    actor = identity.actor_from_cookies(request.cookies)
    if actor is None:
        return jsonify({"error": "authentication_required"}), 401

    record = SHARED_STORE.get_record(record_id) if SHARED_STORE.enabled else None
    if record is None or record.get("source_app") != "stroke":
        abort(404)

    # 404 rather than 403: a patient probing ids should not be able to learn
    # which ones exist. Clinicians may read any record.
    if not actor.is_clinician and not _patient_owns_record(actor, record):
        abort(404)

    status = str(record.get("status") or "").upper()
    payload = {"status": status, "state": "pending", "ready": False, "result": None}

    if status == "REJECTED":
        payload["state"] = "rejected"
        return jsonify(payload)

    if status not in ("APPROVED", "ASSESSED", "CARE_PLAN_READY"):
        return jsonify(payload)

    output = record.get("output_payload") or {}
    prediction = output.get("prediction") or {}
    payload.update({
        "state": "approved",
        "ready": True,
        "result": {
            "risk_category": prediction.get("risk_category"),
            "risk_probability_percent": prediction.get("risk_probability_percent"),
            "care_plan": output.get("care_plan") or "",
            "care_calendar": output.get("care_calendar") or [],
        },
    })
    return jsonify(payload)


@app.get("/review/<record_id>")
def review_form(record_id):
    record = SHARED_STORE.get_record(record_id)
    if record is None or record.get("source_app") != "stroke":
        abort(404)
    output = record.get("output_payload") or {}
    return render_template(
        "review.html", page_title="Review stroke assessment", record=record,
        patient_data=record.get("input_payload") or {},
        prediction=output.get("prediction") or {},
        care_plan_text=output.get("care_plan", ""),
        care_calendar=output.get("care_calendar") or [],
        message=None,
    )


@app.post("/review/<record_id>")
def review_submit(record_id):
    record = SHARED_STORE.get_record(record_id)
    if record is None or record.get("source_app") != "stroke":
        abort(404)
    actor = identity.actor_from_cookies(request.cookies)
    action = request.form.get("action")

    output = dict(record.get("output_payload") or {})
    prediction = dict(output.get("prediction") or {})

    if action in ("save", "approve"):
        risk_category = request.form.get("risk_category", "").strip()
        if risk_category:
            prediction["risk_category"] = risk_category
        try:
            prediction["risk_probability_percent"] = float(request.form.get("risk_probability_percent"))
        except (TypeError, ValueError):
            pass
        output["prediction"] = prediction

        edited_plan = request.form.get("care_plan_text", "").strip()
        if edited_plan:
            output["care_plan"] = edited_plan

    if action == "save":
        if not SHARED_STORE.safe_update_record_if_status(
            record_id, expected_status="PENDING_REVIEW", status="PENDING_REVIEW", output_payload=output
        ):
            abort(409, description="This assessment was already reviewed by another clinician. Refresh the page.")
        SHARED_STORE.safe_append_audit_event(
            source_app="stroke", event_type="stroke_reviewer_edited", record_id=record_id,
            actor_reference=actor.email if actor else None,
        )
        return redirect(_prefixed_url_for("review_form", record_id=record_id))

    if action == "approve":
        if not SHARED_STORE.safe_update_record_if_status(
            record_id, expected_status="PENDING_REVIEW", status="APPROVED", output_payload=output
        ):
            abort(409, description="This assessment was already reviewed by another clinician. Refresh the page.")
        SHARED_STORE.safe_append_audit_event(
            source_app="stroke", event_type="stroke_approved", record_id=record_id,
            actor_reference=actor.email if actor else None,
        )
        return redirect(_prefixed_url_for("review_form", record_id=record_id))

    if action == "reject":
        if not SHARED_STORE.safe_update_record_if_status(
            record_id, expected_status="PENDING_REVIEW", status="REJECTED", output_payload=output
        ):
            abort(409, description="This assessment was already reviewed by another clinician. Refresh the page.")
        SHARED_STORE.safe_append_audit_event(
            source_app="stroke", event_type="stroke_rejected", record_id=record_id,
            actor_reference=actor.email if actor else None,
        )
        return redirect(_prefixed_url_for("review_form", record_id=record_id))

    abort(400)


@app.route("/")
def home():
    return redirect(url_for("prediction"))


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    prediction_result = None
    patient_data = session.get("patient_data")
    error_message = None

    if request.method == "POST":
        try:
            patient_data = parse_patient_form(request.form)
            prediction_result = predict_stroke_risk(patient_data)

            # Store in session for care-plan route.
            session["patient_data"] = patient_data
            session["prediction_result"] = prediction_result
            persist_stroke_record(patient_data, prediction_result)

        except Exception as exc:
            error_message = f"Could not process prediction: {exc}"

    # On GET, show latest session result if available.
    if prediction_result is None:
        prediction_result = session.get("prediction_result")

    return render_template(
        "prediction.html",
        page_title="Prediction",
        prediction_result=prediction_result,
        patient_data=patient_data,
        error_message=error_message,
    )


@app.route("/care-plan")
def care_plan():
    patient_data = session.get("patient_data")
    prediction_result = session.get("prediction_result")

    if not patient_data or not prediction_result:
        return redirect(url_for("prediction"))

    generated_care_plan, retrieved_source_titles, care_calendar = generate_stroke_care_plan(patient_data, prediction_result)
    persist_stroke_record(patient_data, prediction_result, generated_care_plan, care_calendar)

    return render_template(
        "care_plan.html",
        page_title="Care Plan",
        prediction_result=prediction_result,
        patient_data=patient_data,
        retrieved_source_titles=retrieved_source_titles,
        generated_care_plan=generated_care_plan,
        care_calendar=care_calendar,
    )


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes"}
    app.run(debug=debug_mode, use_reloader=debug_mode)
