import hashlib
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, render_template, request
from markupsafe import Markup, escape
from openai import OpenAI
from dotenv import load_dotenv
from vitalhealth_storage import get_store, identity


load_dotenv()
SHARED_STORE = get_store()
APP_TITLE = "IIP-EMC Clinical Copilot"
PROMPT_VERSION = "webapp_genai_emc_v4.0-grounded"
GENAI_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
MODEL_PATH = "iip_emc_production_model.pkl"
PREPROCESS_PATH = "iip_emc_preprocessed_data.pkl"
AUDIT_PATH = "latest_emc_audit_trail.pkl"

POLICY_CORPUS = [
    {"id": "EMC-001", "title": "Required EMC fields", "text": "A draft EMC includes patient identity, consultation date, authorised leave start and end dates, leave duration, clinic name and address, clinician name, registration number, and pending-review status."},
    {"id": "EMC-002", "title": "Diagnosis disclosure", "text": "Do not include diagnosis or medical details in patient-facing EMC text unless diagnosis disclosure consent has been recorded. Do not infer or reveal diagnosis from symptoms or model output."},
    {"id": "EMC-003", "title": "Clinical boundaries", "text": "Use only clinician-confirmed administrative facts. Do not invent symptoms, examination findings, investigations, prescriptions, treatment advice, or leave duration."},
    {"id": "EMC-004", "title": "Approval and issue", "text": "Only an identified clinician may approve an EMC. A final EMC requires certificate identifier, approver, registration number, approval date, and approved-for-issue status."},
    {"id": "EMC-005", "title": "Patient-facing privacy", "text": "Patient-facing text must not expose ML scores, differential alternatives, classifier details, internal audit records, or prompt text."},
]
STYLE_OPTIONS = {"standard": "Standard EMC", "concise": "Concise EMC"}
STYLE_INSTRUCTIONS = {
    "standard": "Use concise, formal clinical-administrative language.",
    "concise": "Use a compact certificate layout while retaining all required fields.",
}
FORBIDDEN_TERMS = ["model confidence", "confidence percentage", "differential", "classifier", "algorithm", "audit", "prompt version", "machine learning"]
DERIVED_FEATURES = {"Age", "Gender", "Duration", "Medical_History"}

app = Flask(__name__)
WORKFLOWS = {}


@app.before_request
def require_clinician_role():
    """Reject patient sessions from EMC review, approval, and issuance.

    The gateway always requires authentication. Keeping direct launches
    usable without a cookie preserves the documented local development path.
    """
    if request.cookies.get(identity.COOKIE_NAME):
        actor = identity.actor_from_cookies(request.cookies)
        if actor is None:
            abort(401)
        if not actor.is_clinician:
            abort(403)


def _prefixed_url_for(endpoint: str, **values) -> str:
    """Build URLs that keep the gateway sub-path when proxied."""
    from flask import url_for

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
    return {"url_for": _prefixed_url_for}


class PrefixMiddleware:
    """Makes url_for() emit paths prefixed with X-Forwarded-Prefix when this
    app is proxied behind the gateway under a sub-path (e.g. /emc). A no-op
    when the header is absent, so standalone runs are unaffected."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        prefix = environ.get("HTTP_X_FORWARDED_PREFIX", "")
        if prefix:
            environ["SCRIPT_NAME"] = prefix
        return self.wsgi_app(environ, start_response)


app.wsgi_app = PrefixMiddleware(app.wsgi_app)


def load_assets():
    model_payload = joblib.load(MODEL_PATH)
    preprocess_payload = joblib.load(PREPROCESS_PATH)
    return (
        model_payload["model"],
        list(model_payload["features_layout"]),
        model_payload.get("scaler", preprocess_payload.get("scaler")),
        model_payload.get("model_metadata", {}),
    )


MODEL, FEATURE_LAYOUT, SCALER, MODEL_METADATA = load_assets()


def label_for(feature):
    return feature.replace("_", " ").replace("diarrhoea", "diarrhea").title()


def symptom_fields():
    return [(feature, label_for(feature)) for feature in FEATURE_LAYOUT if feature not in DERIVED_FEATURES]


def render_certificate(text):
    """Render limited certificate structure while escaping all model-generated text."""
    rendered = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            rendered.append('<div class="certificate-gap"></div>')
            continue
        bold_heading = re.fullmatch(r"\*\*(.+?)\*\*", line)
        plain_heading = re.fullmatch(r"(?:\d+\.\s*)?[A-Z][A-Z0-9 /&()_-]{3,}", line)
        if bold_heading:
            rendered.append(f"<h4>{escape(bold_heading.group(1))}</h4>")
        elif plain_heading:
            heading_text = re.sub(r"^\d+\.\s*", "", line)
            rendered.append(f"<h4>{escape(heading_text)}</h4>")
        elif line.startswith("- "):
            rendered.append(f"<p class=\"certificate-item\">{escape(line[2:])}</p>")
        else:
            rendered.append(f"<p>{escape(line)}</p>")
    return Markup("\n".join(rendered))


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_api_key():
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def call_live_text(system, prompt, api_key, temperature=0.1):
    if not api_key:
        return {"status": "OFFLINE", "text": None, "error": "No OpenRouter API key was provided."}
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        response = client.chat.completions.create(
            model=GENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return {"status": "LIVE", "text": response.choices[0].message.content.strip(), "error": None}
    except Exception as exc:
        return {"status": "ERROR", "text": None, "error": repr(exc)}


def call_live_json(system, prompt, api_key):
    result = call_live_text(system + " Return valid JSON only, without Markdown fences.", prompt, api_key, temperature=0)
    if result["status"] != "LIVE":
        return result | {"json": None}
    try:
        return result | {"json": json.loads(result["text"])}
    except json.JSONDecodeError as exc:
        return {"status": "ERROR", "text": result["text"], "json": None, "error": f"Invalid JSON: {exc}"}


def normalize_text(value, field_name, maximum, required=True):
    cleaned = " ".join((value or "").split())
    if required and not cleaned:
        raise ValueError(f"{field_name} is required.")
    if len(cleaned) > maximum:
        raise ValueError(f"{field_name} must be {maximum} characters or fewer.")
    return cleaned


def parse_iso_date(value, field_name):
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD.") from exc


def current_actor():
    """The signed-in user, from the session cookie the gateway forwards."""
    return identity.actor_from_cookies(request.cookies)


def collect_metadata(form, actor=None):
    today = date.today().isoformat()
    patient_id = normalize_text(form.get("patient_id"), "Patient ID", 32)
    patient_name = normalize_text(form.get("patient_name"), "Patient full name", 120)

    # A signed-in patient can only ever file a certificate against themselves.
    # The form field is advisory; the session is authoritative. Without this a
    # patient could type someone else's ID and attach an EMC to their chart.
    if actor is not None and actor.is_patient and actor.patient_external_id:
        patient_id = actor.patient_external_id
        patient_name = actor.display_name or patient_name

    if not re.fullmatch(r"[A-Za-z0-9-]{3,32}", patient_id):
        raise ValueError("Patient ID may contain only letters, numbers, and hyphens.")
    patient_age = as_int(form.get("patient_age"), -1)
    leave_days = as_int(form.get("authorized_medical_leave_days"), 0)
    if not 0 <= patient_age <= 130:
        raise ValueError("Patient age must be between 0 and 130.")
    if not 1 <= leave_days <= 60:
        raise ValueError("Authorised leave days must be between 1 and 60.")
    return {
        "patient_name": patient_name,
        "patient_id": patient_id,
        "patient_age": patient_age,
        "clinic_name": normalize_text(form.get("clinic_name"), "Clinic name", 120),
        "clinic_address": normalize_text(form.get("clinic_address"), "Clinic address", 240),
        "attending_clinician_name": normalize_text(form.get("attending_clinician_name"), "Attending clinician", 120),
        "clinician_registration_no": normalize_text(form.get("clinician_registration_no"), "Clinician registration number", 64),
        "consultation_date": parse_iso_date(form.get("consultation_date", today), "Consultation date"),
        "medical_leave_start_date": parse_iso_date(form.get("medical_leave_start_date", today), "Leave start date"),
        "authorized_medical_leave_days": leave_days,
        "diagnosis_disclosure_consent": form.get("diagnosis_disclosure_consent") == "on",
        "clinician_review_status": "PENDING_REVIEW",
    }


def collect_features(form, metadata):
    values = {}
    for feature in FEATURE_LAYOUT:
        if feature == "Age":
            values[feature] = metadata["patient_age"]
        elif feature == "Gender":
            values[feature] = 1 if form.get("gender") == "1" else 0
        elif feature == "Duration":
            duration = as_int(form.get("duration"), 1)
            if not 1 <= duration <= 60:
                raise ValueError("Symptom duration must be between 1 and 60 days.")
            values[feature] = duration
        elif feature == "Medical_History":
            values[feature] = 1 if form.get("medical_history") == "on" else 0
        else:
            values[feature] = 1 if form.get(feature) == "on" else 0
    return values


def run_prediction(features):
    frame = pd.DataFrame([features], columns=FEATURE_LAYOUT)
    transformed = frame.copy()
    if SCALER is not None:
        scaler_columns = list(getattr(SCALER, "feature_names_in_", [])) or [name for name in ("Age", "Duration") if name in FEATURE_LAYOUT]
        if scaler_columns:
            transformed[scaler_columns] = SCALER.transform(transformed[scaler_columns])
    predicted = MODEL.predict(transformed)[0]
    probabilities = MODEL.predict_proba(transformed)[0]
    classes = list(MODEL.classes_)
    ranked = np.argsort(probabilities)[::-1]
    return {
        "primary_predicted_diagnosis": predicted,
        "prediction_confidence_percentage": round(float(probabilities[ranked[0]] * 100), 2),
        "prediction_score_label": "calibrated model score; not clinical certainty" if MODEL_METADATA else "model score; not clinical certainty",
        "clinical_differential_alternatives": [{"alternative_condition": classes[index], "confidence": round(float(probabilities[index] * 100), 2)} for index in ranked[1:3]],
        "requires_manual_review": True,
        "triage_feature_snapshot": features,
    }


def leave_end(metadata):
    return (date.fromisoformat(metadata["medical_leave_start_date"]) + timedelta(days=metadata["authorized_medical_leave_days"] - 1)).isoformat()


def evidence_for(payload, metadata):
    return {
        "patient_name": metadata["patient_name"], "patient_id": metadata["patient_id"],
        "consultation_date": metadata["consultation_date"],
        "medical_leave_start_date": metadata["medical_leave_start_date"], "medical_leave_end_date": leave_end(metadata),
        "authorized_medical_leave_days": metadata["authorized_medical_leave_days"], "clinic_name": metadata["clinic_name"],
        "clinic_address": metadata["clinic_address"], "attending_clinician_name": metadata["attending_clinician_name"],
        "clinician_registration_no": metadata["clinician_registration_no"],
        "diagnosis_disclosure_consent": metadata["diagnosis_disclosure_consent"],
        "approved_diagnosis_if_disclosed": payload["primary_predicted_diagnosis"] if metadata["diagnosis_disclosure_consent"] else None,
    }


def policy_context():
    return "\n\n".join(f"[{item['id']}] {item['text']}" for item in POLICY_CORPUS)


def offline_template(evidence):
    diagnosis = f"\nDiagnosis / Medical Details: {evidence['approved_diagnosis_if_disclosed']}" if evidence["diagnosis_disclosure_consent"] else ""
    return f"""DRAFT - PENDING CLINICIAN REVIEW

1. DRAFT STATUS NOTICE
This is an offline template preview, not live GenAI output and not valid until clinician approval.

2. PATIENT AND CONSULTATION DETAILS
Patient Name: {evidence['patient_name']}
Patient ID: {evidence['patient_id']}
Consultation Date: {evidence['consultation_date']}
{diagnosis}

3. MEDICAL LEAVE PERIOD
Medical Leave Start Date: {evidence['medical_leave_start_date']}
Medical Leave End Date: {evidence['medical_leave_end_date']}
Medical Leave Duration: {evidence['authorized_medical_leave_days']} day(s)

4. CLINICIAN REVIEW AND APPROVAL BLOCK
Clinic Name: {evidence['clinic_name']}
Clinic Address: {evidence['clinic_address']}
Attending Clinician: {evidence['attending_clinician_name']}
Clinician Registration No: {evidence['clinician_registration_no']}"""


def generate_draft(evidence, style, api_key, revision="", previous=""):
    revision_context = f"Clinician revision request: {revision}\nPrevious draft:\n{previous}" if revision else "No revision requested."
    prompt = f"""Prompt version: {PROMPT_VERSION}
Style: {STYLE_INSTRUCTIONS[style]}

CONFIRMED EVIDENCE:
{json.dumps(evidence, indent=2)}

POLICY CONTEXT:
{policy_context()}

REVISION CONTEXT:
{revision_context}

Output only a patient-facing EMC draft with headings: DRAFT STATUS NOTICE; PATIENT AND CONSULTATION DETAILS; MEDICAL LEAVE PERIOD; CLINICIAN REVIEW AND APPROVAL BLOCK."""
    result = call_live_text("You are a clinical administrative documentation assistant. Use only confirmed evidence and policy. Do not invent facts, medical advice, ML information, or audit information.", prompt, api_key)
    if result["status"] == "LIVE":
        return {"mode": "LIVE_GENAI", "text": result["text"], "error": None}
    return {"mode": "OFFLINE_TEMPLATE_NOT_GENAI", "text": offline_template(evidence), "error": result["error"]}


def comparable_text(value):
    """Make cosmetic punctuation and whitespace differences irrelevant to safety checks."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def date_variants(value):
    """Accept common human-readable renderings of a confirmed ISO date."""
    parsed = date.fromisoformat(str(value))
    day = str(parsed.day)
    month = parsed.strftime("%B")
    short_month = parsed.strftime("%b")
    return {
        parsed.isoformat(), f"{month} {day}, {parsed.year}", f"{day} {month} {parsed.year}",
        f"{short_month} {day}, {parsed.year}", f"{parsed.day:02d}/{parsed.month:02d}/{parsed.year}",
        f"{parsed.month:02d}/{parsed.day:02d}/{parsed.year}",
    }


def evidence_value_present(text, value, is_date=False):
    comparable_draft = comparable_text(text)
    variants = date_variants(value) if is_date else {str(value)}
    return any(comparable_text(variant) in comparable_draft for variant in variants)


def deterministic_review(text, evidence, payload):
    issues, lowered = [], text.lower()
    required = {
        "patient name": (evidence["patient_name"], False),
        "patient id": (evidence["patient_id"], False),
        "consultation date": (evidence["consultation_date"], True),
        "leave start": (evidence["medical_leave_start_date"], True),
        "leave end": (evidence["medical_leave_end_date"], True),
        "clinic": (evidence["clinic_name"], False),
        "clinician": (evidence["attending_clinician_name"], False),
        "registration": (evidence["clinician_registration_no"], False),
    }
    for field, (value, is_date) in required.items():
        if not evidence_value_present(text, value, is_date=is_date):
            issues.append({"severity": "critical", "category": "missing_required_fact", "detail": field})
    diagnosis = str(payload["primary_predicted_diagnosis"]).lower()
    if not evidence["diagnosis_disclosure_consent"] and diagnosis in lowered:
        issues.append({"severity": "critical", "category": "diagnosis_disclosure", "detail": "Diagnosis appears without consent."})
    for term in FORBIDDEN_TERMS:
        if term in lowered:
            issues.append({"severity": "critical", "category": "internal_information_leak", "detail": term})
    return {"mode": "DETERMINISTIC", "verdict": "PASS" if not issues else "REVIEW_REQUIRED", "issues": issues}


def critic_review(text, evidence, api_key):
    prompt = f"""CONFIRMED EVIDENCE:
{json.dumps(evidence, indent=2)}

POLICY:
{policy_context()}

CERTIFICATE DRAFT (PENDING CLINICIAN REVIEW, NOT A FINAL ISSUED EMC):
{text}

Return JSON with verdict (PASS or REVIEW_REQUIRED), summary, and issues (severity, category, detail).

Return REVIEW_REQUIRED only for a concrete, material evidence or policy breach. Do not require certificate ID, approval date, approval notes, signature, or final approval information: those belong to the final EMC after clinician approval, not to this pending draft. Do not flag a draft merely because it is pending review, omits a diagnosis without disclosure consent, uses plain-text formatting, or is concise. Do not invent additional requirements."""
    result = call_live_json("You are a patient-facing EMC safety reviewer. Check only evidence and policy; do not provide clinical advice.", prompt, api_key)
    if result["status"] == "LIVE" and isinstance(result.get("json"), dict):
        review = result["json"]
        verdict = str(review.get("verdict", "REVIEW_REQUIRED")).upper()
        issues = review.get("issues", [])
        if not isinstance(issues, list):
            issues = []
        return {
            "mode": "LIVE_GENAI_CRITIC",
            "verdict": verdict if verdict in {"PASS", "REVIEW_REQUIRED"} else "REVIEW_REQUIRED",
            "summary": str(review.get("summary", "No critic summary provided.")),
            "issues": issues,
        }
    return {"mode": "OFFLINE_NO_GENAI_CRITIC", "verdict": "NOT_RUN", "summary": "Live critic unavailable.", "issues": []}


def review_gate(local, critic):
    critical_critic_issues = [
        issue for issue in critic.get("issues", [])
        if isinstance(issue, dict) and str(issue.get("severity", "")).lower() == "critical"
    ]
    critic_blocks_issue = critic.get("verdict") == "REVIEW_REQUIRED" and bool(critical_critic_issues)
    return {
        "approval_allowed": local["verdict"] == "PASS" and not critic_blocks_issue,
        "local": local,
        "critic": critic,
        "critical_critic_issues": critical_critic_issues,
    }


def internal_summary(payload, evidence, gate):
    positives = [label_for(name) for name, value in payload["triage_feature_snapshot"].items() if int(value) == 1]
    return {
        "model_label": payload["primary_predicted_diagnosis"],
        "model_score": payload["prediction_confidence_percentage"],
        "score_label": payload["prediction_score_label"],
        "alternatives": payload["clinical_differential_alternatives"],
        "active_symptoms": positives,
        "manual_review_required": True,
        "safety_verdict": gate["local"]["verdict"],
    }


def extract_note(note, api_key):
    if not note.strip():
        return {"status": "NOT_REQUESTED", "proposals": None}
    prompt = f"""Extract only explicitly stated administrative values from this clinician note. Use null for missing values. Do not diagnose or infer leave.

NOTE:
{note}

Return JSON with patient_name, patient_id, consultation_date, medical_leave_start_date, authorized_medical_leave_days, clinic_name, attending_clinician_name, diagnosis_disclosure_consent, unresolved_fields."""
    result = call_live_json("You extract proposed facts for clinician confirmation.", prompt, api_key)
    return {"status": result["status"], "proposals": result.get("json"), "error": result.get("error")}


def make_audit(workflow):
    evidence, draft = workflow["evidence"], workflow["draft"]
    preview = draft["text"].replace(evidence["patient_name"], "[REDACTED_PATIENT_NAME]").replace(evidence["patient_id"], "[REDACTED_PATIENT_ID]")[:1000]
    return {
        "audit_version": "v4.0", "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION, "configured_model": GENAI_MODEL, "certificate_id": workflow["metadata"].get("certificate_id"),
        "issue_status": workflow["issue_status"], "patient_id_hash": hashlib.sha256(evidence["patient_id"].encode()).hexdigest(),
        "policy_source_ids": [item["id"] for item in POLICY_CORPUS], "draft_generation_mode": draft["mode"],
        "draft_hash": hashlib.sha256(draft["text"].encode()).hexdigest(), "redacted_draft_preview": preview,
        "draft_gate": workflow["gate"], "revision_history": workflow["revision_history"],
    }


def store(workflow):
    workflow_id = uuid.uuid4().hex
    WORKFLOWS[workflow_id] = workflow
    persist_workflow(workflow_id, workflow, "emc_draft_created")
    return workflow_id


def workflow_snapshot(workflow):
    """Persist clinical facts and outputs, never the live provider API key."""
    return {
        "style": workflow["style"],
        "metadata": workflow["metadata"],
        "features": workflow["features"],
        "prediction": workflow["payload"],
        "evidence": workflow["evidence"],
        "draft": workflow["draft"],
        "review_gate": workflow["gate"],
        "note_extraction": workflow["note_extraction"],
        "revision_history": workflow["revision_history"],
        "final": workflow["final"],
        "issue_status": workflow["issue_status"],
    }


def persist_workflow(workflow_id, workflow, event_type):
    """Mirror an EMC workflow to PostgreSQL when DATABASE_URL is configured."""
    snapshot = workflow_snapshot(workflow)
    owner_user_id = workflow.get("owner_user_id")
    record_id = workflow.get("database_record_id")
    if record_id:
        SHARED_STORE.safe_update_record(
            record_id,
            status=workflow["issue_status"],
            output_payload=snapshot,
            owner_user_id=owner_user_id,
        )
    else:
        record_id = SHARED_STORE.safe_create_record(
            source_app="emc",
            record_type="electronic_medical_certificate",
            status=workflow["issue_status"],
            input_payload={"metadata": workflow["metadata"], "features": workflow["features"]},
            output_payload=snapshot,
            model_version=PROMPT_VERSION,
            patient_external_id=workflow["metadata"]["patient_id"],
            patient_name=workflow["metadata"]["patient_name"],
            owner_user_id=owner_user_id,
        )
        if record_id:
            workflow["database_record_id"] = record_id
    SHARED_STORE.safe_append_audit_event(
        source_app="emc",
        event_type=event_type,
        record_id=record_id,
        # The authenticated account, not the typed clinician name — free-text a
        # user supplies is not an audit actor.
        actor_reference=workflow.get("actor_email") or workflow["metadata"].get("attending_clinician_name"),
        payload={"workflow_id": workflow_id, "status": workflow["issue_status"]},
    )
    return record_id


def status_class(value):
    status = str(value or "").upper()
    if status in {"PASS", "APPROVED", "APPROVED_FOR_ISSUE"}:
        return "pass"
    if status in {"REJECTED", "REVIEW_REQUIRED"} or status.startswith("BLOCKED"):
        return "block"
    if status in {"LIVE", "LIVE_GENAI", "LIVE_GENAI_CRITIC"}:
        return "live"
    return "pending"


app.jinja_env.globals.update(status_class=status_class)


def patient_identity():
    """Locked patient details for a signed-in patient, or None for clinicians."""
    actor = current_actor()
    if actor is None or not actor.is_patient or not actor.patient_external_id:
        return None
    return {"external_id": actor.patient_external_id, "name": actor.display_name or ""}


def render_state(workflow_id=None, message=None):
    workflow = WORKFLOWS.get(workflow_id)
    return render_template("index.html", title=APP_TITLE, today=date.today().isoformat(), symptom_fields=symptom_fields(), style_options=STYLE_OPTIONS, workflow=workflow, workflow_id=workflow_id, message=message, model_metadata=MODEL_METADATA, feature_count=len(FEATURE_LAYOUT), render_certificate=render_certificate, legacy_schema=any(feature in FEATURE_LAYOUT for feature in ("Gender", "Duration", "Medical_History")), genai_configured=bool(get_api_key()), patient_identity=patient_identity())


@app.get("/")
def index():
    return render_state()


@app.post("/start")
def start():
    actor = current_actor()
    try:
        metadata = collect_metadata(request.form, actor)
        features = collect_features(request.form, metadata)
    except ValueError as exc:
        return render_state(message=f"Input validation: {exc}")
    payload = run_prediction(features)
    payload["patient_admin_metadata"] = metadata
    api_key, style = get_api_key(), request.form.get("style", "standard")
    if style not in STYLE_OPTIONS:
        return render_state(message="Input validation: unsupported certificate style.")
    evidence = evidence_for(payload, metadata)
    draft = generate_draft(evidence, style, api_key)
    gate = review_gate(deterministic_review(draft["text"], evidence, payload), critic_review(draft["text"], evidence, api_key))
    clinician_note = normalize_text(request.form.get("clinician_note"), "Clinician note", 2000, required=False)
    workflow = {"api_key": api_key, "style": style, "metadata": metadata, "features": features, "payload": payload, "evidence": evidence, "draft": draft, "gate": gate, "note_extraction": extract_note(clinician_note, api_key), "revision_history": [], "final": None, "issue_status": "PENDING_REVIEW"}
    # Carried on the workflow so every later persist (revise/approve/reject)
    # attributes to the account that started it, not to whoever is posting now.
    workflow["owner_user_id"] = actor.user_id if actor else None
    workflow["actor_email"] = actor.email if actor else None
    workflow["internal"] = internal_summary(payload, evidence, gate)
    workflow_id = store(workflow)
    persist_workflow(workflow_id, workflow, "emc_draft_created")
    return render_state(workflow_id, "Draft ready for clinician review.")


@app.post("/revise/<workflow_id>")
def revise(workflow_id):
    workflow = WORKFLOWS[workflow_id]
    instructions = request.form.get("revision_instructions", "").strip()
    if not instructions:
        return render_state(workflow_id, "Enter revision instructions before regenerating.")
    workflow["revision_history"].append({"at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "instructions": instructions, "previous_draft_hash": hashlib.sha256(workflow["draft"]["text"].encode()).hexdigest()})
    workflow["draft"] = generate_draft(workflow["evidence"], workflow["style"], workflow["api_key"], instructions, workflow["draft"]["text"])
    workflow["gate"] = review_gate(deterministic_review(workflow["draft"]["text"], workflow["evidence"], workflow["payload"]), critic_review(workflow["draft"]["text"], workflow["evidence"], workflow["api_key"]))
    workflow["internal"] = internal_summary(workflow["payload"], workflow["evidence"], workflow["gate"])
    persist_workflow(workflow_id, workflow, "emc_draft_revised")
    return render_state(workflow_id, "Draft regenerated from clinician instructions.")


@app.post("/approve/<workflow_id>")
def approve(workflow_id):
    workflow = WORKFLOWS[workflow_id]
    if workflow["draft"]["mode"] != "LIVE_GENAI":
        return render_state(workflow_id, "Final issue is blocked: the current draft is an offline template, not live GenAI output.")
    if not workflow["gate"]["approval_allowed"]:
        return render_state(workflow_id, "Final issue is blocked by the safety review.")
    metadata = workflow["metadata"]
    metadata["attending_clinician_name"] = request.form.get("approving_clinician_name", metadata["attending_clinician_name"]).strip()
    metadata["clinician_registration_no"] = request.form.get("approving_registration_no", metadata["clinician_registration_no"]).strip()
    metadata["approval_notes"] = request.form.get("approval_notes", "Reviewed and approved for issue").strip()
    metadata["approval_date"] = date.today().isoformat()
    metadata["certificate_id"] = f"EMC-{date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    metadata["clinician_review_status"] = "APPROVED"
    workflow["evidence"] = evidence_for(workflow["payload"], metadata)
    prompt = f"""CONFIRMED EVIDENCE:\n{json.dumps(workflow['evidence'], indent=2)}\n\nAPPROVAL:\n{json.dumps({'certificate_id': metadata['certificate_id'], 'approval_date': metadata['approval_date'], 'approval_notes': metadata['approval_notes']})}\n\nPOLICY:\n{policy_context()}\n\nREVIEWED DRAFT:\n{workflow['draft']['text']}\n\nOutput only the final EMC with headings: ELECTRONIC MEDICAL CERTIFICATE; PATIENT AND CONSULTATION DETAILS; MEDICAL LEAVE CERTIFICATION; ISSUING CLINICIAN AND CLINIC; ELECTRONIC APPROVAL STATEMENT."""
    result = call_live_text("You generate a final EMC after documented clinician approval. Use only confirmed evidence and policy.", prompt, workflow["api_key"])
    if result["status"] != "LIVE":
        return render_state(workflow_id, "Final issue is blocked because final live GenAI generation failed.")
    final_gate = review_gate(deterministic_review(result["text"], workflow["evidence"], workflow["payload"]), critic_review(result["text"], workflow["evidence"], workflow["api_key"]))
    if not final_gate["approval_allowed"]:
        workflow["issue_status"] = "BLOCKED_FINAL_SAFETY_REVIEW"
        return render_state(workflow_id, "Final issue is blocked by the final safety review.")
    workflow["final"], workflow["final_gate"], workflow["issue_status"] = result["text"], final_gate, "APPROVED_FOR_ISSUE"
    audit = make_audit(workflow)
    record_id = persist_workflow(workflow_id, workflow, "emc_approved")
    SHARED_STORE.safe_append_audit_event(
        source_app="emc", event_type="emc_audit_snapshot", record_id=record_id, payload=audit
    )
    if record_id:
        workflow["audit_path"] = f"shared-database:{record_id}"
    else:
        joblib.dump(audit, AUDIT_PATH)
        workflow["audit_path"] = AUDIT_PATH
    return render_state(workflow_id, "Final EMC generated after clinician approval.")


@app.post("/reject/<workflow_id>")
def reject(workflow_id):
    workflow = WORKFLOWS[workflow_id]
    workflow["metadata"]["clinician_review_status"] = "REJECTED"
    workflow["issue_status"] = "REJECTED"
    persist_workflow(workflow_id, workflow, "emc_rejected")
    return render_state(workflow_id, "Draft rejected. Add clinician revision instructions to prepare a new draft.")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
