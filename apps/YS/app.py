import hashlib
import json
import os
import re
import uuid
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, redirect, render_template, request
from markupsafe import Markup, escape
from openai import OpenAI
from dotenv import load_dotenv
from vitalhealth_storage import get_store, identity, load_shared_env, missing_shared_keys


load_dotenv()
# SESSION_SECRET and DATABASE_URL are shared with the gateway, not local
# config. This app's own .env ships them blank, and a blank value here means
# every gateway-signed cookie fails verification and every record write is a
# no-op — so fill them from the shared source before anything reads them.
load_shared_env()
for _key in missing_shared_keys():
    print(f"[YS] WARNING: {_key} is not set - sessions and saved records will not work.")
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
    """Persist first, then use the DB row's own id as the in-process key.

    Previously this minted an independent uuid4().hex as workflow_id and the
    DB row got a *different* id from persist_workflow() — two unrelated
    identifiers for the same workflow, and the only reason a review was ever
    reachable was by the exact URL a clinician had just been shown. Making the
    DB id authoritative is what lets a queue query (list_pending_review) hand
    back an id that /revise, /approve, /reject, and the new /review route can
    all load directly, including from a different process after a restart
    (see load_workflow()).
    """
    persist_workflow(workflow, "emc_draft_created")
    # safe_create_record swallows write failures rather than raising (DB down,
    # a constraint violation) — persistence must not turn an otherwise valid
    # workflow into a 500. Fall back to an in-memory-only id in that case,
    # same as this function did before database ids were made authoritative;
    # it just won't be queue-discoverable or reviewable from another process.
    record_id = workflow.get("database_record_id") or uuid.uuid4().hex
    WORKFLOWS[record_id] = workflow
    return record_id


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
        # Who originally submitted this — kept separate from "who is acting
        # right now" (persist_workflow's actor_reference) so a later reviewer
        # approving/rejecting doesn't overwrite the original submitter's
        # attribution, and so it survives being reconstructed from the DB by
        # load_workflow() after a restart.
        "submitted_by_email": workflow.get("submitted_by_email"),
    }


def persist_workflow(workflow, event_type, *, expected_status=None):
    """Mirror an EMC workflow to PostgreSQL when DATABASE_URL is configured."""
    snapshot = workflow_snapshot(workflow)
    owner_user_id = workflow.get("owner_user_id")
    record_id = workflow.get("database_record_id")
    if record_id:
        if expected_status is not None:
            persisted = SHARED_STORE.safe_update_record_if_status(
                record_id,
                expected_status=expected_status,
                status=workflow["issue_status"],
                output_payload=snapshot,
                owner_user_id=owner_user_id,
            )
        else:
            persisted = SHARED_STORE.safe_update_record(
                record_id,
                status=workflow["issue_status"],
                output_payload=snapshot,
                owner_user_id=owner_user_id,
            )
        if not persisted:
            return None
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
        # Whoever is acting on this specific call (set by /review's POST
        # handler when a different clinician reviews someone else's
        # submission) takes precedence; otherwise fall back to who submitted
        # it, then the older actor_email/typed-name fallbacks for workflows
        # created before this distinction existed.
        actor_reference=(
            workflow.get("acting_actor_email")
            or workflow.get("submitted_by_email")
            or workflow.get("actor_email")
            or workflow["metadata"].get("attending_clinician_name")
        ),
        payload={"record_id": record_id, "status": workflow["issue_status"]},
    )
    return record_id


def workflow_from_record(record):
    """Rebuild an in-process workflow dict from a persisted DB row.

    The inverse of workflow_snapshot(). Lets a workflow started in one
    process (or by a patient's /submit) be reviewed from any other process —
    a fresh clinician session, or the same one after a restart — since
    WORKFLOWS is otherwise empty there. api_key is re-derived rather than
    read from the snapshot: it is deliberately never persisted (see
    workflow_snapshot's docstring) because it's a single shared env var, not
    per-workflow secret state.
    """
    snapshot = record.get("output_payload") or {}
    input_payload = record.get("input_payload") or {}
    # Demo/seed rows (demo_data/seed_db.py) only ever ran the raw ML step —
    # their output_payload is just {"prediction": {...}}, with no draft,
    # evidence, or review_gate at all, and their metadata/features live in
    # input_payload instead of the output snapshot. A record submitted
    # through /submit or /start always has the full shape; this is only for
    # records this workflow never produced.
    has_full_snapshot = "draft" in snapshot
    payload = snapshot.get("prediction") or {}
    evidence = snapshot.get("evidence") or {}
    gate = snapshot.get("review_gate") or {
        "local": {"mode": "DETERMINISTIC", "verdict": "REVIEW_REQUIRED", "issues": []},
        "critic": {
            "mode": "NOT_RUN", "verdict": "NOT_RUN",
            "summary": "No certificate draft has been generated for this record.",
            "issues": [],
        },
        "approval_allowed": False,
        "critical_critic_issues": [],
    }
    return {
        "api_key": get_api_key(),
        "style": snapshot.get("style", "standard"),
        "metadata": snapshot.get("metadata") or input_payload.get("metadata") or {},
        "features": snapshot.get("features") or input_payload.get("features") or {},
        "payload": payload,
        "evidence": evidence,
        "draft": snapshot.get("draft") or {"mode": "NO_DRAFT", "text": "", "error": None},
        "gate": gate,
        "note_extraction": snapshot.get("note_extraction", {}),
        "revision_history": snapshot.get("revision_history", []),
        "final": snapshot.get("final"),
        "issue_status": record.get("status"),
        "database_record_id": record["id"],
        "owner_user_id": record.get("owner_user_id"),
        "submitted_by_email": snapshot.get("submitted_by_email"),
        "has_reviewable_draft": has_full_snapshot,
        "internal": internal_summary(payload, evidence, gate) if payload else None,
    }


def load_workflow(record_id):
    """WORKFLOWS first (same-process fast path), else reconstruct from the
    DB. 404s on an unknown or non-EMC id instead of the raw dict-index
    KeyError->500 this replaces at every /revise, /approve, /reject call site.
    """
    workflow = WORKFLOWS.get(record_id)
    if workflow is not None:
        return workflow
    record = SHARED_STORE.get_record(record_id)
    if record is None or record.get("source_app") != "emc":
        abort(404)
    workflow = workflow_from_record(record)
    WORKFLOWS[record_id] = workflow
    return workflow


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
    workflow["submitted_by_email"] = actor.email if actor else None
    workflow["has_reviewable_draft"] = True
    workflow["internal"] = internal_summary(payload, evidence, gate)
    workflow_id = store(workflow)
    return render_state(workflow_id, "Draft ready for clinician review.")


# Clinic/clinician identity is administrative information a patient
# self-submitting a request has no reason to know in advance (their reviewing
# clinician's registration number, specifically). collect_metadata still
# requires these fields non-empty, so the patient form supplies placeholders
# and the reviewing clinician fills in the real values on /review before
# approving — see review_form/review_submit below.
_PENDING_CLINIC_METADATA = {
    "clinic_name": "Pending clinician review",
    "clinic_address": "Pending clinician review",
    "attending_clinician_name": "Pending clinician review",
    "clinician_registration_no": "PENDING",
}


@app.get("/submit")
def submit_form():
    actor = current_actor()
    if actor is None or not actor.is_patient:
        abort(403)
    return render_template(
        "submit.html", title=APP_TITLE, today=date.today().isoformat(),
        symptom_fields=symptom_fields(), patient_identity=patient_identity(), message=None,
    )


@app.post("/submit")
def submit():
    """Patient self-submission. Reuses the exact same pipeline /start does —
    collect_metadata's dormant actor.is_patient branch (this module, above)
    locks patient_id/patient_name to the signed-in patient here — but never
    renders the workflow state back to the caller. Nothing model-derived is
    shown; the response is only a generic "submitted" confirmation."""
    actor = current_actor()
    if actor is None or not actor.is_patient:
        abort(403)
    form = request.form.copy()
    for field, placeholder in _PENDING_CLINIC_METADATA.items():
        form.setdefault(field, placeholder)
    try:
        metadata = collect_metadata(form, actor)
        features = collect_features(form, metadata)
    except ValueError as exc:
        return render_template(
            "submit.html", title=APP_TITLE, today=date.today().isoformat(),
            symptom_fields=symptom_fields(), patient_identity=patient_identity(),
            message=f"Input validation: {exc}",
        )
    payload = run_prediction(features)
    payload["patient_admin_metadata"] = metadata
    api_key, style = get_api_key(), "standard"
    evidence = evidence_for(payload, metadata)
    draft = generate_draft(evidence, style, api_key)
    gate = review_gate(deterministic_review(draft["text"], evidence, payload), critic_review(draft["text"], evidence, api_key))
    workflow = {
        "api_key": api_key, "style": style, "metadata": metadata, "features": features,
        "payload": payload, "evidence": evidence, "draft": draft, "gate": gate,
        "note_extraction": {"status": "NOT_REQUESTED", "proposals": None},
        "revision_history": [], "final": None, "issue_status": "PENDING_REVIEW",
        "owner_user_id": actor.user_id, "actor_email": actor.email, "submitted_by_email": actor.email,
        "has_reviewable_draft": True,
    }
    workflow["internal"] = internal_summary(payload, evidence, gate)
    record_id = store(workflow)
    return redirect(_prefixed_url_for("submitted", record_id=record_id))


@app.get("/submitted/<record_id>")
def submitted(record_id):
    """The patient's waiting page. Renders no draft itself — it polls
    /status/<record_id> and reveals the issued certificate in place once a
    clinician has approved it, so the patient can simply wait here instead of
    being sent away to the dashboard."""
    return render_template("submitted.html", title=APP_TITLE, record_id=record_id)


# Lines asserting the document is still a pending draft. Once a clinician
# approves, they are false — an issued certificate that calls itself a draft
# is worse than useless to the patient holding it.
_DRAFT_NOTICE_MARKERS = (
    "draft status notice",
    "is a draft",
    "pending clinician review and approval",
    "not valid until clinician approval",
    # offline_template()'s header, once punctuation is normalised away
    "draft pending clinician review",
)


def _is_draft_notice_line(line):
    """True for prose asserting draft status, never for a `- Field: value` row.

    Field rows are protected explicitly: a value like "Pending clinician
    review" sitting in a clinic-name field must not silently disappear from
    the certificate — the safety gate is what stops that reaching approval.
    """
    stripped = line.strip()
    if stripped.startswith("-") or ":" in stripped.split("**")[-1]:
        return False
    # Strip punctuation, then collapse the whitespace it leaves behind, so
    # "DRAFT - PENDING ..." and "DRAFT PENDING ..." normalise identically.
    lowered = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", stripped.lower())).strip()
    return any(marker in lowered for marker in _DRAFT_NOTICE_MARKERS)


def finalise_certificate_text(text, metadata):
    """Turn the reviewed draft into the issued certificate, verbatim except
    for its status: the draft notice is dropped and an approval record takes
    its place. Everything the clinician wrote is preserved as written."""
    kept = [line for line in str(text or "").splitlines() if not _is_draft_notice_line(line)]
    body = "\n".join(kept).lstrip("\n")

    approval_block = "\n".join([
        "ELECTRONIC APPROVAL STATEMENT",
        f"Certificate ID: {metadata.get('certificate_id', '')}",
        f"Approved by: {metadata.get('approving_clinician_name') or metadata.get('attending_clinician_name', '')}",
        f"Clinician Registration No: {metadata.get('clinician_registration_no', '')}",
        f"Approval Date: {metadata.get('approval_date', '')}",
        "This certificate has been reviewed and approved for issue by the named clinician.",
        "",
    ])
    return f"{approval_block}\n{body}".strip() + "\n"


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
    actor = current_actor()
    record = SHARED_STORE.get_record(record_id) if SHARED_STORE.enabled else None
    if (
        actor is None
        or not actor.is_patient
        or record is None
        or record.get("source_app") != "emc"
        or not _patient_owns_record(actor, record)
    ):
        abort(404)
    return redirect(_prefixed_url_for("submitted", record_id=record_id))


@app.get("/status/<record_id>")
def submission_status(record_id):
    """Patient-facing poll target for the waiting page.

    Returns only the approved, patient-facing certificate. While a record is
    pending the response carries no certificate text at all, and it never
    carries the model's diagnosis, confidence, differentials, or review-gate
    internals in any state — policy EMC-005 (see POLICY_CORPUS) makes those
    clinician-only regardless of approval.
    """
    actor = current_actor()
    if actor is None:
        return jsonify({"error": "authentication_required"}), 401

    record = SHARED_STORE.get_record(record_id) if SHARED_STORE.enabled else None
    if record is None or record.get("source_app") != "emc":
        abort(404)

    # 404 rather than 403: a patient probing ids should not be able to learn
    # which ones exist. Clinicians may read any record.
    if not actor.is_clinician and not _patient_owns_record(actor, record):
        abort(404)

    snapshot = record.get("output_payload") or {}
    status = str(snapshot.get("issue_status") or record.get("status") or "").upper()
    payload = {"status": status, "state": "pending", "ready": False, "result": None}

    if status in ("REJECTED", "BLOCKED_FINAL_SAFETY_REVIEW"):
        payload["state"] = "rejected"
        return jsonify(payload)

    if status != "APPROVED_FOR_ISSUE":
        return jsonify(payload)

    metadata = snapshot.get("metadata") or {}
    payload.update({
        "state": "approved",
        "ready": True,
        "result": {
            "certificate_text": snapshot.get("final") or "",
            "certificate_id": metadata.get("certificate_id"),
            "leave_start": metadata.get("medical_leave_start_date"),
            "leave_days": metadata.get("authorized_medical_leave_days"),
            "clinic_name": metadata.get("clinic_name"),
            "approved_by": metadata.get("approving_clinician_name") or metadata.get("attending_clinician_name"),
        },
    })
    return jsonify(payload)


@app.get("/review/<record_id>")
def review_form(record_id):
    workflow = load_workflow(record_id)
    return render_template(
        "review.html", title=APP_TITLE, workflow=workflow, record_id=record_id,
        render_certificate=render_certificate, message=None,
    )


@app.post("/review/<record_id>")
def review_submit(record_id):
    workflow = load_workflow(record_id)
    actor = current_actor()
    # A normal click on one of the named buttons sends its action. Treat an
    # action-less submission as Save as well: browsers can submit a form with
    # Enter and an already-open page may still have markup from immediately
    # before the named Save button was introduced. Saving is the least
    # privileged review action; approval and rejection always remain explicit.
    action = (request.form.get("action") or "save").strip().lower()
    if action not in {"save", "regenerate", "approve", "reject"}:
        abort(400)

    # Defence in depth: the UI hides the edit form for a record with no real
    # draft (a demo/seed row that only ever ran the ML step), but a direct
    # POST could still reach here. has_reviewable_draft is absent (not False)
    # on every workflow this app actually creates itself, so this only ever
    # trips for the DB-reconstruction fallback's synthetic placeholder.
    if action in ("save", "approve") and not workflow.get("has_reviewable_draft", True):
        return render_template(
            "review.html", title=APP_TITLE, workflow=workflow, record_id=record_id,
            render_certificate=render_certificate,
            message="This record has no certificate draft to save or approve.",
        )

    if action in ("save", "approve", "regenerate"):
        # On regenerate the textarea contents are deliberately discarded — the
        # point of the action is to rebuild the text from the confirmed
        # details, which is the only way to resync after editing clinic or
        # clinician fields the original draft was generated without.
        edited_text = request.form.get("draft_text", "").strip()
        if edited_text and action != "regenerate":
            workflow["draft"] = {**workflow["draft"], "text": edited_text, "mode": "CLINICIAN_EDITED"}

        metadata = workflow["metadata"]
        for field in ("clinic_name", "clinic_address", "attending_clinician_name", "clinician_registration_no"):
            value = request.form.get(field, "").strip()
            if value:
                metadata[field] = value
        leave_days = request.form.get("authorized_medical_leave_days")
        if leave_days:
            try:
                metadata["authorized_medical_leave_days"] = as_int(leave_days, metadata["authorized_medical_leave_days"])
            except (TypeError, ValueError):
                pass
        leave_start = request.form.get("medical_leave_start_date")
        if leave_start:
            try:
                metadata["medical_leave_start_date"] = parse_iso_date(leave_start, "Leave start date")
            except ValueError:
                pass

        workflow["evidence"] = evidence_for(workflow["payload"], metadata)

        if action == "regenerate":
            workflow["draft"] = generate_draft(
                workflow["evidence"], workflow["style"], workflow["api_key"]
            )

        workflow["gate"] = review_gate(
            deterministic_review(workflow["draft"]["text"], workflow["evidence"], workflow["payload"]),
            critic_review(workflow["draft"]["text"], workflow["evidence"], workflow["api_key"]),
        )
        workflow["internal"] = internal_summary(workflow["payload"], workflow["evidence"], workflow["gate"])
        workflow["acting_actor_email"] = actor.email if actor else None

    if action in ("save", "regenerate"):
        if not persist_workflow(
            workflow,
            "emc_draft_regenerated" if action == "regenerate" else "emc_reviewer_edited",
            expected_status="PENDING_REVIEW",
        ):
            # A locally cached workflow may now contain edits from the losing
            # reviewer. Discard it so the next GET reconstructs the winner's
            # database state instead of displaying stale in-process data.
            WORKFLOWS.pop(record_id, None)
            abort(409, description="This certificate request was already reviewed by another clinician. Refresh the page.")
        return redirect(_prefixed_url_for("review_form", record_id=record_id))

    if action == "approve":
        # The safety gate is enforced HERE, not by the disabled attribute on
        # the approve button — a direct POST bypasses the markup entirely.
        # This is the same check /approve/<workflow_id> has always made; the
        # review route needs it just as much, and without it a certificate
        # whose text contradicts its own confirmed evidence can be issued.
        if not workflow["gate"].get("approval_allowed"):
            return render_template(
                "review.html", title=APP_TITLE, workflow=workflow, record_id=record_id,
                render_certificate=render_certificate,
                message=(
                    "Approval is blocked by the safety review. If you changed the clinic or "
                    "clinician details, use "
                    "“Regenerate draft from current details” so the certificate text "
                    "matches them, then approve."
                ),
            )

        metadata = workflow["metadata"]
        metadata["approving_clinician_name"] = (actor.display_name or actor.email) if actor else metadata.get("attending_clinician_name")
        metadata["approval_date"] = date.today().isoformat()
        metadata["certificate_id"] = f"EMC-{date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        metadata["clinician_review_status"] = "APPROVED"
        # The clinician's reviewed/edited draft text IS the final text — no
        # separate LLM call to regenerate it. Once real hand-editing exists,
        # a fresh regeneration has no memory of what the clinician just
        # deliberately edited or removed, so it could silently reintroduce it.
        # It only gets its draft-status notice swapped for the approval record,
        # since by definition it is no longer pending.
        workflow["final"] = finalise_certificate_text(workflow["draft"]["text"], metadata)
        workflow["issue_status"] = "APPROVED_FOR_ISSUE"
        if not persist_workflow(workflow, "emc_approved", expected_status="PENDING_REVIEW"):
            WORKFLOWS.pop(record_id, None)
            abort(409, description="This certificate request was already reviewed by another clinician. Refresh the page.")
        return redirect(_prefixed_url_for("review_form", record_id=record_id))

    if action == "reject":
        workflow["metadata"]["clinician_review_status"] = "REJECTED"
        workflow["issue_status"] = "REJECTED"
        workflow["acting_actor_email"] = actor.email if actor else None
        if not persist_workflow(workflow, "emc_rejected", expected_status="PENDING_REVIEW"):
            WORKFLOWS.pop(record_id, None)
            abort(409, description="This certificate request was already reviewed by another clinician. Refresh the page.")
        return redirect(_prefixed_url_for("review_form", record_id=record_id))

    abort(400)


@app.post("/revise/<workflow_id>")
def revise(workflow_id):
    workflow = load_workflow(workflow_id)
    instructions = request.form.get("revision_instructions", "").strip()
    if not instructions:
        return render_state(workflow_id, "Enter revision instructions before regenerating.")
    workflow["revision_history"].append({"at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "instructions": instructions, "previous_draft_hash": hashlib.sha256(workflow["draft"]["text"].encode()).hexdigest()})
    workflow["draft"] = generate_draft(workflow["evidence"], workflow["style"], workflow["api_key"], instructions, workflow["draft"]["text"])
    workflow["gate"] = review_gate(deterministic_review(workflow["draft"]["text"], workflow["evidence"], workflow["payload"]), critic_review(workflow["draft"]["text"], workflow["evidence"], workflow["api_key"]))
    workflow["internal"] = internal_summary(workflow["payload"], workflow["evidence"], workflow["gate"])
    if not persist_workflow(workflow, "emc_draft_revised", expected_status="PENDING_REVIEW"):
        WORKFLOWS.pop(workflow_id, None)
        return render_state(message="This certificate request was already reviewed by another clinician. Refresh the page.")
    return render_state(workflow_id, "Draft regenerated from clinician instructions.")


@app.post("/approve/<workflow_id>")
def approve(workflow_id):
    workflow = load_workflow(workflow_id)
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
    record_id = persist_workflow(workflow, "emc_approved", expected_status="PENDING_REVIEW")
    if not record_id:
        WORKFLOWS.pop(workflow_id, None)
        return render_state(message="This certificate request was already reviewed by another clinician. Refresh the page.")
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
    workflow = load_workflow(workflow_id)
    workflow["metadata"]["clinician_review_status"] = "REJECTED"
    workflow["issue_status"] = "REJECTED"
    if not persist_workflow(workflow, "emc_rejected", expected_status="PENDING_REVIEW"):
        WORKFLOWS.pop(workflow_id, None)
        return render_state(message="This certificate request was already reviewed by another clinician. Refresh the page.")
    return render_state(workflow_id, "Draft rejected. Add clinician revision instructions to prepare a new draft.")


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
