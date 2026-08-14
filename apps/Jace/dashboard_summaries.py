"""Turn a stored clinical_record into something a dashboard tile can render.

Each of the three modules writes a differently shaped output_payload, and this
is the one place that knows all three shapes. The frontend receives a uniform
tile and does no interpretation of its own — the same division of labour
app.js already follows for triage (see its header comment: it renders supplied
facts and computes no clinical fact).

Two rules hold everywhere in this file:

1. Never raise. A dashboard is a read-only view of history; a payload that is
   missing, malformed, or written by an older version of an app degrades to
   "Details unavailable" rather than taking the whole page down.
2. audience="patient" is a redaction boundary, not a styling hint. The EMC
   module's own policy (apps/YS/app.py, EMC-005 and FORBIDDEN_TERMS) forbids
   showing model scores, differential diagnoses, and audit internals to
   patients. A patient-facing tile must not carry them.
"""

from __future__ import annotations

from typing import Any, Mapping

# Only for PATIENT_LEVEL_HEADLINES / PATIENT_COMPLAINT_LABELS — plain constant
# tables, not a call into init()-gated model state, so this stays cheap and
# safe to import at module load. Keeps §0's "single source of truth" literally
# true: the patient-facing wording lives in ctrse_core, this file only looks
# it up rather than re-authoring it.
import ctrse_core as core

MODULE_LABELS = {
    "triage": "Clinical Triage",
    "stroke": "Stroke Assessment",
    "emc": "Medical Certificate",
}

MODULE_HREFS = {
    "triage": "/triage/",
    "stroke": "/stroke/prediction",
    "emc": "/emc/",
}

# A patient's "Start"/"Run again" link must point at the patient-facing
# submission route, not the clinician-instant one — the latter now 403s them.
PATIENT_MODULE_HREFS = {
    "triage": "/triage/self-check.html",
    "stroke": "/stroke/submit",
    "emc": "/emc/submit",
}

MODULE_BLURBS = {
    "triage": "Assess triage priority from intake details and clinician notes.",
    "stroke": "Estimate stroke risk and generate educational care planning guidance.",
    "emc": "Prepare an electronic medical certificate for clinician review.",
}

AUDIENCE_PATIENT = "patient"
AUDIENCE_CLINICIAN = "clinician"

# Presentation only — these become CSS classes, never a clinical judgement.
TONE_CRITICAL = "critical"
TONE_WARNING = "warning"
TONE_OK = "ok"
TONE_NEUTRAL = "neutral"

_TRIAGE_TONES = {"P1": TONE_CRITICAL, "P2": TONE_WARNING, "P3": TONE_OK, "P4": TONE_OK}

_EMC_STATUS_LABELS = {
    "PENDING_REVIEW": "Awaiting clinician review",
    "APPROVED_FOR_ISSUE": "Approved for issue",
    "REJECTED": "Rejected by clinician",
    "BLOCKED_FINAL_SAFETY_REVIEW": "Blocked by safety review",
}

_EMC_STATUS_TONES = {
    "PENDING_REVIEW": TONE_NEUTRAL,
    "APPROVED_FOR_ISSUE": TONE_OK,
    "REJECTED": TONE_WARNING,
    "BLOCKED_FINAL_SAFETY_REVIEW": TONE_CRITICAL,
}

_STROKE_STATUS_LABELS = {
    "PENDING_REVIEW": "Awaiting clinician review",
    "APPROVED": "Reviewed by clinician",
    "REJECTED": "Rejected by clinician",
}

_STROKE_STATUS_TONES = {
    "PENDING_REVIEW": TONE_NEUTRAL,
    "APPROVED": TONE_OK,
    "REJECTED": TONE_WARNING,
}

# Stroke and triage inputs are flat dicts of raw feature names. These are the
# ones worth showing; anything unlisted falls back to a de-underscored title.
_INPUT_LABELS = {
    "Blood_Pressure_Systolic": "Systolic BP",
    "Blood_Pressure_Diastolic": "Diastolic BP",
    "Glucose_Level": "Glucose",
    "Physical_Activity": "Physically active",
    "Family_History": "Family history",
    "Heart_Disease": "Heart disease",
    "Alcohol_Intake": "Alcohol intake",
    "BMI": "BMI",
    "hr": "Heart rate",
    "sbp": "Systolic BP",
    "dbp": "Diastolic BP",
    "rr": "Respiratory rate",
    "o2": "Oxygen saturation",
    "temp": "Temperature",
    "arrival_mode": "Arrival mode",
}

_YES_NO_INPUTS = {
    "Smoking",
    "Diabetes",
    "Heart_Disease",
    "Family_History",
    "Physical_Activity",
}


def module_label(source_app: str) -> str:
    return MODULE_LABELS.get(source_app, str(source_app or "Unknown").title())


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _humanise(key: str) -> str:
    if key in _INPUT_LABELS:
        return _INPUT_LABELS[key]
    return str(key).replace("_", " ").strip().capitalize()


def _payload(record: Mapping, field: str) -> dict:
    value = record.get(field)
    return value if isinstance(value, dict) else {}


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _blank_tile(source_app: str, audience: str = AUDIENCE_CLINICIAN) -> dict:
    """The 'you haven't run this yet' state — still a full tile so the frontend
    renders one shape either way."""
    hrefs = PATIENT_MODULE_HREFS if audience == AUDIENCE_PATIENT else MODULE_HREFS
    return {
        "record_id": None,
        "module": source_app,
        "module_label": module_label(source_app),
        "blurb": MODULE_BLURBS.get(source_app, ""),
        "completed": False,
        "headline": "Not started",
        "tone": TONE_NEUTRAL,
        "facts": [],
        "inputs": [],
        "status": None,
        "created_at": None,
        "href": hrefs.get(source_app, "/"),
    }


def _summarize_triage(output: Mapping, status: str, audience: str) -> tuple[str, str, list]:
    # Triage priority, confidence, and red-flag state are clinician decision
    # support and stay withheld from a *clinician-run* assessment on a
    # patient's own tile. A patient's own self-check is different: it was
    # released to them directly (ctrse_core.patient_view,
    # released_without_clinician_review=True), so its own band belongs on
    # their tile — read back from the stored patient_view, never re-derived
    # here (this file computes no clinical fact of its own).
    if audience == AUDIENCE_PATIENT:
        status_upper = status.upper()
        if status_upper == "PATIENT_SELF_CHECK":
            patient_view = output.get("patient_view")
            urgency = patient_view.get("urgency") if isinstance(patient_view, Mapping) else None
            if isinstance(urgency, Mapping) and urgency.get("band_label"):
                # band_tone's values ("critical"/"warning"/"neutral") are the
                # same strings this file's own TONE_* constants use.
                tone = _text(urgency.get("band_tone")) or TONE_NEUTRAL
                return _text(urgency["band_label"]), tone, []
            return "Self-check completed", TONE_NEUTRAL, []
        if status_upper == "REFUSED":
            return "We could not complete this check", TONE_NEUTRAL, []
        if status_upper in ("ASSESSED", "APPROVED"):
            level = _text(output.get("predicted_level"))
            headline = core.PATIENT_LEVEL_HEADLINES.get(level)
            if headline:
                return headline, _TRIAGE_TONES.get(level, TONE_NEUTRAL), []
        return "Submitted for clinician review", TONE_NEUTRAL, []

    # A refused prediction is checked two ways on purpose: records written
    # before the status bug in api.py was fixed carry status="ASSESSED" even
    # though the model declined to run.
    if output.get("model_refused") or status.upper() == "REFUSED":
        reason = _text(output.get("refusal_reason")) or "Input was outside the model's validated range."
        return "Model declined to assess", TONE_NEUTRAL, [("Why", reason)]

    level = _text(output.get("predicted_level"))
    label = _text(output.get("level_label"))
    headline = f"{level} — {label}".strip(" —") or "Assessed"

    # A clinician must never mistake a patient's self-typed check for a
    # colleague's triage — this is the most important safety line in this
    # branch, so it goes first.
    facts: list[tuple[str, str]] = []
    if status.upper() == "PATIENT_SELF_CHECK":
        facts.append(("Source", "Patient self-check — self-reported, not clinician-verified"))
    if output.get("confidence_word"):
        facts.append(("Confidence", _text(output["confidence_word"])))
    if output.get("red_flag_triggered"):
        facts.append(("Red flag", _text(output.get("red_flag_complaint")) or "Triggered"))
    if audience == AUDIENCE_CLINICIAN:
        if output.get("escalation_basis"):
            facts.append(("Escalation basis", _text(output["escalation_basis"])))
        if output.get("threshold_context"):
            facts.append(("Threshold", _text(output["threshold_context"])))
        contributors = output.get("shap_top_contributors")
        if isinstance(contributors, list):
            for item in contributors[:3]:
                if isinstance(item, Mapping):
                    name = _text(item.get("feature") or item.get("name"))
                    if name:
                        facts.append(("Contributor", name))

    return headline, _TRIAGE_TONES.get(level, TONE_NEUTRAL), facts


def _summarize_stroke(output: Mapping, status: str, audience: str) -> tuple[str, str, list]:
    prediction = output.get("prediction")
    if not isinstance(prediction, Mapping):
        return "Details unavailable", TONE_NEUTRAL, []

    status_upper = status.upper()

    # Risk category and probability are withheld from the patient until a
    # clinician has approved the (possibly edited) result — this is the
    # human-in-the-loop review workflow, not a blanket withhold. ASSESSED and
    # CARE_PLAN_READY are the pre-review-workflow statuses a clinician-run
    # assessment still gets (apps/Jeslyn/app.py's persist_stroke_record) —
    # those were already produced directly by a clinician, so there is no
    # pending action left to complete; treat them as already reviewed.
    if audience == AUDIENCE_PATIENT:
        if status_upper in ("APPROVED", "ASSESSED", "CARE_PLAN_READY"):
            category = _text(prediction.get("risk_category")) or "Assessed"
            percent = prediction.get("risk_probability_percent")
            headline = f"{category} ({percent}%)" if percent is not None else category
            tone = TONE_WARNING if "high" in category.lower() else TONE_OK
            facts: list[tuple[str, str]] = []
            if output.get("care_plan"):
                facts.append(("Care plan", "Available"))
            if output.get("care_calendar"):
                facts.append(("7-day calendar", "Available"))
            return headline, tone, facts

        headline = _STROKE_STATUS_LABELS.get(status_upper, "Submitted")
        tone = _STROKE_STATUS_TONES.get(status_upper, TONE_NEUTRAL)
        return headline, tone, []

    category = _text(prediction.get("risk_category")) or "Assessed"
    percent = prediction.get("risk_probability_percent")
    headline = f"{category} ({percent}%)" if percent is not None else category
    tone = TONE_WARNING if "high" in category.lower() else TONE_OK

    facts: list[tuple[str, str]] = []
    if status_upper in _STROKE_STATUS_LABELS:
        facts.append(("Review status", _STROKE_STATUS_LABELS[status_upper]))
    if output.get("care_plan"):
        facts.append(("Care plan", "Generated"))
    if output.get("care_calendar"):
        facts.append(("7-day calendar", "Available"))
    if prediction.get("threshold_used") is not None:
        facts.append(("Decision threshold", _text(prediction["threshold_used"])))

    return headline, tone, facts


def _summarize_emc(output: Mapping, status: str, audience: str) -> tuple[str, str, list]:
    # Seeded records carry only `prediction`; live workflow records carry the
    # full snapshot including issue_status. Fall back to the record's status.
    issue_status = _text(output.get("issue_status")).upper() or status.upper()
    headline = _EMC_STATUS_LABELS.get(issue_status, issue_status.replace("_", " ").capitalize() or "In progress")
    tone = _EMC_STATUS_TONES.get(issue_status, TONE_NEUTRAL)

    facts: list[tuple[str, str]] = []
    metadata = output.get("metadata")
    if isinstance(metadata, Mapping):
        for key, label in (
            ("certificate_id", "Certificate ID"),
            ("authorized_medical_leave_days", "Leave days"),
            ("medical_leave_start_date", "Leave starts"),
        ):
            if metadata.get(key):
                facts.append((label, _text(metadata[key])))

    # EMC-005 / FORBIDDEN_TERMS in apps/YS/app.py: model scores, differential
    # diagnoses and review internals are clinician-only, never shown to the
    # patient the certificate is about.
    if audience != AUDIENCE_CLINICIAN:
        return headline, tone, facts

    prediction = output.get("prediction")
    if isinstance(prediction, Mapping):
        if prediction.get("primary_predicted_diagnosis"):
            facts.append(("Model suggestion", _text(prediction["primary_predicted_diagnosis"])))
        if prediction.get("prediction_confidence_percentage") is not None:
            facts.append(("Model confidence", f"{_text(prediction['prediction_confidence_percentage'])}%"))
        alternatives = prediction.get("clinical_differential_alternatives")
        if isinstance(alternatives, list):
            named = [
                _text(item.get("alternative_condition"))
                for item in alternatives
                if isinstance(item, Mapping) and item.get("alternative_condition")
            ]
            if named:
                facts.append(("Differentials", ", ".join(named[:2])))

    gate = output.get("review_gate")
    if isinstance(gate, Mapping):
        verdict = gate.get("verdict") or gate.get("decision")
        if verdict:
            facts.append(("Safety review", _text(verdict)))

    return headline, tone, facts


def describe_inputs(record: Mapping, *, audience: str = AUDIENCE_CLINICIAN, limit: int = 12) -> list:
    """The fields that produced this result, as (label, value) pairs."""
    try:
        source_app = record.get("source_app")
        payload = _payload(record, "input_payload")
        if not payload:
            return []

        if source_app == "emc":
            return _describe_emc_inputs(payload, limit)

        pairs: list[tuple[str, str]] = []
        for key, value in payload.items():
            if value is None or value == "":
                continue
            if key == "vitals" and isinstance(value, Mapping):
                for vital_key, vital_value in value.items():
                    if vital_value not in (None, "") and not vital_key.endswith("_unit"):
                        pairs.append((_humanise(vital_key), _text(vital_value)))
                continue
            if key == "complaints" and isinstance(value, list):
                tokens = [
                    _text(item.get("token")) if isinstance(item, Mapping) else _text(item)
                    for item in value
                ]
                tokens = [token for token in tokens if token]
                if audience == AUDIENCE_PATIENT and source_app == "triage":
                    # Raw cc_ tokens (e.g. "cardiacarrest") are an internal
                    # vocabulary, not patient-facing copy — map through the
                    # same label table patient_view() itself uses, so a
                    # patient's own "what I entered" list never shows one.
                    tokens = [core.PATIENT_COMPLAINT_LABELS.get(token, "A reported concern")
                              for token in tokens]
                if tokens:
                    pairs.append(("Complaints", ", ".join(tokens)))
                continue
            if key in _YES_NO_INPUTS:
                pairs.append((_humanise(key), "Yes" if value in (1, "1", True) else "No"))
                continue
            pairs.append((_humanise(key), _text(value)))

        return pairs[:limit]
    except Exception:  # never let a bad payload break a read-only view
        return []


def _describe_emc_inputs(payload: Mapping, limit: int) -> list:
    """Metadata plus only the symptoms that were actually reported.

    `features` is ~132 zero/one columns; listing them all would bury the few
    that matter, so this mirrors internal_summary() in apps/YS/app.py and shows
    the positives only.
    """
    pairs: list[tuple[str, str]] = []
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key, label in (
            ("patient_name", "Patient"),
            ("patient_age", "Age"),
            ("consultation_date", "Consultation"),
            ("authorized_medical_leave_days", "Leave days"),
            ("attending_clinician_name", "Clinician"),
            ("clinic_name", "Clinic"),
        ):
            if metadata.get(key) not in (None, ""):
                pairs.append((label, _text(metadata[key])))

    features = payload.get("features")
    if isinstance(features, Mapping):
        reported = [
            _humanise(key)
            for key, value in features.items()
            if value in (1, "1", True) and key not in {"Age", "Gender", "Duration", "Medical_History"}
        ]
        if reported:
            pairs.append(("Reported symptoms", ", ".join(reported[:8])))
        if features.get("Duration"):
            pairs.append(("Symptom duration", f"{_text(features['Duration'])} days"))

    return pairs[:limit]


def summarize(record: Mapping | None, *, audience: str = AUDIENCE_CLINICIAN) -> dict:
    """One record -> one dashboard tile. Pass None for a module never run."""
    if not record:
        return _blank_tile("")

    source_app = str(record.get("source_app") or "")
    tile = _blank_tile(source_app, audience=audience)
    tile.update({
        "record_id": record.get("id"),
        "completed": True,
        "status": record.get("status"),
        "created_at": _iso(record.get("created_at")),
    })

    try:
        output = _payload(record, "output_payload")
        status = str(record.get("status") or "")

        if source_app == "triage":
            headline, tone, facts = _summarize_triage(output, status, audience)
        elif source_app == "stroke":
            headline, tone, facts = _summarize_stroke(output, status, audience)
        elif source_app == "emc":
            headline, tone, facts = _summarize_emc(output, status, audience)
        else:
            headline, tone, facts = "Recorded", TONE_NEUTRAL, []

        tile["headline"] = headline
        tile["tone"] = tone
        tile["facts"] = [{"label": label, "value": value} for label, value in facts]
        tile["inputs"] = [
            {"label": label, "value": value}
            for label, value in describe_inputs(record, audience=audience)
        ]
    except Exception:
        tile["headline"] = "Details unavailable"
        tile["tone"] = TONE_NEUTRAL

    return tile


def blank_tile(source_app: str, *, audience: str = AUDIENCE_CLINICIAN) -> dict:
    return _blank_tile(source_app, audience=audience)


def iso_timestamp(value: Any) -> str | None:
    """JSON-safe timestamp for anything the store hands back."""
    return _iso(value)
