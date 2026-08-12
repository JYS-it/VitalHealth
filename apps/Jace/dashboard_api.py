"""Cross-module dashboard reads for the VitalHealth portal.

Kept out of api.py deliberately. That file is transport for CTRSE and only
CTRSE — ctrse_core.py owns every clinical fact it serves (see CLAUDE.md). This
router serves no clinical logic at all: it reads records the three apps have
already written and hands them to the SPA as pre-formatted tiles. Mixing the
two in one module would blur a boundary the app has been careful about.

Authorisation happens here, on the server, against the signed session cookie.
The SPA also branches on role, but that is presentation only — a patient who
edits their own JavaScript still cannot read another patient's chart.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from vitalhealth_storage import get_store, identity
from vitalhealth_storage.store import SOURCE_APPS

import dashboard_summaries as summaries

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

STORE = get_store()

_UNAVAILABLE = {
    "error": "records_unavailable",
    "detail": "Saved records could not be read right now. This is a display problem, not a sign that there are no records.",
}


def _actor(request: Request) -> identity.Actor | None:
    """The proxy forwards the browser's cookies untouched, so the signed
    session is readable here without asking the gateway anything."""
    return identity.actor_from_cookies(request.cookies)


def _record_view(record, *, audience: str) -> dict:
    tile = summaries.summarize(record, audience=audience)
    tile["patient_id"] = record.get("patient_id")
    tile["source_app"] = record.get("source_app")
    return tile


def _patient_row_view(row: dict, *, audience: str) -> dict:
    return {
        "patient_id": row["patient_id"],
        "external_id": row["external_id"],
        "display_name": row["display_name"] or row["external_id"],
        "completed_modules": row["completed_modules"],
        "record_count": row["record_count"],
        "last_activity": summaries.iso_timestamp(row["last_activity"]),
        "modules": {
            app: summaries.summarize(row["modules"].get(app), audience=audience)
            if row["modules"].get(app)
            else summaries.blank_tile(app, audience=audience)
            for app in SOURCE_APPS
        },
    }


@router.get("/me")
def dashboard_me(request: Request):
    """Who the SPA should render for. Mirrors the gateway's /api/me, but
    answered locally so the dashboard has an identity even if the gateway's
    endpoint is unreachable."""
    actor = _actor(request)

    if actor is None:
        return {"authenticated": False, "role": None, "display_name": "guest"}

    return {
        "authenticated": True,
        "role": actor.role,
        "email": actor.email,
        "display_name": actor.display_name or actor.email or "user",
        "patient_external_id": actor.patient_external_id,
        "storage_enabled": STORE.enabled,
    }


@router.get("/summary")
def patient_summary(request: Request):
    """The patient's own dashboard: one tile per module, plus their history."""
    actor = _actor(request)

    if actor is None:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    if not STORE.enabled:
        return JSONResponse(_UNAVAILABLE, status_code=503)

    patient_id = None
    patient = None

    try:
        if actor.patient_external_id:
            patient = STORE.get_patient_by_external_id(actor.patient_external_id)
            patient_id = patient["id"] if patient else None

        # patient_id and owner_user_id are OR-ed by the store: a record is
        # "mine" whether it is about me or was created by me.
        latest = STORE.latest_record_per_app(
            patient_id=patient_id,
            owner_user_id=actor.user_id,
        )
        history = STORE.list_records(
            patient_id=patient_id,
            owner_user_id=actor.user_id,
            limit=25,
        )
    except SQLAlchemyError:
        LOGGER.warning("Dashboard summary read failed", exc_info=True)
        return JSONResponse(_UNAVAILABLE, status_code=503)

    audience = summaries.AUDIENCE_PATIENT if actor.is_patient else summaries.AUDIENCE_CLINICIAN

    modules = {
        app: summaries.summarize(latest.get(app), audience=audience)
        if latest.get(app)
        else summaries.blank_tile(app, audience=audience)
        for app in SOURCE_APPS
    }

    return {
        "role": actor.role,
        "display_name": actor.display_name or actor.email,
        "patient": {
            "external_id": actor.patient_external_id,
            "display_name": (patient or {}).get("display_name") or actor.display_name,
        },
        "modules": [modules[app] for app in SOURCE_APPS],
        "completed_count": sum(1 for tile in modules.values() if tile["completed"]),
        "total_count": len(SOURCE_APPS),
        "history": [_record_view(record, audience=audience) for record in history],
    }


@router.get("/patients")
def clinician_patients(request: Request, q: str | None = None):
    """Every patient, for the clinician dashboard."""
    actor = _actor(request)

    if actor is None:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    if not actor.is_clinician:
        return JSONResponse({"detail": "Clinician access required"}, status_code=403)

    if not STORE.enabled:
        return JSONResponse(_UNAVAILABLE, status_code=503)

    try:
        rows = STORE.list_patient_summaries(search=q)
        unassigned = STORE.list_unassigned_records(limit=25)
    except SQLAlchemyError:
        LOGGER.warning("Clinician patient list read failed", exc_info=True)
        return JSONResponse(_UNAVAILABLE, status_code=503)

    audience = summaries.AUDIENCE_CLINICIAN
    active_subject, _ = identity.resolve_subject(request.cookies, actor)

    return {
        "role": actor.role,
        "active_subject": active_subject,
        "patients": [_patient_row_view(row, audience=audience) for row in rows],
        # Triage and stroke have no patient field on their forms, so anything
        # run without a selected subject lands here instead of disappearing.
        "unassigned": [_record_view(record, audience=audience) for record in unassigned],
    }


@router.get("/pending")
def clinician_pending(request: Request):
    """Every stroke/EMC item awaiting clinician review, across all patients.

    Open shared queue: no per-item claiming, any clinician sees the same
    list, and a row simply drops off once its status moves past
    PENDING_REVIEW. Triage is excluded — it stays instant and clinician-only,
    out of scope for this queue.
    """
    actor = _actor(request)

    if actor is None:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    if not actor.is_clinician:
        return JSONResponse({"detail": "Clinician access required"}, status_code=403)

    if not STORE.enabled:
        return JSONResponse(_UNAVAILABLE, status_code=503)

    try:
        pending = STORE.list_pending_review()
    except SQLAlchemyError:
        LOGGER.warning("Pending review queue read failed", exc_info=True)
        return JSONResponse(_UNAVAILABLE, status_code=503)

    review_paths = {"stroke": "/stroke/review", "emc": "/emc/review"}
    items = []
    for record in pending:
        tile = summaries.summarize(record, audience=summaries.AUDIENCE_CLINICIAN)
        tile["patient_id"] = record.get("patient_id")
        tile["patient_display_name"] = record.get("patient_display_name") or record.get("patient_external_id")
        review_path = review_paths.get(record.get("source_app"))
        if review_path:
            tile["href"] = f"{review_path}/{record['id']}"
        items.append(tile)

    return {"pending": items, "count": len(items)}


@router.get("/patients/{patient_id}")
def clinician_patient_detail(patient_id: str, request: Request):
    actor = _actor(request)

    if actor is None:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    if not actor.is_clinician:
        return JSONResponse({"detail": "Clinician access required"}, status_code=403)

    if not STORE.enabled:
        return JSONResponse(_UNAVAILABLE, status_code=503)

    try:
        patient = STORE.get_patient(patient_id)
        if patient is None:
            return JSONResponse({"detail": "Unknown patient"}, status_code=404)

        records = STORE.list_records(patient_id=patient_id, limit=100)
        latest = STORE.latest_record_per_app(patient_id=patient_id)
    except SQLAlchemyError:
        LOGGER.warning("Clinician patient detail read failed", exc_info=True)
        return JSONResponse(_UNAVAILABLE, status_code=503)

    audience = summaries.AUDIENCE_CLINICIAN

    return {
        "patient": {
            "patient_id": patient["id"],
            "external_id": patient["external_id"],
            "display_name": patient["display_name"] or patient["external_id"],
        },
        "modules": [
            summaries.summarize(latest.get(app), audience=audience)
            if latest.get(app)
            else summaries.blank_tile(app, audience=audience)
            for app in SOURCE_APPS
        ],
        "history": [_record_view(record, audience=audience) for record in records],
    }


@router.get("/records/{record_id}")
def record_detail(record_id: str, request: Request):
    actor = _actor(request)

    if actor is None:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    if not STORE.enabled:
        return JSONResponse(_UNAVAILABLE, status_code=503)

    try:
        record = STORE.get_record(record_id)
        if record is None:
            return JSONResponse({"detail": "Unknown record"}, status_code=404)

        if not actor.is_clinician and not _patient_may_read(actor, record):
            # 404 rather than 403: a patient probing record ids should not be
            # able to learn which ones exist.
            return JSONResponse({"detail": "Unknown record"}, status_code=404)

        audit = STORE.list_audit_events(record_id)
    except SQLAlchemyError:
        LOGGER.warning("Record detail read failed", exc_info=True)
        return JSONResponse(_UNAVAILABLE, status_code=503)

    audience = summaries.AUDIENCE_CLINICIAN if actor.is_clinician else summaries.AUDIENCE_PATIENT

    audit_view = [
        {
            "event_type": event["event_type"],
            "actor_reference": event["actor_reference"],
            "occurred_at": summaries.iso_timestamp(event["occurred_at"]),
        }
        for event in audit
    ]
    if not actor.is_clinician:
        # A patient's history may show that a record changed, but never the
        # clinician's email or other staff identity metadata.
        audit_view = [
            {
                "event_type": event["event_type"],
                "occurred_at": event["occurred_at"],
            }
            for event in audit_view
        ]

    return {
        "record": _record_view(record, audience=audience),
        "audit": audit_view,
    }


def _patient_may_read(actor: identity.Actor, record) -> bool:
    if record.get("owner_user_id") and record["owner_user_id"] == actor.user_id:
        return True

    if not record.get("patient_id") or not actor.patient_external_id:
        return False

    patient = STORE.get_patient(record["patient_id"])
    return bool(patient and patient["external_id"] == actor.patient_external_id)
