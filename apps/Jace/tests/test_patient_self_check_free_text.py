"""tests/test_patient_self_check_free_text.py — POST /api/self-check/extract and
POST /api/self-check under the confirm-screen contract (§Unify the patient intake
interface with the clinician's).

The patient confirm screen is now the SAME interface the clinician uses: every
field, including complaints, is directly editable against the full controlled
vocabulary — there is no separate patient-safe picker (api.py's
get_self_check_options() serves the identical vocabulary GET /api/vocab does for
the clinician). What these tests protect is what's specific to the patient path:
the two-step extract-then-confirm contract, the `other`-only rejection, and — the
one constraint full editability doesn't get to override — that a red flag the
extractor found cannot be silently dropped by editing the confirmed complaint
list (see post_self_check's own comment in api.py).

Run:  python -m pytest tests/test_patient_self_check_free_text.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import api
import ctrse_core as core

client = TestClient(api.app)


def _extraction(*, complaints=None, red_flags=None):
    return {
        "age": {"value": 53, "span": "53yo"},
        "sex": {"value": "Female", "span": "woman"},
        "arrival_mode": {"value": "walk_in", "span": "walk-in", "ambiguous": False,
                         "alternates": [], "reason": None},
        "complaints": complaints if complaints is not None else [],
        "onset": [], "history_mentions": [], "medications": [], "allergies": [],
        "pain_score": None, "red_flags": red_flags or [], "unmapped": [], "guardrail_flags": [],
        "model_refused": False, "note_used": "prepared symptom description", "source": "live",
    }


def _prediction(fields):
    tokens = {item["token"] for item in fields["complaints"]}
    return {
        "predicted_level": "P1" if "cardiacarrest" in tokens else "P3",
        "level_label": "Urgent", "level_colour": "#000",
        "age": fields["age"], "arrival_mode": fields["arrival_mode"],
        "active_chief_complaints": [item["token"] for item in fields["complaints"]],
        "triage_vitals": fields["vitals"], "vitals_not_recorded": [],
        "red_flag_triggered": "cardiacarrest" in tokens,
        "abnormal_vitals": [], "confidence_word": "Moderate", "model_refused": False,
    }


def _avoid_persistence(monkeypatch):
    monkeypatch.setattr(api.SHARED_STORE, "safe_create_record", lambda **_: "free-text-test")
    monkeypatch.setattr(api.SHARED_STORE, "safe_append_audit_event", lambda **_: None)


# ---------------------------------------------------------------------------
# POST /api/self-check/extract — the same guarded front door as /api/extract
# ---------------------------------------------------------------------------

def test_self_check_extract_uses_the_same_guarded_extractor(monkeypatch):
    seen = {}

    def fake_extract(note, pinned):
        seen["note"] = note
        return _extraction(complaints=[{"token": "chestpain", "span": "chest pressure"}])

    monkeypatch.setattr(core, "extract_from_note", fake_extract)

    response = client.post("/api/self-check/extract", json={"note": "I have chest pressure"})

    assert response.status_code == 200
    assert seen["note"] == "I have chest pressure"
    assert response.json()["complaints"][0]["token"] == "chestpain"


def test_self_check_extract_unavailable_is_honest_and_never_scores(monkeypatch):
    monkeypatch.setattr(core, "extract_from_note", lambda note, pinned: {
        "error": "extraction_unavailable",
    })

    response = client.post("/api/self-check/extract", json={"note": "Chest pain"})

    assert response.status_code == 503
    assert response.json()["error"] == "extraction_unavailable"


def test_self_check_extract_rejects_an_empty_note():
    response = client.post("/api/self-check/extract", json={"note": "   "})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/self-check — confirmed fields + the extraction they were built from
# ---------------------------------------------------------------------------

def test_self_check_scores_the_confirmed_complaints(monkeypatch):
    seen = {}

    def predict(fields):
        seen.update(fields)
        return _prediction(fields)

    monkeypatch.setattr(core, "predict_from_fields", predict)
    _avoid_persistence(monkeypatch)

    response = client.post("/api/self-check", json={
        "age": 53, "sex": "Female", "arrival_mode": "walk_in",
        "complaints": [{"token": "chestpain", "evidence": None}],
        "vitals": {},
        "extraction": _extraction(complaints=[{"token": "chestpain"}]),
    })

    assert response.status_code == 200
    assert seen["complaints"] == [{"token": "chestpain", "evidence": None}]
    body = response.json()
    assert body["reported_concerns"] == ["Chest pain or tightness"]
    assert "chestpain" not in str(body)


def test_self_check_rejects_an_other_only_submission(monkeypatch):
    monkeypatch.setattr(core, "predict_from_fields",
                        lambda fields: (_ for _ in ()).throw(AssertionError("must not score fallback")))

    response = client.post("/api/self-check", json={
        "age": 53, "complaints": [{"token": "other", "evidence": None}], "vitals": {},
        "extraction": _extraction(complaints=[{"token": "other", "fallback": True}]),
    })

    assert response.status_code == 422
    assert "could not identify a symptom" in response.json()["detail"]


def test_self_check_never_scores_an_empty_submission_with_no_red_flag(monkeypatch):
    monkeypatch.setattr(core, "predict_from_fields",
                        lambda fields: (_ for _ in ()).throw(AssertionError("must not score an empty submission")))

    response = client.post("/api/self-check", json={
        "age": 53, "complaints": [], "vitals": {}, "extraction": _extraction(),
    })

    assert response.status_code == 422


def test_self_check_cannot_drop_a_red_flag_the_extractor_found(monkeypatch):
    """The one constraint full editability doesn't get to override: the confirm screen is
    fully editable, but editing a red flag out of the complaint list must not silently
    drop it from what gets scored — patient_urgency_band's max-lattice can then never see
    it and could produce a de-escalated, falsely reassuring band."""
    seen = {}

    def predict(fields):
        seen.update(fields)
        return _prediction(fields)

    monkeypatch.setattr(core, "predict_from_fields", predict)
    _avoid_persistence(monkeypatch)

    response = client.post("/api/self-check", json={
        "age": 53,
        # The patient edited the confirm screen down to a single, calmer complaint — the
        # extractor originally found `cardiacarrest` too (see `extraction` below).
        "complaints": [{"token": "chestpain", "evidence": None}],
        "vitals": {},
        "extraction": _extraction(
            complaints=[{"token": "chestpain"}, {"token": "cardiacarrest"}],
            red_flags=["cardiacarrest"],
        ),
    })

    assert response.status_code == 200
    tokens = {c["token"] for c in seen["complaints"]}
    assert "cardiacarrest" in tokens
    assert "chestpain" in tokens
    # End-to-end proof, not just that the token reached predict_from_fields: the real
    # patient_urgency_band still escalates on it.
    body = response.json()
    assert body["urgency"]["band"] == "EMERGENCY_NOW"


def test_self_check_red_flag_union_can_rescue_an_other_only_submission(monkeypatch):
    """The `other`-only rejection runs AFTER the red-flag union, so a red flag the
    extractor found still gets scored even if every complaint the patient confirmed
    resolved to `other`."""
    seen = {}

    def predict(fields):
        seen.update(fields)
        return _prediction(fields)

    monkeypatch.setattr(core, "predict_from_fields", predict)
    _avoid_persistence(monkeypatch)

    response = client.post("/api/self-check", json={
        "age": 53, "complaints": [{"token": "other", "evidence": None}], "vitals": {},
        "extraction": _extraction(
            complaints=[{"token": "other", "fallback": True}, {"token": "cardiacarrest"}],
            red_flags=["cardiacarrest"],
        ),
    })

    assert response.status_code == 200
    assert "cardiacarrest" in {c["token"] for c in seen["complaints"]}
