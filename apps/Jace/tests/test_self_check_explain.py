"""tests/test_self_check_explain.py — POST /api/self-check/explain (§Patient guidance, use case C).

This is purely supplementary content generated AFTER a self-check result is already final. The
one property every test here protects: nothing an LLM produces reaches the patient response body
unless core.generate()'s guardrail_check() passed it — a guardrail failure (or an offline/empty
generation) must degrade to an "unavailable" body, never to showing the flagged text with a
warning the way the clinician confirm screen does, because there is no clinician here to review it.

Also covers the redaction-bypass regression: the suggestion payload's "note" field must be
extraction.note_used (prepared/redacted), never raw browser text — and that the confirm-screen
extras (history/medications/allergies/onset/confirmed_complaint_tokens) reach the payload
core.generate() sees, since §Patient guidance is grounded in everything the patient supplied.

Run:  python -m pytest tests/test_self_check_explain.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import api
import ctrse_core as core

client = TestClient(api.app)

_PATIENT_VIEW = {
    "predicted_level": "P3",
    "level_label": "URGENT",
    "age": 40,
    "arrival_mode": "Car",
    "reported_concerns": ["Chest pain or tightness"],
    "recorded_vitals": {},
    "vitals_not_recorded": [],
    "red_flag_triggered": False,
    "urgency": {"band": "URGENT_TODAY", "band_label": "See a clinician urgently today",
                "action_line": "Please see a clinician today.", "reasons": []},
    "confidence_word": "Moderate",
    "disclaimer": core.DISCLAIMER,
}

_EXTRACTION = {
    "note_used": "redacted chest pain description",
    "history_mentions": [{"text": "history of migraines", "span": "history of migraines"}],
    "medications": [{"text": "ibuprofen", "span": "ibuprofen"}],
    "allergies": [{"value": "penicillin", "span": "penicillin allergy"}],
    "onset": [{"complaint": "chestpain", "value": "30 minutes", "span": "30 minutes"}],
    "pain_score": {"value": 6, "span": "pain 6/10"},
}


def test_explain_returns_available_guidance_on_guardrail_pass(monkeypatch):
    monkeypatch.setattr(core, "generate", lambda payload, use_case, pinned=None: {
        "source": "live",
        "text": "You told us about chest pain, so see a clinician today. Watch for worsening pain.",
        "guardrails": {"passed": True, "flags": []}, "disclaimer": core.DISCLAIMER,
    })

    response = client.post("/api/self-check/explain", json={
        "patient_view": _PATIENT_VIEW, "extraction": _EXTRACTION,
    })

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["text"] == ("You told us about chest pain, so see a clinician today. "
                             "Watch for worsening pain.")
    assert "watch_for" not in body


def test_explain_hides_guidance_when_guardrail_fails(monkeypatch):
    flagged_text = "This is probably fine and nothing serious."
    monkeypatch.setattr(core, "generate", lambda payload, use_case, pinned=None: {
        "source": "live", "text": flagged_text,
        "guardrails": {"passed": False, "flags": ["reassuring language: 'probably fine'"]},
        "disclaimer": core.DISCLAIMER,
    })

    response = client.post("/api/self-check/explain", json={"patient_view": _PATIENT_VIEW})

    assert response.status_code == 503
    body = response.json()
    assert body["available"] is False
    # The one property that matters most here: a guardrail-rejected generation must never
    # surface in the response body, flagged or not — unlike the clinician confirm screen, there
    # is no one downstream to review it.
    assert flagged_text not in response.text


def test_explain_hides_guidance_when_offline_or_unparsed(monkeypatch):
    monkeypatch.setattr(core, "generate", lambda payload, use_case, pinned=None: {
        "source": "offline", "text": "",
        "guardrails": {"passed": None, "flags": ["offline — no live model this session"]},
        "disclaimer": core.DISCLAIMER,
    })

    response = client.post("/api/self-check/explain", json={"patient_view": _PATIENT_VIEW})

    assert response.status_code == 503
    assert response.json()["available"] is False


def test_explain_never_sends_the_clinician_payload_or_a_stray_record_id(monkeypatch):
    seen = {}

    def fake_generate(payload, use_case, pinned=None):
        seen["payload"] = payload
        seen["use_case"] = use_case
        return {"source": "live", "text": "ok",
                "guardrails": {"passed": True, "flags": []}, "disclaimer": core.DISCLAIMER}

    monkeypatch.setattr(core, "generate", fake_generate)

    # record_id and confirmed_complaint_tokens are what post_self_check merges into its response
    # body — the client's `result` (and therefore what it sends back here) carries them, but
    # record_id must never reach core.generate(), and confirmed_complaint_tokens must be pulled
    # OUT into its own payload key, not left sitting inside the patient_view fields.
    sent = {**_PATIENT_VIEW, "record_id": "abc-123", "confirmed_complaint_tokens": ["chestpain"]}
    response = client.post("/api/self-check/explain", json={"patient_view": sent})

    assert response.status_code == 200
    assert seen["use_case"] == "patient_guidance"
    assert "record_id" not in seen["payload"]
    assert seen["payload"]["confirmed_complaint_tokens"] == ["chestpain"]
    # Only patient_view fields were ever in play — no clinician-only key could have leaked in,
    # since none were sent in the first place.
    assert "probabilities" not in seen["payload"]
    assert "escalation_basis" not in seen["payload"]


def test_explain_grounds_in_note_used_never_raw_browser_text(monkeypatch):
    """Redaction-bypass regression: the suggestion payload's "note" must be
    extraction.note_used (already prepared/redacted by core.extract_from_note), never anything
    the browser could have sent unredacted."""
    seen = {}

    def fake_generate(payload, use_case, pinned=None):
        seen["payload"] = payload
        return {"source": "live", "text": "ok",
                "guardrails": {"passed": True, "flags": []}, "disclaimer": core.DISCLAIMER}

    monkeypatch.setattr(core, "generate", fake_generate)

    response = client.post("/api/self-check/explain", json={
        "patient_view": _PATIENT_VIEW, "extraction": _EXTRACTION,
    })

    assert response.status_code == 200
    assert seen["payload"]["note"] == _EXTRACTION["note_used"]


def test_explain_folds_confirm_screen_extras_into_the_payload(monkeypatch):
    """§Patient guidance is grounded in everything the patient supplied — history, medications,
    allergies, onset, pain_score — not just the bare patient_view result."""
    seen = {}

    def fake_generate(payload, use_case, pinned=None):
        seen["payload"] = payload
        return {"source": "live", "text": "ok",
                "guardrails": {"passed": True, "flags": []}, "disclaimer": core.DISCLAIMER}

    monkeypatch.setattr(core, "generate", fake_generate)

    client.post("/api/self-check/explain", json={
        "patient_view": _PATIENT_VIEW, "extraction": _EXTRACTION,
    })

    for field in ("history_mentions", "medications", "allergies", "onset", "pain_score"):
        assert seen["payload"][field] == _EXTRACTION[field]


def test_explain_requires_a_non_empty_patient_view():
    response = client.post("/api/self-check/explain", json={"patient_view": {}})
    assert response.status_code == 400
