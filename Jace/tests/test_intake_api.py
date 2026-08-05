"""test_intake_api.py — Phase 4 additive endpoints: /api/vocab, /api/log-correction,
and the /api/explain inline-payload path (a note-derived patient has no id).

All tests import api -> one heavy core.init at module load (same as test_api.py).
Run: python -m pytest tests/test_intake_api.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import ctrse_core as core
import api

client = TestClient(api.app)


def test_vocab_shape_and_membership():
    r = client.get("/api/vocab")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"complaints", "arrival_modes"}
    # complaints == the extractor's allowed-emit set, prevalence-sorted (no hardcoding)
    assert set(body["complaints"]) == set(core._ALLOWED_EMIT)
    assert body["complaints"][0] == max(core._ALLOWED_EMIT,
                                        key=lambda t: core._EMIT_PREVALENCE.get(t, 0))
    assert body["arrival_modes"] == list(core._ARRIVAL_ENUM)
    # conditioned forms are NOT offered in the dropdowns
    assert "fall>65" not in body["complaints"]


def test_log_correction_appends_jsonl():
    # snapshot + restore: the audit log is a real demo artefact — tests must not pollute it
    try:
        with open(api._CORRECTION_LOG, encoding="utf-8") as f:
            before = f.read()
    except FileNotFoundError:
        before = None
    try:
        rec = {"note": "68yo test note", "extracted": {"age": 68, "complaints": ["chestpain"]},
               "corrected": {"age": 68, "complaints": ["chesttightness"]},
               "prediction": "P2", "timestamp": "2026-07-14T00:00:00Z"}
        r1 = client.post("/api/log-correction", json=rec)
        assert r1.status_code == 200 and r1.json()["logged"] is True
        c1 = r1.json()["count"]
        r2 = client.post("/api/log-correction", json=rec)
        assert r2.json()["count"] == c1 + 1
        # file exists and its last line round-trips with a server timestamp added
        with open(api._CORRECTION_LOG, encoding="utf-8") as f:
            last = json.loads(f.readlines()[-1])
        assert last["note"] == rec["note"]
        assert last["corrected"]["complaints"] == ["chesttightness"]
        assert "logged_at" in last
    finally:
        if before is None:
            os.remove(api._CORRECTION_LOG)
        else:
            with open(api._CORRECTION_LOG, "w", encoding="utf-8") as f:
                f.write(before)


def test_explain_inline_payload_offline_never_500():
    # a real payload straight from the sample -> the inline path must work without an id
    payload = api._PATIENTS[0]["payload"]
    r = client.post("/api/explain",
                    json={"payload": payload, "use_case": "justify", "prefer_live": False})
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] is None
    assert body["source"] in ("live", "pinned", "offline")
    assert "text" in body
    assert body["disclaimer"] == core.DISCLAIMER


def test_explain_requires_id_or_payload():
    r = client.post("/api/explain", json={"use_case": "justify", "prefer_live": False})
    assert r.status_code == 400
    assert r.json() == {"error": "patient_id or payload required"}


def test_explain_by_id_path_unchanged():
    r = client.post("/api/explain",
                    json={"patient_id": "clear_p1", "use_case": "justify", "prefer_live": False})
    assert r.status_code == 200
    assert r.json()["patient_id"] == "clear_p1"


def test_predict_response_carries_payload_for_explain():
    # the frontend chains predict -> explain via this key; it must be the 16-key payload
    r = client.post("/api/predict", json={
        "age": 68, "sex": "Female", "arrival_mode": "car",
        "complaints": [{"token": "chestpain"}], "vitals": {"hr": 104}})
    assert r.status_code == 200
    body = r.json()
    assert "payload" in body and isinstance(body["payload"], dict)
    assert body["payload"]["predicted_level"] == body["predicted_level"]
    # and the inline-payload explain accepts it directly
    r2 = client.post("/api/explain",
                     json={"payload": body["payload"], "use_case": "handover", "prefer_live": False})
    assert r2.status_code == 200
    assert "assessment" in r2.json()


# ===========================================================================
# Phase 5 — pinned extraction fallbacks (offline demo resilience)
# ===========================================================================

import re

import pytest


def _appjs_seed_notes():
    """The seed notes exactly as static/app.js ships them (drift guard). Allows a
    backslash-escaped apostrophe inside the single-quoted JS string literal."""
    js = open(os.path.join(os.path.dirname(api.__file__), "static", "app.js"),
              encoding="utf-8").read()
    block = js[js.index("seeds: ["):js.index("]", js.index("seeds: ["))]
    raw = re.findall(r"note: '((?:[^'\\]|\\.)+)'", block)
    return [n.replace("\\'", "'") for n in raw]


def test_pinned_extractions_cover_all_appjs_seeds():
    notes = _appjs_seed_notes()
    assert len(notes) == 10, "expected the 10 demo seeds in app.js"
    for note in notes:
        fp = core.note_fingerprint(note)
        assert fp in api._PINNED_EXTRACTIONS, f"seed not pinned: {note[:40]}…"
        entry = api._PINNED_EXTRACTIONS[fp]
        assert entry["extraction"].get("note_used"), "pinned extraction missing note_used"


def test_extract_pinned_fallback_when_offline(monkeypatch):
    monkeypatch.setattr(core, "GEMINI_AVAILABLE", False)
    note = _appjs_seed_notes()[0]                      # chest-pain seed
    r = client.post("/api/extract", json={"note": note})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "pinned"
    assert body["note_used"] and body["complaints"]
    # paediatric seed: the pinned refusal survives the fallback path
    paed = next(n for n in _appjs_seed_notes() if "6yo" in n)
    rp = client.post("/api/extract", json={"note": paed}).json()
    assert rp["source"] == "pinned" and rp["model_refused"] is True


def test_extract_adhoc_note_offline_honest_unavailable(monkeypatch):
    monkeypatch.setattr(core, "GEMINI_AVAILABLE", False)
    r = client.post("/api/extract", json={"note": "55yo woman, ad-hoc note typed by the assessor"})
    assert r.status_code == 503                        # structured, never a 500
    body = r.json()
    assert body["error"] == "extraction_unavailable"
    assert "no pinned demo result" in body["detail"]
    assert body["pinned_available"] is False


def test_extract_live_carries_source_field():
    if not core.GEMINI_AVAILABLE:
        pytest.skip("no key — live source check needs Gemini")
    r = client.post("/api/extract", json={"note": "45yo male, walk-in, sore throat 3 days"})
    assert r.status_code == 200 and r.json()["source"] == "live"


def test_explain_pinned_payload_offline(monkeypatch):
    monkeypatch.setattr(core, "GEMINI_AVAILABLE", False)
    recs = api._PINNED_GEN
    assert recs, "no pinned gen records — run prep_pinned_extractions.py"
    ra = client.post("/api/explain", json={"payload": recs[0]["payload"],
                                           "use_case": "justify", "prefer_live": True})
    assert ra.status_code == 200 and ra.json()["source"] == "pinned"
    rb = client.post("/api/explain", json={"payload": recs[0]["payload"],
                                           "use_case": "handover", "prefer_live": True})
    assert rb.status_code == 200 and rb.json()["source"] == "pinned"
    assert "assessment" in rb.json()


def test_extract_guardrail_backstop():
    # the REAL extraction guardrail on the planted "LLM obeyed the injection" payload
    r = client.post("/api/extract-guardrail-test")
    assert r.status_code == 200
    body = r.json()
    assert "cardiacarrest" in body["candidate_complaints"]
    assert body["kept_complaints"] == ["chestpain"]
    assert body["dropped_complaints"] == ["cardiacarrest"]
    assert body["red_flags"] == []                      # the floor never sees the injected token
    assert any(f.startswith("injection_span_dropped:") for f in body["flags"])
    assert "synthetic" in body and "synthetic" in body["synthetic"]
