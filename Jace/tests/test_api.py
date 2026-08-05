"""test_api.py — FastAPI TestClient checks for the §3 schemas and the Gate-3 offline path.

Run from the project directory:  python -m pytest tests/test_api.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import ctrse_core as core
import api

client = TestClient(api.app)

PICKER_KEYS = {"id", "predicted_level", "level_label", "level_colour",
               "archetype", "summary", "p1_probability"}
DETAIL_KEYS = {"id", "predicted_level", "level_label", "level_colour", "confidence_word",
               "probabilities", "threshold_context", "red_flag_triggered", "red_flag_complaint",
               "active_chief_complaints", "complaint_base_rates", "abnormal_vitals", "age",
               "arrival_mode", "department", "utilisation_history",
               "high_importance_features_present", "shap_top_contributors",
               "escalation_basis", "threshold_sensitive", "triage_vitals", "vitals_not_recorded"}


def test_meta_schema():
    r = client.get("/api/meta")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"model_name", "thr_p1", "chosen_recall", "n_patients",
                                 "dep_name_present", "gemini_available", "feature_count"}
    assert body["model_name"] == "gemini-3.5-flash"
    assert isinstance(body["thr_p1"], float)
    assert isinstance(body["chosen_recall"], float)
    assert body["chosen_recall"] == core.CHOSEN_RECALL
    assert isinstance(body["n_patients"], int) and body["n_patients"] > 0
    assert isinstance(body["dep_name_present"], bool)
    assert isinstance(body["gemini_available"], bool)
    assert isinstance(body["feature_count"], int)


def test_patients_schema():
    r = client.get("/api/patients")
    assert r.status_code == 200
    body = r.json()
    assert body["levels"] == ["P1", "P2", "P3", "P4"]
    patients = body["patients"]
    assert len(patients) == client.get("/api/meta").json()["n_patients"]
    for p in patients:
        # picker fields ONLY — no full payload, no index
        assert set(p.keys()) == PICKER_KEYS
        assert p["predicted_level"] in ("P1", "P2", "P3", "P4")
    ids = [p["id"] for p in patients]
    for a in ("clear_p1", "borderline", "thin_payload"):
        assert a in ids


def test_patient_detail_schema():
    r = client.get("/api/patients/clear_p1")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == DETAIL_KEYS
    assert body["id"] == "clear_p1"
    assert isinstance(body["confidence_word"], str) and body["confidence_word"]
    assert set(body["probabilities"].keys()) == {"P1", "P2", "P3", "P4"}
    # R4: basis fields exposed for the basis badge
    assert body["escalation_basis"] in {"red_flag", "protocol", "physiology", "complaint", "mixed", "routine"}
    assert isinstance(body["threshold_sensitive"], bool)
    # Gate-1 sanity: a real arrival mode, never an insurance value
    assert str(body["arrival_mode"]).lower() not in {
        "medicaid", "medicare", "private", "self pay", "self-pay", "uninsured", "commercial"}


def test_patient_detail_presets_basis():
    for pid, basis in [("demo_protocol_p2", "protocol"), ("demo_physiology_p2", "physiology")]:
        body = client.get(f"/api/patients/{pid}").json()
        assert body["predicted_level"] == "P2"
        assert body["escalation_basis"] == basis


def test_patient_detail_404():
    r = client.get("/api/patients/nonexistent_id")
    assert r.status_code == 404
    assert r.json() == {"error": "unknown patient id"}


def test_explain_offline_justify_never_500():
    r = client.post("/api/explain",
                    json={"patient_id": "clear_p1", "use_case": "justify", "prefer_live": False})
    assert r.status_code == 200          # Gate 3: never 500 on the offline path
    body = r.json()
    assert body["patient_id"] == "clear_p1"
    assert body["use_case"] == "justify"
    assert body["source"] in ("pinned", "offline")
    assert "text" in body
    assert set(body["guardrails"].keys()) == {"passed", "flags"}
    assert body["disclaimer"] == core.DISCLAIMER


def test_explain_offline_handover_never_500():
    r = client.post("/api/explain",
                    json={"patient_id": "borderline", "use_case": "handover", "prefer_live": False})
    assert r.status_code == 200
    body = r.json()
    assert body["use_case"] == "handover"
    assert body["source"] in ("pinned", "offline")
    assert "assessment" in body and "recommendation" in body
    # No synthesised triage clock exists in this system — the envelope must not carry one.
    assert "triaged_at" not in body and "minutes_ago" not in body
    assert body["disclaimer"] == core.DISCLAIMER


def test_explain_offline_across_random_patient():
    # a non-archetype patient has no pinned record -> must resolve 'offline', still 200
    r = client.post("/api/explain",
                    json={"patient_id": "p3_000", "use_case": "justify", "prefer_live": False})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] in ("pinned", "offline")


def test_explain_unknown_patient_not_500():
    r = client.post("/api/explain",
                    json={"patient_id": "nope", "use_case": "justify", "prefer_live": False})
    assert r.status_code == 404
    assert r.json() == {"error": "unknown patient id"}


def test_guardrail_test_flags_synthetic():
    # §3.3 adversarial control: the real guardrail_check on planted non-compliant text must flag.
    for uc in ("justify", "handover"):
        r = client.post("/api/guardrail-test", json={"use_case": uc})
        assert r.status_code == 200
        body = r.json()
        assert body["passed"] is False
        assert body["flags"], "expected the synthetic candidate to trip at least one rule"
        assert body["note"] == "synthetic test input"
        assert body["use_case"] == uc
        assert isinstance(body["candidate"], str) and body["candidate"]
