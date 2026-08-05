"""test_parity.py — the anti-drift guard (build spec §2.7 + refinement §6).

Asserts the shipped sample/ artefacts still match what ctrse_core computes today:
level, the full 16-key payload (incl. escalation_basis / threshold_sensitive), and the
guardrail verdict on each pinned (payload, output) pair. No live key needed — everything
here is deterministic (SHAP is init-seeded).

Run from the project directory:  python -m pytest tests/test_parity.py -v
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import ctrse_core as core  # noqa: E402

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(APP_DIR, "sample")

# prefer_precomputed=False: this anti-drift guard reproduces explain() (incl. SHAP) on
# real X_test rows, so it needs the full arrays + SHAP environment the shipped payloads
# were built in — not the deployable precomputed-stats path (which leaves X_test None and
# SHAP off). Requires the local .npy arrays, as it always has.
core.init(APP_DIR, prefer_precomputed=False)

with open(os.path.join(SAMPLE_DIR, "patients.json"), encoding="utf-8") as f:
    PATIENTS = json.load(f)
with open(os.path.join(SAMPLE_DIR, "pinned.json"), encoding="utf-8") as f:
    PINNED = json.load(f)

PAYLOAD_KEYS = {
    "predicted_level", "probabilities", "threshold_context", "red_flag_triggered", "red_flag_complaint",
    "active_chief_complaints", "complaint_base_rates", "abnormal_vitals", "age", "arrival_mode",
    "department", "utilisation_history", "high_importance_features_present", "shap_top_contributors",
    "escalation_basis", "threshold_sensitive", "triage_vitals", "vitals_not_recorded",
}


def _norm(obj):
    """JSON round-trip so numpy scalars / tuples compare equal to the on-disk payload."""
    def default(o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.loads(json.dumps(obj, default=default))


# --- shape: every payload carries exactly the 16 keys ---

def test_all_patients_payload_16_keys():
    assert PATIENTS, "patients.json is empty"
    for p in PATIENTS:
        keys = set(p["payload"].keys())
        assert keys == PAYLOAD_KEYS, f"{p['id']}: {keys ^ PAYLOAD_KEYS}"


def test_pinned_payload_16_keys():
    assert PINNED, "pinned.json is empty"
    for r in PINNED:
        keys = set(r["payload"].keys())
        assert keys == PAYLOAD_KEYS, f"{r['id']}/{r['use_case']}: {keys ^ PAYLOAD_KEYS}"


# --- parity of each pinned record against a fresh core computation on its row ---

def test_pinned_level_parity():
    for r in PINNED:
        lvl = core.predict_level(core.X_test[r["index"]])
        assert lvl == r["payload"]["predicted_level"], f"{r['id']}: level {lvl} != {r['payload']['predicted_level']}"


def test_pinned_payload_matches_explain():
    for r in PINNED:
        fresh = _norm(core.explain(core.X_test[r["index"]]))
        assert fresh == r["payload"], f"{r['id']}/{r['use_case']}: explain() drifted from pinned payload"


def test_pinned_new_fields_present_and_match():
    for r in PINNED:
        fresh = core.explain(core.X_test[r["index"]])
        for key in ("escalation_basis", "threshold_sensitive"):
            assert key in r["payload"], f"{r['id']}: pinned payload missing {key}"
            assert fresh[key] == r["payload"][key], f"{r['id']}: {key} drift {fresh[key]} != {r['payload'][key]}"


def test_pinned_guardrail_verdict_parity():
    for r in PINNED:
        passed, flags = core.guardrail_check(r["payload"], r["output"], r["use_case"])
        assert passed == r["guardrails_passed"], f"{r['id']}/{r['use_case']}: verdict {passed} != {r['guardrails_passed']}"
        assert flags == r["flags"], f"{r['id']}/{r['use_case']}: flags {flags} != {r['flags']}"


# --- the §3.1 demo presets exist with the intended basis tags ---

def test_demo_presets_present_with_basis():
    by_id = {p["id"]: p for p in PATIENTS}
    for pid, basis in [("demo_protocol_p2", "protocol"), ("demo_physiology_p2", "physiology")]:
        assert pid in by_id, f"missing demo preset {pid}"
        entry = by_id[pid]
        assert entry["archetype"] == pid, f"{pid}: archetype tag not set"
        assert entry["payload"]["predicted_level"] == "P2", f"{pid}: not P2"
        assert entry["payload"]["escalation_basis"] == basis, \
            f"{pid}: basis {entry['payload']['escalation_basis']} != {basis}"
