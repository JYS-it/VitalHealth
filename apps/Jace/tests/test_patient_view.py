"""test_patient_view.py — the patient self-check surface's safety net.

Tests patient_urgency_band() and patient_view() in ctrse_core.py: the
deterministic urgency banding and the patient-safe projection that the
/api/self-check route (api.py) returns verbatim. These tests make "never
de-escalate" and "never leak a withheld key" falsifiable, the same way
/api/guardrail-test makes the LLM layer's guardrails falsifiable — proving the
safety layer can fail a planted input, rather than just trusting it by
inspection.

Run:  python -m pytest tests/test_patient_view.py -v
"""

import itertools
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctrse_core as core

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_model():
    """PROTOCOL_COMPLAINTS (needed by patient_urgency_band) is only populated
    by core.init() — the same precomputed-stats fast path api.py itself uses
    at startup, not the heavy X_*.npy arrays path."""
    if core.model is None:
        core.init(APP_DIR)
        core.init_extraction(APP_DIR)


def _synthetic_result(level="P3", red_flag=False, protocol=False, abnormal=False,
                       vitals_not_recorded=None):
    """A hand-built predict_from_fields()-shaped dict — patient_urgency_band()
    only reads a handful of keys, so exercising it doesn't require running the
    real model for every combination."""
    return {
        "model_refused": False,
        "predicted_level": level,
        "red_flag_triggered": red_flag,
        "active_chief_complaints": ["suicidal"] if protocol else [],
        "abnormal_vitals": ["hr=140"] if abnormal else [],
        "vitals_not_recorded": vitals_not_recorded or [],
        "confidence_word": "Moderate",
        "level_label": "URGENT",
        "level_colour": "#2ca02c",
        "age": 40,
        "arrival_mode": "Car",
        "triage_vitals": {},
    }


_RANK_ORDER = ("SEE_CLINICIAN", "URGENT_TODAY", "EMERGENCY_NOW")
_LEVEL_ONLY_RANK = {"P1": 3, "P2": 2, "P3": 1, "P4": 1}


# ===========================================================================
# patient_urgency_band() — the max-lattice monotonicity property
# ===========================================================================

@pytest.mark.model
@pytest.mark.parametrize(
    "level,red_flag,protocol,abnormal",
    list(itertools.product(("P1", "P2", "P3", "P4"), (False, True), (False, True), (False, True))),
)
def test_band_never_falls_below_the_model_level_alone(level, red_flag, protocol, abnormal):
    """The core falsifiability property, across all 32 combinations: the band
    is never lower than what predicted_level alone would produce. A max()
    over ranks cannot regress this structurally; this test exists to catch a
    future edit that turns the lattice back into an if/elif chain that can."""
    _ensure_model()
    result = _synthetic_result(level, red_flag, protocol, abnormal)
    band = core.patient_urgency_band(result)

    level_only_band = _RANK_ORDER[_LEVEL_ONLY_RANK[level] - 1]
    assert _RANK_ORDER.index(band["band"]) >= _RANK_ORDER.index(level_only_band)

    non_model_rank = max(
        [rank for rank, fired in ((3, red_flag), (3, protocol), (2, abnormal)) if fired],
        default=0,
    )
    assert band["escalated_above_model"] == (non_model_rank > _LEVEL_ONLY_RANK[level])


@pytest.mark.model
def test_red_flag_always_produces_the_top_band():
    _ensure_model()
    band = core.patient_urgency_band(_synthetic_result(level="P4", red_flag=True))
    assert band["band"] == "EMERGENCY_NOW"
    assert band["band_basis"] == "red_flag"


@pytest.mark.model
def test_protocol_complaint_uses_crisis_wording_not_resuscitation():
    """SYSTEM_PROMPT_HANDOVER already draws this line for the clinician
    register ("mental-health risk assessment; do not route to medical
    resuscitation") — the patient register must honour the same split."""
    _ensure_model()
    band = core.patient_urgency_band(_synthetic_result(level="P3", protocol=True))
    assert band["band"] == "EMERGENCY_NOW"
    assert band["band_basis"] == "protocol"
    assert core.EMERGENCY_CONTACTS["crisis_number"] in band["action_line"]
    lowered = band["action_line"].lower()
    assert "resuscitat" not in lowered
    assert "cardiac" not in lowered


@pytest.mark.model
def test_abnormal_vitals_escalate_a_non_urgent_level():
    """The regression the max-lattice exists to guarantee: a P4 with one
    out-of-range vital must not read as routine."""
    _ensure_model()
    band = core.patient_urgency_band(_synthetic_result(level="P4", abnormal=True))
    assert band["band"] == "URGENT_TODAY"
    assert band["escalated_above_model"] is True


@pytest.mark.model
def test_no_rule_firing_still_has_a_floor():
    """There is no rank 0 — even the quietest case is an instruction to be
    seen, never a dismissal, and safety-netting is always present."""
    _ensure_model()
    band = core.patient_urgency_band(_synthetic_result(level="P4"))
    assert band["band"] == "SEE_CLINICIAN"
    assert band["safety_netting"]


@pytest.mark.model
def test_missing_vitals_never_reads_as_normal():
    _ensure_model()
    band = core.patient_urgency_band(_synthetic_result(
        level="P3", vitals_not_recorded=["o2", "sbp", "dbp", "hr", "rr", "temp"]))
    assert band["vitals_checked"] is False
    assert "vitals_not_recorded" in [r["code"] for r in band["reasons"]]
    combined = json.dumps(band).lower()
    assert "vitals are normal" not in combined
    assert "vitals were normal" not in combined


def test_refusal_shape_has_no_synthesized_level():
    """No core.init() needed — the refusal branch is a pure function of
    model_refused, independent of the model bundle."""
    result = {"model_refused": True, "refusal_reason": "age 6 is outside the model's "
              "training distribution (adults 18-102)", "age": 6, "derived_from_note": True}
    band = core.patient_urgency_band(result)
    assert band["band"] == "UNDETERMINED"
    assert band["band_tone"] == "warning"          # never neutral
    for key in ("predicted_level", "level_label", "confidence_word"):
        assert key not in band
    # plain-language rewrite, not core's verbose training-distribution wording
    assert "training distribution" not in json.dumps(band)


def test_refusal_age_none_is_not_a_refusal():
    assert core.age_refusal_reason(None) is None


# ===========================================================================
# patient_view() — the redaction boundary
# ===========================================================================

_FORBIDDEN_SUBSTRINGS = [
    "probabilit", "shap", "threshold", "base_rate", "escalation_basis",
    "provenance", "cc_", "utilisation", "dep_name", "n_edvisits", '"payload"',
    "n_features", "label_noise",
]


@pytest.mark.model
def test_patient_view_never_leaks_withheld_keys():
    _ensure_model()
    fields = {"age": 68, "sex": "Female", "arrival_mode": "ambulance",
              "complaints": [{"token": "chestpain", "evidence": None}],
              "vitals": {"hr": 104, "sbp": 148, "dbp": 92, "o2": 94, "temp": 37.1, "temp_unit": "C"}}
    result = core.predict_from_fields(fields)
    dumped = json.dumps(core.patient_view(result)).lower()
    hits = [s for s in _FORBIDDEN_SUBSTRINGS if s in dumped]
    assert not hits, f"patient_view leaked: {hits}"


@pytest.mark.model
def test_patient_view_maps_raw_complaint_tokens_to_labels():
    _ensure_model()
    fields = {"age": 40, "complaints": [{"token": "chestpain", "evidence": None}]}
    result = core.predict_from_fields(fields)
    view = core.patient_view(result)
    assert "chestpain" not in json.dumps(view)
    assert any("chest" in c.lower() for c in view["reported_concerns"])


def test_patient_view_refusal_shape_is_minimal():
    result = {"model_refused": True, "refusal_reason": "x", "age": 6, "derived_from_note": True}
    view = core.patient_view(result)
    assert view["model_refused"] is True
    assert set(view.keys()) == {
        "model_refused", "urgency", "disclaimer", "scope_note", "limitations",
        "released_without_clinician_review",
    }


# ===========================================================================
# The never-reassure discipline — a unit test over the constant tables, not a
# runtime scanner. Cheaper than a text-scan guardrail and strictly stronger:
# it fails at commit time, not at demo time.
# ===========================================================================

def test_no_patient_copy_table_contains_a_reassuring_phrase():
    tables = [
        core._BAND_LABELS,
        core._ACTION_LINES,
        core._REASON_TEXTS,
        {"safety_netting": core.SAFETY_NETTING},
        core.PATIENT_LEVEL_HEADLINES,
        {"scope_note": core.PATIENT_SCOPE_NOTE},
    ]
    hits = []
    for table in tables:
        for key, value in table.items():
            lowered = str(value).lower()
            for word in core.PATIENT_REASSURING_WORDS:
                if word in lowered:
                    hits.append((key, word))
    assert not hits, f"reassuring language found: {hits}"


@pytest.mark.model
def test_patient_symptom_options_are_all_real_tokens():
    """A PATIENT_SYMPTOM_OPTIONS_CANDIDATES entry silently dropped at init()
    means a label the code promises to show never actually appears in the
    picker — this must be zero, not just non-crashing."""
    _ensure_model()
    dropped = len(core.PATIENT_SYMPTOM_OPTIONS_CANDIDATES) - len(core.PATIENT_SYMPTOM_OPTIONS)
    assert dropped == 0, f"{dropped} candidate token(s) have no matching cc_ column"


@pytest.mark.model
def test_red_flags_and_protocol_complaints_have_patient_labels():
    """Every token RED_FLAGS/PROTOCOL_COMPLAINTS can actually fire must have a
    human label, or patient_urgency_band's reasons/patient_view's
    reported_concerns would fall back to a raw cc_ token on a patient's
    screen. RED_FLAGS is loaded from the model bundle, not hardcoded — this
    is what protects against a bundle update silently reintroducing a leak."""
    _ensure_model()
    for red_flag_col in core.RED_FLAGS:
        token = red_flag_col[3:] if red_flag_col.startswith("cc_") else red_flag_col
        assert token in core.PATIENT_COMPLAINT_LABELS, f"no patient label for red flag {token!r}"
    for token in core.PROTOCOL_COMPLAINTS:
        assert token in core.PATIENT_COMPLAINT_LABELS, f"no patient label for protocol complaint {token!r}"
