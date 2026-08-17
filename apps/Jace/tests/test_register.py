"""tests/test_register.py — the A/B register separation gate.

CTRSE_GenAI_Refinement_Spec.md:135 specified this as a Phase-R2 build-time GATE:

    "Register test (build-time check): generate A and B for the same patient and confirm B is
     measurably shorter and more abbreviated. If they're similar length/tone, the register
     feature failed."

It was never built, and the failure it predicted had happened: A (justification) and B (handover)
were both citing the same base rates and feature names, so "one payload, two audiences" was an
assertion with nothing enforcing it.

TWO DELIBERATE DEPARTURES FROM THE SPEC'S WORDING, both recorded so they read as decisions:

  1. "measurably SHORTER" is NOT asserted. The handover is an SBAR summary a receiving clinician
     picks the patient up from; a word budget that made B shorter than A also made its Assessment
     and Recommendation too thin to serve that purpose. Density is tested instead of brevity —
     B must be written in telegraphic register, and may be as long as it needs to be.
  2. Guardrail-enforced register separation was removed. The prompts still forbid raw identifiers
     and cross-register statistics, but no scan checks it, so drift will not be caught here. What
     remains testable without a model is that the two prompts ASK for different content.

Run:  python -m pytest tests/test_register.py -v
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctrse_core as core

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_model():
    """cc_cols / feature_cols back the clinician guardrail branch's complaint scan. init() alone
    is enough — the heavy X_*/y_*.npy path is not touched."""
    if core.cc_cols is None or core.feature_cols is None:
        core.init(APP_DIR)


def _payload(**over):
    """A P2 complaint-driven payload, shaped exactly like explain()'s return."""
    base = {
        "predicted_level": "P2",
        "probabilities": {"P1": 0.31, "P2": 0.42, "P3": 0.18, "P4": 0.09},
        "threshold_context": "just below the P1 alert threshold",
        "red_flag_triggered": False,
        "red_flag_complaint": None,
        "active_chief_complaints": ["chestpain"],
        "complaint_base_rates": [
            {"complaint": "chestpain", "historical_emergency_rate": 0.797, "n_historical": 2497},
        ],
        "abnormal_vitals": ["hr=121"],
        "age": 68,
        "arrival_mode": "Car",
        "department": "unknown",
        "utilisation_history": {},
        "high_importance_features_present": ["cc_chestpain", "arrivalmode"],
        "shap_top_contributors": [{"feature": "cc_chestpain", "contribution": 0.31}],
        "escalation_basis": "complaint",
        "threshold_sensitive": False,
        "triage_vitals": {"hr": {"value": 121, "status": "high"}},
        "vitals_not_recorded": ["o2", "sbp"],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# The prompts must ask for different things
# ---------------------------------------------------------------------------

def test_handover_prompt_forbids_statistics_and_justification_permits_them():
    _ensure_model()
    a = core.build_user_prompt(_payload(), "A")
    b = core.build_user_prompt(_payload(), "B")

    assert "historical base rates" in a
    assert "NOT base rates" in b
    # The exact convergence that broke the feature: B being told to reach for the same
    # statistical fields A uses. Checked on the instruction head, not the payload dump.
    assert "historical base rates" not in b.split("PAYLOAD:")[0]


def test_neither_prompt_solicits_shap_any_more():
    """shap_top_contributors stays in the payload — _payload_dominant_driver needs it to pick
    the basis directive — but no register is told to cite raw feature attributions."""
    _ensure_model()
    for uc in ("A", "B"):
        head = core.build_user_prompt(_payload(), uc).split("PAYLOAD:")[0]
        assert "SHAP" not in head


def test_readable_label_supplied_for_a_covered_token():
    _ensure_model()
    assert 'chestpain -> "chest pain or tightness"' in core.build_user_prompt(_payload(), "A")


def test_no_readable_label_block_when_no_complaint_is_covered():
    """PATIENT_COMPLAINT_LABELS covers 44 of ~200 cc_ tokens; an uncovered complaint gets the
    write-it-as-English rule but no worked example, and must not emit an empty block."""
    _ensure_model()
    prompt = core.build_user_prompt(_payload(active_chief_complaints=["zzznotarealtoken"]), "A")
    assert "READABLE LABELS" not in prompt


def test_handover_prompt_asks_for_full_sbar_content_not_a_word_budget():
    """The handover must be rich enough to hand a patient over. Guards against the 40/15-word
    budget being reintroduced, and checks each SBAR obligation is actually requested."""
    _ensure_model()
    head = core.build_user_prompt(_payload(), "B").split("PAYLOAD:")[0]
    # The prompt is a wrapped literal, so collapse whitespace before matching — otherwise this
    # test breaks on a reflow rather than on a change of meaning.
    system = re.sub(r"\s+", " ", core.SYSTEM_PROMPT_HANDOVER)
    assert "at most" not in system.lower()
    assert "DENSITY, not brevity" in system
    for obligation in ("WHAT DROVE THE ACUITY", "how COMPLETE the picture is", "monitoring"):
        assert obligation in system, obligation
    assert head  # the B branch still builds


# ---------------------------------------------------------------------------
# Guardrail: still advisory, still catches what it kept
# ---------------------------------------------------------------------------

def test_rounded_percentage_of_a_payload_rate_is_traceable():
    """The false positive that cost half the live justifications: the prompt asks for the base
    rate "as a frequency" and the model writes "80%" for a 0.797 rate. That is a real payload
    figure reported to the nearest whole percent, not an invented one."""
    _ensure_model()
    text = ("Based on the information recorded at triage, the model weighted the reported chest "
            "pain. Historically about 80% of such patients are assigned a high-acuity level.\n"
            + core.DISCLAIMER)
    passed, flags = core.guardrail_check(_payload(), text, "A")
    assert passed is True, flags


def test_a_genuinely_invented_number_is_still_rejected():
    """The loosening above must not blunt the scan: 42% traces to no payload rate."""
    _ensure_model()
    text = ("The model weighted the reported chest pain. Historically about 47% of such patients "
            "are assigned a high-acuity level.\n" + core.DISCLAIMER)
    passed, flags = core.guardrail_check(_payload(), text, "A")
    assert passed is False
    assert any("untraceable number" in f for f in flags), flags


def test_handover_still_requires_both_sbar_labels():
    _ensure_model()
    text = "Assessment: 68 y/o, c/o chest pain, HR 121.\n" + core.DISCLAIMER
    passed, flags = core.guardrail_check(_payload(), text, "B")
    assert passed is False
    assert any("handover lines missing" in f for f in flags), flags


def test_handover_still_rejects_a_disposition_decision():
    """Information and action only — the R may say what to obtain, never admit/discharge."""
    _ensure_model()
    text = ("Assessment: 68 y/o, c/o chest pain, HR 121.\n"
            "Recommendation: admit to the medical ward.\n" + core.DISCLAIMER)
    passed, flags = core.guardrail_check(_payload(), text, "B")
    assert passed is False
    assert any("treatment/disposition" in f for f in flags), flags


def test_statistics_in_the_handover_are_no_longer_flagged():
    """Records the removal deliberately: the register split is prompt-only now. If a scan is ever
    reintroduced this test should be deleted along with the decision, not quietly edited."""
    _ensure_model()
    text = ("Assessment: 68 y/o, c/o chest pain, historical emergency rate 0.797. HR 121.\n"
            "Recommendation: obtain SpO2 and SBP.\n" + core.DISCLAIMER)
    passed, flags = core.guardrail_check(_payload(), text, "B")
    assert passed is True, flags


# ---------------------------------------------------------------------------
# The spec's gate, live — density rather than length
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(not core.GEMINI_AVAILABLE,
                          reason="GEMINI_API_KEY not set — live register gate skipped")

# Telegraphic markers a real handover uses and an explanation does not.
_ABBREVIATIONS = (r"\bc/o\b", r"\bhx\b", r"\bpt\b", r"\by/o\b", r"\bwnl\b", r"\bra\b", r"\byo\b")


def _density(text):
    return sum(1 for pat in _ABBREVIATIONS if re.search(pat, text, re.IGNORECASE))


@live
def test_handover_register_is_more_abbreviated_than_justification():
    """CTRSE_GenAI_Refinement_Spec.md:135's register gate, as density rather than brevity — see
    the module docstring for why the "shorter" half was dropped."""
    _ensure_model()
    payload = _payload()
    a = core.generate(payload, "justify", prefer_live=True)
    b = core.generate(payload, "handover", prefer_live=True)
    if a.get("source") != "live" or b.get("source") != "live":
        pytest.skip("live generation unavailable this run")

    a_body = (a.get("text") or "").replace(core.DISCLAIMER, "")
    b_body = f"{b.get('assessment') or ''} {b.get('recommendation') or ''}"

    assert _density(b_body) > _density(a_body), (
        f"register failed: handover density {_density(b_body)} is not above justification "
        f"{_density(a_body)}\nA: {a_body}\nB: {b_body}")


@live
def test_handover_is_substantive_enough_to_hand_over_from():
    """The complaint that prompted dropping the word budget: a one-line Recommendation and a
    thin Assessment do not serve an SBAR."""
    _ensure_model()
    b = core.generate(_payload(), "handover", prefer_live=True)
    if b.get("source") != "live":
        pytest.skip("live generation unavailable this run")

    assessment = (b.get("assessment") or "").split()
    recommendation = (b.get("recommendation") or "").split()
    assert len(assessment) >= 30, f"assessment too thin ({len(assessment)}w): {b.get('assessment')}"
    assert len(recommendation) >= 12, (
        f"recommendation too thin ({len(recommendation)}w): {b.get('recommendation')}")
