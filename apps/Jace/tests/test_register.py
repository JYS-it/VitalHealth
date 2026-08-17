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
  3. "More abbreviated" is measured as ARTICLE RATE, not as a count of telegraphic markers
     (c/o, hx, pt, y/o). The marker count worked while B opened with the presenting picture —
     "68 y/o, c/o chest pain, arrived by car" is where those markers live. That picture is now
     rendered by code in Situation and Background and B is explicitly forbidden from restating
     it, so B lost the markers along with the content they attached to and the gate started
     failing on a change it was never meant to catch. Articles survive the move: the handover is
     told to "drop articles and copulas" and the justification to write "clean, readable clinical
     prose", so the ratio still separates the two registers. The marker count is still computed
     and reported in the failure message, just not asserted.

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
    assert "WHAT DROVE THE ACUITY" in system
    assert head  # the B branch still builds


def test_recommendation_asks_for_the_five_heads_one_per_line():
    """R is the section a receiving clinician acts on, and the free-form ask produced one clipped
    line. The named heads are what made it substantial, so they are asserted in BOTH prompts —
    the system instruction and, beside the payload, the user prompt."""
    _ensure_model()
    system = re.sub(r"\s+", " ", core.SYSTEM_PROMPT_HANDOVER)
    user = core.build_user_prompt(_payload(), "B")
    for head in ("Immediate:", "Obtain:", "Complete:", "Monitor:", "Pathway:"):
        assert head in system, f"{head} missing from SYSTEM_PROMPT_HANDOVER"
        assert head in user, f"{head} missing from the B user prompt"
    # Line breaks in R are load-bearing now; the justification's no-bullets rule must not creep in.
    assert "Line breaks inside the Recommendation are REQUIRED" in system


def test_handover_is_told_not_to_restate_the_code_rendered_blocks():
    """The echo this change exists to remove: S and B are rendered from the same payload and
    printed directly above the model's text, so asking B for the values printed them twice."""
    _ensure_model()
    system = re.sub(r"\s+", " ", core.SYSTEM_PROMPT_HANDOVER)
    user_head = core.build_user_prompt(_payload(), "B").split("PAYLOAD:")[0]

    assert "WRITE NO DIGITS" in system
    assert "Do NOT restate any of it" in system
    assert "named not valued" in system
    # The exact wording that licensed the echo, in either prompt.
    assert "vital VALUES" not in system
    assert "vital VALUES" not in user_head
    assert "already printed above your text" in user_head


def test_multi_line_recommendation_survives_parsing():
    """recLines() in the frontend splits R on newlines. That only works if the parser keeps
    them — a strip() that collapsed the block would silently flatten every handover."""
    text = ("Assessment: c/o chest pain, acuity driven by the coded complaint.\n"
            "Recommendation:\n"
            "- Immediate: for prompt senior review, continuous observation.\n"
            "- Obtain: ECG and troponin.\n"
            "- Monitor: escalate if HR climbs further.\n" + core.DISCLAIMER)
    _asmt, rec = core._parse_handover_lines(text)
    lines = [l for l in rec.split("\n") if l.strip()]
    assert len(lines) == 3, rec
    assert lines[0].startswith("- Immediate:")
    assert core.DISCLAIMER not in rec


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

# Telegraphic markers a real handover uses and an explanation does not. Kept as a reported
# signal, no longer the assertion — see _article_rate below for why.
_ABBREVIATIONS = (r"\bc/o\b", r"\bhx\b", r"\bpt\b", r"\by/o\b", r"\bwnl\b", r"\bra\b", r"\byo\b")

_ARTICLES = re.compile(r"\b(?:the|a|an)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z][A-Za-z/'-]*")


def _density(text):
    return sum(1 for pat in _ABBREVIATIONS if re.search(pat, text, re.IGNORECASE))


def _article_rate(text):
    """Articles per word. The prompts' own definition of the register difference: the handover is
    told to "drop articles and copulas", the justification is told to write "clean, readable
    clinical prose". Prose cannot avoid articles; telegraphic fragments barely use them."""
    words = _WORD.findall(text)
    return len(_ARTICLES.findall(text)) / len(words) if words else 0.0


@live
def test_handover_register_is_more_telegraphic_than_justification():
    """CTRSE_GenAI_Refinement_Spec.md:135's register gate, as tone rather than brevity — see the
    module docstring for why the "shorter" half was dropped, and departure 3 for why this counts
    articles rather than abbreviations."""
    _ensure_model()
    payload = _payload()
    a = core.generate(payload, "justify", prefer_live=True)
    b = core.generate(payload, "handover", prefer_live=True)
    if a.get("source") != "live" or b.get("source") != "live":
        pytest.skip("live generation unavailable this run")

    a_body = (a.get("text") or "").replace(core.DISCLAIMER, "")
    b_body = f"{b.get('assessment') or ''} {b.get('recommendation') or ''}"

    assert _article_rate(b_body) < _article_rate(a_body), (
        f"register failed: handover article rate {_article_rate(b_body):.3f} is not below "
        f"justification {_article_rate(a_body):.3f} (abbreviation counts "
        f"B={_density(b_body)} A={_density(a_body)})\nA: {a_body}\nB: {b_body}")


@live
def test_handover_is_substantive_enough_to_hand_over_from():
    """The complaint that prompted dropping the word budget: a one-line Recommendation and a
    thin Assessment do not serve an SBAR. The floors are deliberately asymmetric now — R is the
    section the receiving clinician acts on and carries the weight; A is the reading of a page
    that is already printed above it, so it is meant to be tight."""
    _ensure_model()
    b = core.generate(_payload(), "handover", prefer_live=True)
    if b.get("source") != "live":
        pytest.skip("live generation unavailable this run")

    assessment = (b.get("assessment") or "").split()
    recommendation = (b.get("recommendation") or "").split()
    assert len(assessment) >= 25, f"assessment too thin ({len(assessment)}w): {b.get('assessment')}"
    assert len(recommendation) >= 25, (
        f"recommendation too thin ({len(recommendation)}w): {b.get('recommendation')}")
    # The shape the frontend renders as a list.
    lines = [l for l in (b.get("recommendation") or "").split("\n") if l.strip()]
    assert len(lines) >= 2, f"recommendation is not one item per line: {b.get('recommendation')}"


@live
def test_assessment_does_not_echo_the_code_rendered_vitals():
    """The reported defect. S and B print every number; A restating them was pure duplication.
    Asserted as "no digits at all", because that is the rule the prompt actually states and it
    is the only version of this check that cannot be satisfied by rounding or rephrasing."""
    _ensure_model()
    payload = _payload()
    b = core.generate(payload, "handover", prefer_live=True)
    if b.get("source") != "live":
        pytest.skip("live generation unavailable this run")

    assessment = b.get("assessment") or ""
    # P1/P2/P3/P4 are level labels, not measurements — they are the one permitted digit.
    stripped = re.sub(r"\bP[1-4]\b", "", assessment)
    assert not re.search(r"\d", stripped), f"assessment restates a number: {assessment}"
    for name, entry in payload["triage_vitals"].items():
        assert str(entry["value"]) not in assessment, f"{name} value echoed in A: {assessment}"
