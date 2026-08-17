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
    at startup, not the heavy X_*.npy arrays path.

    The two initialisers are guarded SEPARATELY on the globals each one actually sets. Gating
    both behind `core.model is None` made this order-dependent: any earlier test file that
    called core.init() (test_parity, test_register) left core.model set but _ORD_ENC None, so
    this short-circuited and assemble_vector then raised "init_extraction() must run before
    assemble_vector()" — 9 failures that appear only when the whole suite runs.
    """
    if core.model is None:
        core.init(APP_DIR)
    if core.feature_cols is None or core._ORD_ENC is None:
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
        dict(enumerate(core.PATIENT_WATCH_FOR)),
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


# ===========================================================================
# §Patient guidance (use case C) — guardrail_check()'s patient-only branch.
# Grounded in a hand-built patient_view()-shaped dict, the same style as
# _synthetic_result() above: exercising the guardrail doesn't require a live
# LLM call or the real model.
# ===========================================================================

def _synthetic_suggestion_payload(complaint_tokens=None, reported_concerns=None,
                                   history_mentions=None, medications=None, allergies=None):
    """The WIDENED payload api.py assembles for use case C (§Patient guidance): patient_view
    plus the confirm-screen extras (history/medications/allergies/onset/note) and the raw
    confirmed complaint tokens — see ctrse_core.py's build_user_prompt "C" branch and
    _patient_reported_text. `confirmed_complaint_tokens` (not `reported_concerns` alone) is
    what distinguishes this shape from bare patient_view for guardrail_check."""
    return {
        "predicted_level": "P3",
        "level_label": "URGENT",
        "age": 40,
        "arrival_mode": "Car",
        "reported_concerns": reported_concerns or ["Chest pain or tightness"],
        "confirmed_complaint_tokens": complaint_tokens if complaint_tokens is not None else ["chestpain"],
        "recorded_vitals": {},
        "vitals_not_recorded": [],
        "red_flag_triggered": False,
        "urgency": {"band": "URGENT_TODAY", "band_label": "See a clinician urgently today",
                    "action_line": "Please see a clinician today.", "reasons": []},
        "confidence_word": "Moderate",
        "history_mentions": history_mentions or [],
        "medications": medications or [],
        "allergies": allergies or [],
        "onset": [],
        "pain_score": None,
        "note": "chest hurts a bit",
        "disclaimer": core.DISCLAIMER,
    }


@pytest.mark.model
def test_patient_guardrail_passes_a_clean_response():
    _ensure_model()
    payload = _synthetic_suggestion_payload()
    text = ("You told us about chest pain or tightness, so this check is asking you to see a "
            "clinician today. While you wait, try to rest somewhere calm. If the pain gets "
            "worse or spreads, or you develop new or worsening shortness of breath, that "
            "means acting sooner.\n" + core.DISCLAIMER)
    passed, flags = core.guardrail_check(payload, text, "patient_guidance")
    assert passed, flags


@pytest.mark.model
def test_patient_guardrail_rejects_reassuring_language():
    _ensure_model()
    payload = _synthetic_suggestion_payload()
    text = (f"This is {core.PATIENT_REASSURING_WORDS[0]} and you'll likely be seen quickly. "
            + core.PATIENT_WATCH_FOR[0] + " is a sign to watch for.\n" + core.DISCLAIMER)
    passed, flags = core.guardrail_check(payload, text, "patient_guidance")
    assert not passed
    assert any("reassuring" in f for f in flags)


@pytest.mark.model
def test_patient_guardrail_rejects_an_invented_watch_for_symptom():
    """The one genuinely new-content risk the escalation-signs half of this feature carries:
    they may only draw from PATIENT_WATCH_FOR, never invent a complaint-specific red flag not
    on that list. (While-waiting/self-care content is deliberately NOT list-constrained — see
    _self_care_suppressed — this test is about the escalation signs specifically.)"""
    _ensure_model()
    payload = _synthetic_suggestion_payload()
    watch_for_text = " ".join(core.PATIENT_WATCH_FOR).lower()
    candidate = next(
        cc[3:] for cc in core.cc_cols
        if len(cc) - 3 >= 6
        and cc[3:].lower() not in watch_for_text
        and cc[3:].lower() != "chestpain"
    )
    text = ("You told us about chest pain, so this check asks you to see a clinician today. "
            f"Watch for {candidate}, which can be serious.\n" + core.DISCLAIMER)
    passed, flags = core.guardrail_check(payload, text, "patient_guidance")
    assert not passed
    assert any("unreported symptom" in f for f in flags)


@pytest.mark.model
def test_patient_guardrail_does_not_flag_grounded_history_or_medications():
    """The widened payload's own history_mentions/medications/allergies are legitimately
    grounded (SYSTEM_PROMPT_PATIENT rule 1) — mentioning them must not trip the
    unreported-symptom scan the way an invented symptom would."""
    _ensure_model()
    payload = _synthetic_suggestion_payload(
        history_mentions=[{"text": "history of migraines", "span": "history of migraines"}],
        medications=[{"text": "ibuprofen", "span": "ibuprofen"}],
    )
    text = ("You told us about chest pain, and mentioned a history of migraines and taking "
            "ibuprofen, so this check asks you to see a clinician today.\n" + core.DISCLAIMER)
    passed, flags = core.guardrail_check(payload, text, "patient_guidance")
    assert passed, flags


@pytest.mark.model
def test_patient_guardrail_requires_the_disclaimer():
    _ensure_model()
    payload = _synthetic_suggestion_payload()
    text = "You told us about chest pain. " + core.PATIENT_WATCH_FOR[0]
    passed, flags = core.guardrail_check(payload, text, "patient_guidance")
    assert not passed
    assert any("disclaimer" in f for f in flags)


# ===========================================================================
# §Patient guidance (C) — self-care/while-waiting suppression. While-waiting content
# (including medication suggestions) is deliberately free-generated with no drug-specific
# list — an accepted trade for usefulness — EXCEPT for overdose, protocol
# (suicidal/homicidal/psychiatricevaluation/alcoholintoxication), and red-flag complaints,
# where it is withheld entirely: both by omitting the instruction from the prompt AND by a
# guardrail scan rejecting it if it appears anyway.
# ===========================================================================

@pytest.mark.model
@pytest.mark.parametrize("token", [
    "overdose", "overdose-intentional", "overdose-accidental",   # prefix match
    "suicidal", "homicidal", "psychiatricevaluation", "alcoholintoxication",  # PROTOCOL_COMPLAINTS
    "cardiacarrest", "unresponsive", "strokealert", "fulltrauma",  # RED_FLAGS
])
def test_self_care_suppressed_for_overdose_protocol_and_red_flag_tokens(token):
    _ensure_model()
    assert core._self_care_suppressed([token]) is True


@pytest.mark.model
def test_self_care_not_suppressed_for_an_ordinary_complaint():
    _ensure_model()
    assert core._self_care_suppressed(["sorethroat"]) is False
    assert core._self_care_suppressed([]) is False
    assert core._self_care_suppressed(None) is False


@pytest.mark.model
def test_build_user_prompt_c_includes_while_waiting_directive_when_allowed():
    _ensure_model()
    payload = _synthetic_suggestion_payload(complaint_tokens=["sorethroat"])
    prompt = core.build_user_prompt(payload, "patient_guidance")
    assert core._WHILE_WAITING_ALLOWED in prompt
    assert core._WHILE_WAITING_SUPPRESSED not in prompt


@pytest.mark.model
@pytest.mark.parametrize("token", ["overdose", "suicidal", "cardiacarrest"])
def test_build_user_prompt_c_suppresses_while_waiting_directive(token):
    _ensure_model()
    payload = _synthetic_suggestion_payload(complaint_tokens=[token])
    prompt = core.build_user_prompt(payload, "patient_guidance")
    assert core._WHILE_WAITING_SUPPRESSED in prompt
    assert core._WHILE_WAITING_ALLOWED not in prompt


@pytest.mark.model
@pytest.mark.parametrize("token", ["overdose", "suicidal", "homicidal", "alcoholintoxication",
                                    "cardiacarrest", "unresponsive"])
def test_patient_guardrail_rejects_medication_content_when_suppressed(token):
    _ensure_model()
    payload = _synthetic_suggestion_payload(complaint_tokens=[token], reported_concerns=["symptom"])
    text = "You should take paracetamol while you wait to be seen.\n" + core.DISCLAIMER
    passed, flags = core.guardrail_check(payload, text, "patient_guidance")
    assert not passed
    assert any("suppressed" in f for f in flags)


@pytest.mark.model
def test_patient_guardrail_allows_medication_content_when_not_suppressed():
    """Proves suppression is targeted, not blanket: the same medication-shaped text passes
    for an ordinary complaint, where self-care guidance is intentionally unconstrained."""
    _ensure_model()
    payload = _synthetic_suggestion_payload(complaint_tokens=["sorethroat"],
                                             reported_concerns=["Sore throat"])
    text = ("You told us about your sore throat, so this check is asking you to see a "
            "clinician today. While you wait, you may take paracetamol for the discomfort.\n"
            + core.DISCLAIMER)
    passed, flags = core.guardrail_check(payload, text, "patient_guidance")
    assert passed, flags


# ===========================================================================
# §Describe-help (use case D) — guardrail_check()'s other patient-only branch.
# Shares _guardrail_check_patient with C, but payload is the raw extraction
# object (before any prediction exists), and D does NOT require the closing
# DISCLAIMER — see SYSTEM_PROMPT_DESCRIBE_HELP's own comment on why.
# ===========================================================================

def _synthetic_extraction(complaints=None):
    return {
        "age": None, "sex": None, "arrival_mode": None,
        "complaints": complaints if complaints is not None else [
            {"token": "chestpain", "span": "chest hurts", "ambiguous": False,
             "alternates": [], "evidence": None},
        ],
        "onset": [], "history_mentions": [], "medications": [], "allergies": [], "pain_score": None,
        "red_flags": [], "unmapped": [], "guardrail_flags": [], "model_refused": False,
        "note_used": "my chest hurts", "truncated": False, "redactions": 0,
        "extracted_nothing": False, "source": "live",
    }


@pytest.mark.model
def test_describe_help_guardrail_passes_a_clean_response():
    _ensure_model()
    payload = _synthetic_extraction()
    text = "Could you tell us a bit more about when your chest pain started and how severe it feels?"
    passed, flags = core.guardrail_check(payload, text, "describe_help")
    assert passed, flags


@pytest.mark.model
def test_describe_help_guardrail_does_not_require_a_disclaimer():
    """At confirm time there is no prediction yet, so the closing DISCLAIMER sentence
    ('the model's prediction of a triage assignment') would describe something that
    doesn't exist. This is the one deliberate divergence from A/B/C."""
    _ensure_model()
    payload = _synthetic_extraction()
    text = "Thanks — we've clearly captured your chest pain and when it started."
    passed, flags = core.guardrail_check(payload, text, "describe_help")
    assert passed, flags
    assert not any("disclaimer" in f for f in flags)


@pytest.mark.model
def test_describe_help_guardrail_rejects_reassuring_language():
    _ensure_model()
    payload = _synthetic_extraction()
    text = f"This is {core.PATIENT_REASSURING_WORDS[0]}, nothing to worry about."
    passed, flags = core.guardrail_check(payload, text, "describe_help")
    assert not passed
    assert any("reassuring" in f for f in flags)


@pytest.mark.model
def test_describe_help_guardrail_rejects_a_suggested_symptom_not_mentioned():
    """The one rule this whole use case exists to make falsifiable: it may ask for more
    detail on something already reported, never suggest a symptom the patient didn't
    mention — that would corrupt the very extraction that feeds the triage model."""
    _ensure_model()
    payload = _synthetic_extraction()
    candidate = next(
        cc[3:] for cc in core.cc_cols
        if len(cc) - 3 >= 6 and cc[3:].lower() != "chestpain"
    )
    text = f"Do you also have {candidate}? Please tell us if that's happening too."
    passed, flags = core.guardrail_check(payload, text, "describe_help")
    assert not passed
    assert any("unreported symptom" in f for f in flags)


@pytest.mark.model
def test_describe_help_guardrail_rejects_diagnostic_language():
    _ensure_model()
    payload = _synthetic_extraction()
    text = "The patient is diagnosed with a mild condition, no further detail needed."
    passed, flags = core.guardrail_check(payload, text, "describe_help")
    assert not passed
    assert any("diagnostic" in f for f in flags)
