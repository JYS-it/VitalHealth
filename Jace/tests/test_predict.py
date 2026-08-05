"""test_predict.py — Phase 3 Gate-3 round-trip: the encoder contract + predict parity.

The real test (spec §12 Gate 3): take a real X_test row, build intake fields directly
from it (NO extractor — this isolates the encoder contract from the LLM), run them
through assemble_vector, and prove the assembled vector reproduces the row's
note-suppliable columns exactly and predicts identically to a note-shaped version of
that same row.

Layers:
  1. Encoder contract (fast — init_extraction only, no model): cc_/age/gender/
     arrivalmode match the row; length == len(feature_cols); arrival enum encodes
     non-NaN; conditioned derivation; missing fields -> NaN.
  2. Predict parity (model — one heavy core.init): predict_level(assembled) ==
     predict_level(note_shaped(row)).
  3. Paediatric refusal: core + endpoint return a clean structured refusal, not 500.

Fast layer:  python -m pytest tests/test_predict.py -v -k "contract or refusal_core"
Everything:  python -m pytest tests/test_predict.py -v
"""

import os
import sys

import joblib
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctrse_core as core

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
core.init_extraction(APP_DIR)

X_TEST = np.load(os.path.join(APP_DIR, "X_test.npy"))
FC = core.feature_cols
COL = core.COL
CC_COLS = [c for c in FC if c.startswith("cc_")]
VITAL_NUM = ("hr", "sbp", "dbp", "rr", "o2", "temp")

# feature buckets for the note-shaped baseline (mirror ablate_note_shaped semantics)
KEEP = set(CC_COLS) | {"age", "gender", "arrivalmode"} | {c for c in FC if c.startswith("triage_vital_")}
NAN_COLS = list(core._UTILISATION_COLS) + list(core._UNKNOWN_CATEGORICALS)
ZERO_COLS = [c for c in FC if c not in KEEP and c not in set(NAN_COLS)]   # CCS + meds_

_CAT_POS = {c: k for k, c in enumerate(core._CAT_COLS)}
_ARRIVAL_INV = {v: k for k, v in core._ARRIVAL_MAP.items()}   # exact string -> enum key


def _decode(colname, code):
    cats = core._ORD_ENC.categories_[_CAT_POS[colname]]
    return str(cats[int(code)])


def _derivation_stable(row, age):
    """True iff re-deriving every active token (with the evidence the round-trip
    supplies — age only) is identity. Rows that are NOT stable are ones whose coding
    contradicts §7 (e.g. cc_fall=1 on a >65yo, which §7 says must be cc_fall>65, or
    cc_fever=1 which is always age-derived). Such rows are unreproducible BY DESIGN;
    excluding them isolates the encoder contract from the (separately unit-tested)
    derivation. cc_fall>65 / fever-9weeksto74years etc. stay stable and are kept."""
    for c in CC_COLS:
        if row[COL[c]] == 1:
            t = c[3:]
            if core.derive_conditioned_token(t, age=age) != t:
                return False
    return True


def _clean_enum_rows(limit):
    """Indices of X_test rows whose arrivalmode decodes to one of the 6 enum strings,
    whose age is an in-range integer, and whose complaint coding is derivation-stable
    (§7) — the rows the encoder contract round-trips exactly."""
    idx = []
    for i in range(len(X_TEST)):
        arr = _decode("arrivalmode", X_TEST[i, COL["arrivalmode"]])
        age = X_TEST[i, COL["age"]]
        if arr in _ARRIVAL_INV and not np.isnan(age) and float(age).is_integer() \
                and core.AGE_MIN <= age <= core.AGE_MAX and _derivation_stable(X_TEST[i], int(age)):
            idx.append(i)
        if len(idx) >= limit:
            break
    return idx


def _fields_from_row(row):
    """Reconstruct intake fields from a real row (final cc_ tokens, decoded sex/arrival,
    typed vitals already in °C). Isolates the encoder contract — no LLM involved."""
    vitals = {}
    for key in VITAL_NUM:
        feat = core._VITAL_RULES["vitals"][key]["feature"]
        v = row[COL[feat]]
        vitals[key] = None if np.isnan(v) else float(v)
    vitals["temp_unit"] = "C"
    dev = row[COL["triage_vital_o2_device"]]
    vitals["o2_device"] = None if np.isnan(dev) else float(dev)
    return {
        "age": int(row[COL["age"]]),
        "sex": _decode("gender", row[COL["gender"]]),
        "arrival_mode": _ARRIVAL_INV[_decode("arrivalmode", row[COL["arrivalmode"]])],
        "complaints": [{"token": c[3:]} for c in CC_COLS if row[COL[c]] == 1],
        "vitals": vitals,
    }


def _note_shape(row):
    """Note-shaped baseline of a real row: binarize cc_ (NaN->0, the real note
    semantic — complaint present=1 else 0), zero CCS/meds, NaN utilisation + the 11
    demographics, keep age/gender/arrival/vitals."""
    xn = row.astype(float).copy()
    for c in CC_COLS:
        xn[COL[c]] = 1.0 if row[COL[c]] == 1 else 0.0
    for c in ZERO_COLS:
        xn[COL[c]] = 0.0
    for c in NAN_COLS:
        xn[COL[c]] = np.nan
    return xn


# ===========================================================================
# Layer 1 — encoder contract (fast, no model)
# ===========================================================================

def test_contract_vector_length_and_order():
    asm = core.assemble_vector({"age": 40, "sex": "Male", "arrival_mode": "walk_in",
                                "complaints": [{"token": "sorethroat"}]})
    assert len(asm["vector"]) == len(FC)
    # order honored: the token we set is 1 exactly at its named column, neighbours 0
    assert asm["vector"][COL["cc_sorethroat"]] == 1.0
    assert asm["derived_complaints"] == ["sorethroat"]


def test_contract_roundtrip_matches_row_exactly():
    rows = _clean_enum_rows(40)
    assert len(rows) >= 20, "need clean enum rows to test the contract"
    for i in rows:
        row = X_TEST[i]
        asm = core.assemble_vector(_fields_from_row(row))
        v = asm["vector"]
        # cc_ : assembled (0/1) matches the row binarized (NaN/!=1 -> 0)
        for c in CC_COLS:
            expect = 1.0 if row[COL[c]] == 1 else 0.0
            assert v[COL[c]] == expect, f"row {i}: {c} mismatch"
        # age / gender / arrivalmode encode back to the original codes exactly
        assert v[COL["age"]] == row[COL["age"]], f"row {i}: age"
        assert v[COL["gender"]] == row[COL["gender"]], f"row {i}: gender code"
        assert v[COL["arrivalmode"]] == row[COL["arrivalmode"]], f"row {i}: arrivalmode code"


def test_contract_assembled_equals_note_shaped_vector():
    # the strongest form: assembled == note_shaped(row) on ALL 552 columns
    for i in _clean_enum_rows(25):
        row = X_TEST[i]
        v = core.assemble_vector(_fields_from_row(row))["vector"]
        ns = _note_shape(row)
        assert np.allclose(v, ns, equal_nan=True), f"row {i}: assembled != note_shaped"


def test_contract_arrival_enum_all_nonnan():
    for enum in ("ambulance", "car", "walk_in", "public_transport", "wheelchair", "other"):
        v = core.assemble_vector({"arrival_mode": enum})["vector"]
        assert not np.isnan(v[COL["arrivalmode"]]), f"{enum} encoded to NaN"
    # not-stated arrival -> NaN (the 4th-strongest feature, honestly missing)
    v = core.assemble_vector({})["vector"]
    assert np.isnan(v[COL["arrivalmode"]])


def test_contract_conditioned_derivation_in_assembly():
    # fall + age>65 -> cc_fall>65 (not cc_fall)
    v = core.assemble_vector({"age": 72, "complaints": [{"token": "fall"}]})["vector"]
    assert v[COL["cc_fall>65"]] == 1.0 and v[COL["cc_fall"]] == 0.0
    # fall + age<=65 -> cc_fall
    v = core.assemble_vector({"age": 40, "complaints": [{"token": "fall"}]})["vector"]
    assert v[COL["cc_fall"]] == 1.0 and v[COL["cc_fall>65"]] == 0.0
    # overdose + intentional intent -> cc_overdose-intentional
    v = core.assemble_vector({"age": 34, "complaints": [
        {"token": "overdose", "evidence": {"intent": "intentional"}}]})["vector"]
    assert v[COL["cc_overdose-intentional"]] == 1.0 and v[COL["cc_overdose-accidental"]] == 0.0


def test_contract_missing_fields_are_nan_not_zero():
    v = core.assemble_vector({})["vector"]
    assert np.isnan(v[COL["age"]])
    for key in VITAL_NUM:
        assert np.isnan(v[COL[core._VITAL_RULES["vitals"][key]["feature"]]]), key
    assert np.isnan(v[COL["gender"]])
    for c in core._UTILISATION_COLS:
        assert np.isnan(v[COL[c]])
    # cc_ / CCS / meds default 0
    assert v[COL["cc_chestpain"]] == 0.0 and v[COL["htn"]] == 0.0
    assert (v[[COL[c] for c in ZERO_COLS]] == 0.0).all()


def test_contract_temp_fahrenheit_converted():
    v = core.assemble_vector({"vitals": {"temp": 98.6, "temp_unit": "F"}})["vector"]
    assert abs(v[COL["triage_vital_temp"]] - 37.0) < 1e-6
    v = core.assemble_vector({"vitals": {"temp": 37.0, "temp_unit": "C"}})["vector"]
    assert abs(v[COL["triage_vital_temp"]] - 37.0) < 1e-6


def test_contract_filled_feature_count():
    asm = core.assemble_vector({"age": 68, "sex": "Female", "arrival_mode": "car",
                                "complaints": [{"token": "chestpain"}],
                                "vitals": {"hr": 104, "sbp": 148}})
    assert asm["filled_feature_count"] == 1 + 3 + 2   # cc + demo + vitals


def test_contract_new_fields_do_not_touch_vector():
    """Handoff 2 invariant: allergies / pain_score / onset are display/handover-only.
    The assembled vector must be byte-identical with and without them — the model's
    input contract is unchanged, so the Gate-2 ablation numbers remain valid."""
    base = {"age": 68, "sex": "Female", "arrival_mode": "car",
            "complaints": [{"token": "chestpain"}],
            "vitals": {"hr": 104, "sbp": 148, "temp": 37.1, "temp_unit": "C"}}
    with_extras = dict(base)
    with_extras.update({
        "allergies": [{"value": "penicillin", "span": "penicillin allergy"},
                       {"value": "NKDA", "span": "NKDA"}],
        "pain_score": {"value": 8, "span": "pain 8/10"},
        "onset": [{"complaint": "chestpain", "value": "since this morning",
                    "span": "since this morning"}],
    })
    a = core.assemble_vector(base)
    b = core.assemble_vector(with_extras)
    assert np.allclose(a["vector"], b["vector"], equal_nan=True), \
        "new display-only fields altered the model input vector"
    assert a["filled_feature_count"] == b["filled_feature_count"]
    assert a["derived_complaints"] == b["derived_complaints"]


# ===========================================================================
# Layer 3 — paediatric refusal (fast: refusal path never touches the model)
# ===========================================================================

@pytest.mark.parametrize("age", [6, 0, 17, 103, 150])
def test_refusal_core_out_of_distribution(age):
    r = core.predict_from_fields({"age": age, "complaints": [{"token": "fever"}]})
    assert r["model_refused"] is True
    assert r["derived_from_note"] is True
    assert "refusal_reason" in r and str(age) in r["refusal_reason"]
    assert "predicted_level" not in r


def test_refusal_core_adult_not_refused_shape():
    # age in range -> NOT a refusal (this one needs the model; covered fully in parity test)
    assert core.age_refusal_reason(40) is None
    assert core.age_refusal_reason(18) is None and core.age_refusal_reason(102) is None


# ===========================================================================
# Layer 2 — predict parity (model; one heavy core.init)
# ===========================================================================

def _ensure_model():
    if core.model is None:
        core.init(APP_DIR)


@pytest.mark.model
def test_predict_parity_assembled_equals_note_shaped():
    _ensure_model()
    rows = _clean_enum_rows(60)
    mism = []
    for i in rows:
        row = X_TEST[i]
        v = core.assemble_vector(_fields_from_row(row))["vector"]
        if core.predict_level(v) != core.predict_level(_note_shape(row)):
            mism.append(i)
    assert not mism, f"predict_level mismatch on rows {mism[:10]} ({len(mism)}/{len(rows)})"


@pytest.mark.model
def test_predict_from_fields_payload_shape():
    _ensure_model()
    row = X_TEST[_clean_enum_rows(1)[0]]
    r = core.predict_from_fields(_fields_from_row(row))
    detail_keys = {"id", "predicted_level", "level_label", "level_colour", "confidence_word",
                   "probabilities", "threshold_context", "red_flag_triggered", "red_flag_complaint",
                   "active_chief_complaints", "complaint_base_rates", "abnormal_vitals", "age",
                   "arrival_mode", "department", "utilisation_history",
                   "high_importance_features_present", "shap_top_contributors", "escalation_basis",
                   "threshold_sensitive", "triage_vitals", "vitals_not_recorded"}
    assert detail_keys <= set(r.keys())
    assert r["derived_from_note"] is True and r["model_refused"] is False
    assert isinstance(r["filled_feature_count"], int)
    assert r["provenance"] is not None and "label_noise_ceiling_pct" in r["provenance"]
    assert r["predicted_level"] in ("P1", "P2", "P3", "P4")


# ===========================================================================
# Endpoint transport (marker: imports api -> heavy core.init at startup)
# ===========================================================================

@pytest.mark.endpoint
def test_endpoint_predict_success_shape():
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app)
    r = client.post("/api/predict", json={
        "age": 68, "sex": "Female", "arrival_mode": "ambulance",
        "complaints": [{"token": "chestpain"}],
        "vitals": {"hr": 104, "sbp": 148, "dbp": 92, "o2": 94, "temp": 37.1, "temp_unit": "C"}})
    assert r.status_code == 200
    body = r.json()
    assert body["derived_from_note"] is True and body["model_refused"] is False
    assert body["predicted_level"] in ("P1", "P2", "P3", "P4")
    assert "filled_feature_count" in body and "provenance" in body


@pytest.mark.endpoint
def test_endpoint_predict_paediatric_refusal_not_500():
    from fastapi.testclient import TestClient
    import api
    client = TestClient(api.app)
    r = client.post("/api/predict", json={"age": 6, "complaints": [{"token": "fever"}]})
    assert r.status_code == 200                      # renderable refusal, never 500
    body = r.json()
    assert body["model_refused"] is True
    assert "predicted_level" not in body
