"""Gate 2 (spec §2.1, §12) — reproduce the note-shaped ablation on the REAL bundle.

Measurement only. Loads the real ctrse_p1p4_model.pkl + X_test/y_test, evaluates the
existing prediction chain (argmax -> P1-threshold override -> red-flag floor, identical
to ctrse_core.predict_level) on:
  (A) the full test set                         -> baseline
  (B) a note-shaped test set                    -> only note-suppliable features kept
  (C) note-shaped with NO vitals                -> most notes carry none
and writes vocab/ablation_results.json for the Phase-4 provenance banner.

Nothing is trained; ctrse_core.py is not imported or touched.
Run:  python ablate_note_shaped.py    (from the ctrse_app dir, or anywhere — paths are self-locating)
"""
import json
import os

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, fbeta_score

APP = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(APP, "vocab")


def _p(name):
    return os.path.join(APP, name)


# --- artefacts -------------------------------------------------------------
feature_cols = list(joblib.load(_p("feature_cols.pkl")))
categorical_cols = list(joblib.load(_p("categorical_cols.pkl")))
X_test = np.load(_p("X_test.npy"))
y_test = np.load(_p("y_test.npy"))
bundle = joblib.load(_p("ctrse_p1p4_model.pkl"))

model = bundle["model"]
THR_P1 = bundle["p1_threshold"]
RED_FLAGS = bundle["red_flags"]
p1c = bundle["p1_class_index"]           # column index of class P1 in predict_proba
classes = np.asarray(model.classes_)
P1_INT = 3                               # P1/critical is integer class 3 (see ctrse_core)

COL = {c: i for i, c in enumerate(feature_cols)}
N_FEAT = len(feature_cols)

# --- feature buckets (exhaustive, disjoint partition of all columns) -------
cc_cols = [c for c in feature_cols if c.startswith("cc_")]
vital_cols = [c for c in feature_cols if c.startswith("triage_vital_")]
KEEP = set(cc_cols) | {"age", "gender", "arrivalmode"} | set(vital_cols)

# utilisation counts + the 11 non-note demographics -> NaN ("unknown", HGB-native)
NAN_COLS = ["n_edvisits", "n_admissions", "n_surgeries",
            "ethnicity", "race", "lang", "religion", "maritalstatus", "employstatus",
            "insurance_status", "previousdispo", "arrivalmonth", "arrivalday", "arrivalhour_bin"]
# everything else = CCS history flags + meds_* class flags -> 0 (not note-codable)
ZERO_COLS = [c for c in feature_cols if c not in KEEP and c not in set(NAN_COLS)]

# partition sanity
assert set(KEEP) | set(NAN_COLS) | set(ZERO_COLS) == set(feature_cols)
assert len(KEEP) + len(NAN_COLS) + len(ZERO_COLS) == N_FEAT, "buckets not disjoint"
n_ccs = sum(1 for c in ZERO_COLS if not c.startswith("meds_"))
n_meds = sum(1 for c in ZERO_COLS if c.startswith("meds_"))
print(f"features={N_FEAT}  keep(note)={len(KEEP)}  zero(CCS {n_ccs}+meds {n_meds})={len(ZERO_COLS)}  nan={len(NAN_COLS)}")

# red flags that survive in a note-shaped row (i.e. are note-suppliable) vs stripped
rf_kept = [rf for rf in RED_FLAGS if rf in KEEP]
rf_stripped = [rf for rf in RED_FLAGS if rf in COL and rf not in KEEP]
rf_absent = [rf for rf in RED_FLAGS if rf not in COL]
print(f"red_flags={len(RED_FLAGS)}  note-suppliable={len(rf_kept)}  stripped_by_note_shaping={rf_stripped}  not_in_cols={rf_absent}")

nan_idx = [COL[c] for c in NAN_COLS]
zero_idx = [COL[c] for c in ZERO_COLS]
cc_idx = [COL[c] for c in cc_cols]
vital_idx = [COL[c] for c in vital_cols]
demo_present_idx = [COL[c] for c in ("age", "gender", "arrivalmode")]
rf_idx = [COL[rf] for rf in RED_FLAGS if rf in COL]


# --- prediction chain (vectorised form of ctrse_core.predict_level) --------
def predict_chain(X):
    proba = model.predict_proba(X)
    pred = classes[proba.argmax(1)].astype(int).copy()
    pred[proba[:, p1c] >= THR_P1] = P1_INT
    for j in rf_idx:
        pred[X[:, j] == 1] = P1_INT
    return pred


def metrics(y_true, y_pred):
    ba = balanced_accuracy_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2, average="macro", labels=[0, 1, 2, 3], zero_division=0)
    # Recall@P1 = TP_P1 / actual_P1
    mask = y_true == P1_INT
    rec_p1 = float((y_pred[mask] == P1_INT).mean()) if mask.any() else float("nan")
    return {"balanced_accuracy": round(float(ba), 4),
            "macro_f2": round(float(f2), 4),
            "recall_at_p1": round(rec_p1, 4)}


# --- build the ablated matrices -------------------------------------------
def note_shape(X, drop_vitals=False):
    Xn = X.copy().astype(float)
    Xn[:, zero_idx] = 0.0
    Xn[:, nan_idx] = np.nan
    if drop_vitals:
        Xn[:, vital_idx] = np.nan
    return Xn


X_full = X_test.astype(float)
X_note = note_shape(X_test, drop_vitals=False)
X_note_nv = note_shape(X_test, drop_vitals=True)

# sanity: note-shaping must not touch the kept note features
keep_idx = [COL[c] for c in KEEP]
assert np.allclose(np.nan_to_num(X_full[:, keep_idx]), np.nan_to_num(X_note[:, keep_idx]), equal_nan=True)

res_full = metrics(y_test, predict_chain(X_full))
res_note = metrics(y_test, predict_chain(X_note))
res_note_nv = metrics(y_test, predict_chain(X_note_nv))


# --- filled_feature_count: informative (non-default) features per note row --
# A real note fills: its ACTIVE complaints (cc_==1; a 0 is a default, not a fill),
# stated age/gender/arrival, and any typed vitals. Zeroed flags/NaN demographics
# are defaults, not fills. Measured on the note-shaped rows (real demographics/vitals).
active_cc = (X_note[:, cc_idx] == 1).sum(1)
demo_filled = (~np.isnan(X_note[:, demo_present_idx])).sum(1)
vitals_filled = (~np.isnan(X_note[:, vital_idx])).sum(1)
filled = active_cc + demo_filled + vitals_filled
filled_nv = active_cc + demo_filled  # no-vitals variant
fc = {
    "definition": "informative note-derived features per row = active cc_ (==1) + stated age/gender/arrival + typed vitals; zeroed CCS/meds flags and NaN demographics are defaults, not fills",
    "of_total_features": N_FEAT,
    "note_supply_capacity": len(KEEP),
    "with_vitals": {"median": int(np.median(filled)), "mean": round(float(filled.mean()), 2),
                    "min": int(filled.min()), "max": int(filled.max())},
    "without_vitals": {"median": int(np.median(filled_nv)), "mean": round(float(filled_nv.mean()), 2),
                       "min": int(filled_nv.min()), "max": int(filled_nv.max())},
}


# --- comparison table ------------------------------------------------------
def delta(a, b, k):
    return round(b[k] - a[k], 4)


rows = [("Full records (baseline)", res_full),
        ("Note-shaped (+vitals)", res_note),
        ("Note-shaped (no vitals)", res_note_nv)]
print("\n" + "=" * 78)
print(f"{'variant':28s} {'bal_acc':>9s} {'macro_F2':>9s} {'Recall@P1':>10s}")
print("-" * 78)
for name, r in rows:
    print(f"{name:28s} {r['balanced_accuracy']:9.4f} {r['macro_f2']:9.4f} {r['recall_at_p1']:10.4f}")
print("-" * 78)
for name, r in rows[1:]:
    print(f"{'  delta vs baseline: ' + name.split(' ')[1]:28s} "
          f"{delta(res_full, r, 'balanced_accuracy'):+9.4f} "
          f"{delta(res_full, r, 'macro_f2'):+9.4f} "
          f"{delta(res_full, r, 'recall_at_p1'):+10.4f}")
print("=" * 78)
print(f"filled_feature_count (informative, of {N_FEAT}): "
      f"with vitals median={fc['with_vitals']['median']} mean={fc['with_vitals']['mean']} | "
      f"no vitals median={fc['without_vitals']['median']} mean={fc['without_vitals']['mean']}")

# --- expectation check (spec §2.1: ~1 pt bal-acc drop, Recall@P1 flat/up) ----
# The direct §2.1 analog is note-shaped WITH vitals (history/meds/util/demographics
# stripped, vitals retained). The no-vitals variant is the realistic thin-note case.
ba_drop_pts = (res_full["balanced_accuracy"] - res_note["balanced_accuracy"]) * 100
ba_drop_nv_pts = (res_full["balanced_accuracy"] - res_note_nv["balanced_accuracy"]) * 100
rec_delta = res_note["recall_at_p1"] - res_full["recall_at_p1"]
rec_delta_nv = res_note_nv["recall_at_p1"] - res_full["recall_at_p1"]
warnings = []
if not (-0.5 <= ba_drop_pts <= 3.0):
    warnings.append(f"balanced-accuracy drop (note-shaped +vitals) is {ba_drop_pts:+.2f} pts, outside the "
                    "expected ~1 pt band (§2.1). The real bundle may behave differently — revisit the design.")
# Safety claim: P1 sensitivity must not be materially degraded in EITHER variant.
if rec_delta < -0.03 or rec_delta_nv < -0.03:
    warnings.append(f"Recall@P1 materially degraded (+vitals {rec_delta:+.4f}, no-vitals {rec_delta_nv:+.4f}); "
                    "§2.1 expected it flat or up. Investigate before relying on note-shaped P1 predictions.")
print("\nExpectation check (§2.1: ~1 pt bal-acc drop, Recall@P1 flat/slightly up):")
print(f"  bal-acc drop:  +vitals {ba_drop_pts:+.2f} pts | no-vitals {ba_drop_nv_pts:+.2f} pts")
print(f"  Recall@P1 delta: +vitals {rec_delta:+.4f} | no-vitals {rec_delta_nv:+.4f}")
if warnings:
    for w in warnings:
        print("  !! WARNING:", w)
else:
    print("  OK — same qualitative story as the proxy (modest bal-acc cost, P1 sensitivity preserved); "
          "pipeline viable on the existing model.")

# --- write results ---------------------------------------------------------
out = {
    "_meta": {
        "purpose": "Gate 2 (spec §2.1) note-shaped ablation on the REAL ctrse_p1p4_model.pkl. "
                   "These numbers populate the Phase-4 provenance banner (§10).",
        "prediction_chain": "argmax -> P1-threshold override (p1_threshold) -> red-flag floor "
                            "(identical to ctrse_core.predict_level)",
        "p1_threshold": float(THR_P1),
        "n_test_rows": int(len(y_test)),
        "n_features": N_FEAT,
        "note_shaping": {
            "kept_note_features": len(KEEP),
            "zeroed_flags": {"ccs_history": n_ccs, "med_classes": n_meds},
            "nan_features": NAN_COLS,
            "red_flags_stripped_by_note_shaping": rf_stripped,
        },
        "metrics": {"balanced_accuracy": "sklearn balanced_accuracy_score",
                    "macro_f2": "sklearn fbeta_score(beta=2, average='macro')",
                    "recall_at_p1": "TP_P1 / actual_P1 (class 3)"},
        "label_noise_ceiling_pct": 80,
        "label_noise_conflict_pct": 61.9,
        "warnings": warnings,
    },
    "full_records": res_full,
    "note_shaped_with_vitals": res_note,
    "note_shaped_no_vitals": res_note_nv,
    "deltas_vs_full": {
        "note_shaped_with_vitals": {k: delta(res_full, res_note, k) for k in res_full},
        "note_shaped_no_vitals": {k: delta(res_full, res_note_nv, k) for k in res_full},
    },
    "filled_feature_count": fc,
}
os.makedirs(VOCAB, exist_ok=True)
with open(os.path.join(VOCAB, "ablation_results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nwrote {os.path.join('vocab', 'ablation_results.json')}")
