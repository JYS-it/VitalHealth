"""prep_sample.py — one-off builder for the demo patient sample artefact.

Selects the 3 NB3 §5 archetypes + 2 named basis-contrast demo presets (§3.1) + a
stratified random sample (60/level) from X_test, calls ctrse_core for every clinical
fact (single source of truth), and writes:
  sample/patients.json   render-ready 16-key payloads for every demo patient
  sample/pinned.json      demo Gen-AI outputs regenerated live under the CURRENT prompts
  sample/meta.json        header facts for GET /api/meta

Run once (from the ctrse_app directory, where the artefacts live):
    python prep_sample.py
Re-run whenever the NB1/NB2 artefacts OR the prompts change. Pinned regeneration
requires GEMINI_API_KEY in the environment (read by ctrse_core at import).
"""

import json
import os
import sys

import numpy as np

import ctrse_core as core

SAMPLE_DIR = "sample"
N_PER_LEVEL = 60
RANDOM_STATE = 42

# Demo-facing patients whose Gen-AI outputs are pinned to disk (offline + parity ground-truth).
PINNED_IDS = ["clear_p1", "borderline", "thin_payload", "demo_protocol_p2", "demo_physiology_p2"]
PINNED_MAX_RETRIES = 4


def _json_default(o):
    """Safety net for any stray numpy scalar in a payload."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def build_summary(payload):
    """Short human picker line: age · arrival_mode · salient complaint/flag.
    Matches spec §3 example: '74y · ambulance · cardiacarrest red flag'."""
    parts = []
    age = payload.get("age")
    if age is not None:
        parts.append(f"{int(age)}y")
    arrival = payload.get("arrival_mode")
    if arrival is not None:
        parts.append(str(arrival).lower())
    if payload.get("red_flag_triggered") and payload.get("red_flag_complaint"):
        parts.append(f"{payload['red_flag_complaint']} red flag")
    elif payload.get("active_chief_complaints"):
        parts.append(payload["active_chief_complaints"][0])
    elif payload.get("abnormal_vitals"):
        parts.append(payload["abnormal_vitals"][0])
    else:
        parts.append("limited triage info")
    return " · ".join(parts)


def make_entry(idx, entry_id, archetype):
    row = core.X_test[idx]
    payload = core.explain(row)
    lvl = payload["predicted_level"]
    # cheap parity check: predict_level must agree with the payload's level
    assert core.predict_level(row) == lvl, f"predict_level disagrees with explain at row {idx}"
    meta = core.LEVEL_META[lvl]
    return {
        "id": entry_id,
        "index": int(idx),
        "predicted_level": lvl,
        "level_label": meta["label"],
        "level_colour": meta["colour"],
        "archetype": archetype,
        "summary": build_summary(payload),
        "p1_probability": payload["probabilities"]["P1"],
        "payload": payload,
    }


def _first_basis_row(candidate_indices, want_basis):
    """First ascending candidate index whose escalation_basis == want_basis, else None."""
    for idx in candidate_indices:
        pl = core.explain(core.X_test[int(idx)])
        if pl["predicted_level"] == "P2" and pl["escalation_basis"] == want_basis:
            return int(idx)
    return None


def regenerate_pinned(entries_by_id):
    """Regenerate the pinned demo Gen-AI outputs LIVE under the current prompts.
    Reuses core's exact live-call seam so pinned == true model output (core untouched)."""
    records = []
    for pid in PINNED_IDS:
        entry = entries_by_id.get(pid)
        if entry is None:
            print(f"WARNING: pinned patient '{pid}' not in sample — skipping its records")
            continue
        payload = entry["payload"]
        for uc in ("A", "B"):
            system = core.SYSTEM_PROMPT_JUSTIFY if uc == "A" else core.SYSTEM_PROMPT_HANDOVER
            user_prompt = core.build_user_prompt(payload, uc)
            raw, passed, flags = None, False, ["not generated"]
            for _ in range(PINNED_MAX_RETRIES):
                raw = core._live_call(system, user_prompt)
                passed, flags = core.guardrail_check(payload, raw, uc)
                if passed:
                    break
            if not passed:
                print(f"WARNING: pinned {pid}/{uc} did NOT pass guardrails after "
                      f"{PINNED_MAX_RETRIES} tries: {flags}")
            records.append({"id": pid, "index": entry["index"], "use_case": uc,
                            "payload": payload, "output": raw,
                            "guardrails_passed": passed, "flags": flags})
        print(f"  pinned {pid}: A+B regenerated")
    return records


def main():
    print("Loading artefacts via ctrse_core.init('.') ...")
    core.init(".")
    print(f"  THR_P1={core.THR_P1:.4f} | features={len(core.feature_cols)} | "
          f"HAS_SHAP={core.HAS_SHAP} | GEMINI_AVAILABLE={core.GEMINI_AVAILABLE}")

    # --- Model outputs over the whole test set (one batch call) ---
    proba_te = core.model.predict_proba(core.X_test)
    p1c, THR = core.p1c, core.THR_P1

    # --- Archetype selection — verbatim NB3 §5 ---
    cc_idx = [core.COL[c] for c in core.cc_cols]
    vit_idx = [core.COL[v] for v in core.NORMAL if v in core.COL]
    archetype_idx = {
        "clear_p1": int(np.argsort(-proba_te[:, p1c])[0]),
        "borderline": int(np.argmin(np.abs(proba_te[:, p1c] - THR))),
        "thin_payload": int(np.argmin((core.X_test[:, cc_idx] == 1).sum(1) * 10
                                       + (~np.isnan(core.X_test[:, vit_idx])).sum(1))),
    }
    print(f"Archetypes: {archetype_idx}")

    # --- pred_final (same chain as core.predict_level; parity-verified) for stratification ---
    pred_op = core.model.classes_[proba_te.argmax(1)].copy()
    pred_op[proba_te[:, p1c] >= THR] = 3
    pred_final = pred_op.copy()
    for rf in core.RED_FLAGS:
        pred_final[core.X_test[:, core.COL[rf]] == 1] = 3

    # --- Demo presets (§3.1): first P2 rows with protocol- vs physiology-basis ---
    # Cheap vectorized gates keep explain() calls bounded; each gate is a safe superset of the
    # target basis (physiology ⇒ abnormal vitals; protocol ⇒ active protocol complaint + no
    # abnormal vitals), so the first gated row that passes the full basis check is the first match.
    abnormal_mask = np.zeros(len(core.X_test), dtype=bool)
    for v, (lo, hi) in core.NORMAL.items():
        if v in core.COL:
            col = core.X_test[:, core.COL[v]]
            abnormal_mask |= (~np.isnan(col)) & ((col < lo) | (col > hi))
    protocol_cols = [core.COL["cc_" + t] for t in core.PROTOCOL_COMPLAINTS if ("cc_" + t) in core.COL]
    protocol_mask = (core.X_test[:, protocol_cols] == 1).any(1) if protocol_cols else np.zeros(len(core.X_test), bool)

    is_p2 = pred_final == 2
    proto_cands = np.where(is_p2 & protocol_mask & ~abnormal_mask)[0]
    physio_cands = np.where(is_p2 & abnormal_mask)[0]
    preset_idx = {
        "demo_protocol_p2": _first_basis_row(proto_cands, "protocol"),
        "demo_physiology_p2": _first_basis_row(physio_cands, "physiology"),
    }
    for name, idx in preset_idx.items():
        if idx is None:
            print(f"WARNING: could not find a demo preset row for '{name}' — it will be MISSING")
        else:
            print(f"Demo preset {name}: row {idx}")

    rng = np.random.RandomState(RANDOM_STATE)
    arche = set(archetype_idx.values())
    exclude = set(arche) | {i for i in preset_idx.values() if i is not None}

    patients = []

    # Archetypes first (id = archetype name)
    for name, idx in archetype_idx.items():
        patients.append(make_entry(idx, name, name))

    # Demo presets (id = archetype = preset name)
    for name, idx in preset_idx.items():
        if idx is None:
            continue
        if idx in arche:
            print(f"WARNING: preset '{name}' (row {idx}) coincides with an archetype — "
                  "emitting a distinct preset entry for the same row")
        patients.append(make_entry(idx, name, name))

    # Stratified random sample, 60/level, in order P1,P2,P3,P4 (excludes archetypes + presets)
    for lvl, cls in [("P1", 3), ("P2", 2), ("P3", 1), ("P4", 0)]:
        pool = np.array([i for i in np.where(pred_final == cls)[0] if i not in exclude])
        chosen = rng.choice(pool, size=min(N_PER_LEVEL, len(pool)), replace=False)
        for n, idx in enumerate(chosen):
            patients.append(make_entry(int(idx), f"{lvl.lower()}_{n:03d}", None))
        print(f"  {lvl}: pool={len(pool)} sampled={len(chosen)}")

    # --- Write sample/ ---
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    with open(os.path.join(SAMPLE_DIR, "patients.json"), "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2, default=_json_default)

    meta = {
        "thr_p1": float(core.THR_P1),
        "model_name": core.MODEL_NAME,
        "n_patients": len(patients),
        "dep_name_present": ("dep_name" in core.COL),
        "gemini_available": bool(core.GEMINI_AVAILABLE),
        "feature_count": len(core.feature_cols),
    }
    with open(os.path.join(SAMPLE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {len(patients)} patients to {SAMPLE_DIR}/patients.json")
    print(f"meta.json: {meta}")

    # --- Regenerate pinned demo outputs LIVE under the current prompts ---
    if not core.GEMINI_AVAILABLE:
        print("\nERROR: GEMINI_API_KEY not available — pinned.json NOT regenerated (would be stale "
              "under the new prompts). Set the key and re-run. patients.json/meta.json were written.")
        sys.exit(1)

    print("\nRegenerating pinned demo outputs (live)...")
    entries_by_id = {p["id"]: p for p in patients}
    pinned = regenerate_pinned(entries_by_id)
    with open(os.path.join(SAMPLE_DIR, "pinned.json"), "w", encoding="utf-8") as f:
        json.dump(pinned, f, indent=2, default=_json_default)
    n_pass = sum(1 for r in pinned if r["guardrails_passed"])
    print(f"Wrote {len(pinned)} pinned records ({n_pass}/{len(pinned)} passed guardrails) "
          f"to {SAMPLE_DIR}/pinned.json")


if __name__ == "__main__":
    main()
