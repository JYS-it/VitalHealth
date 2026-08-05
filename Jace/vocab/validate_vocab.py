"""Gate 0 validator — the four checks from the task, run against the real artefacts."""
import json, os, sys
import numpy as np
import pandas as pd
import joblib

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
feature_cols = list(joblib.load(APP + "/feature_cols.pkl"))
cat_cols = list(joblib.load(APP + "/categorical_cols.pkl"))
enc = joblib.load(APP + "/ordinal_encoder.pkl")
real_cc = set(c[3:] for c in feature_cols if c.startswith("cc_"))

cc_vocab = json.load(open(APP + "/vocab/cc_vocab.json"))
am_map = json.load(open(APP + "/vocab/arrivalmode_map.json"))
vital_rules = json.load(open(APP + "/vocab/vital_rules.json"))

fails = []
def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond: fails.append(msg)

print("[1] cc_vocab.json has exactly 200 tokens, all matching real cc_ columns")
toks = cc_vocab["tokens"]
check(len(toks) == 200, f"token count == 200 (got {len(toks)})")
check(set(toks) == real_cc, "token set == real cc_ column set")
extra = set(toks) - real_cc
missing = real_cc - set(toks)
check(not extra and not missing, f"no extra ({extra}) / missing ({missing})")

print("\n[2] Every conditioned token is marked with a base token and a derivation rule")
cond = [t for t, e in toks.items() if e.get("conditioned")]
check(len(cond) == 17, f"exactly 17 conditioned tokens (got {len(cond)})")
for t in cond:
    e = toks[t]
    ok = bool(e.get("base_token")) and bool(e.get("derivation_rule")) and "requires" in e
    check(ok, f"'{t}' has base_token + derivation_rule + requires")
# conditioned_tokens section mirrors the flags
check(set(cc_vocab["conditioned_tokens"]) == set(cond), "conditioned_tokens section matches flagged tokens")

print("\n[3] arrivalmode map round-trips to non-NaN through the fitted ordinal_encoder")
cat_pos = {c: k for k, c in enumerate(cat_cols)}
def encode_arrival(s):
    row = {c: enc.categories_[cat_pos[c]][0] for c in cat_cols}
    row["arrivalmode"] = s
    return enc.transform(pd.DataFrame([row], columns=cat_cols))[0, cat_pos["arrivalmode"]]
for enum, s in am_map["enum_to_training_string"].items():
    code = encode_arrival(s)
    check(not np.isnan(code), f"{enum:16s} -> {s!r:24s} -> code={code} (non-NaN)")
# case-quirk sanity: the WRONG capitalisation must encode to NaN (proves the map matters)
bad = encode_arrival("Ambulance")
check(np.isnan(bad), f"wrong-case 'Ambulance' -> NaN (got {bad}) — confirms exact-string requirement")

print("\n[3b] vital_rules NORMAL bands match ctrse_core.NORMAL exactly")
CORE_NORMAL = {  # verbatim from ctrse_core.NORMAL
    "triage_vital_o2": (94, 100), "triage_vital_sbp": (90, 180), "triage_vital_dbp": (60, 90),
    "triage_vital_hr": (50, 110), "triage_vital_rr": (10, 24), "triage_vital_temp": (35.5, 38.0),
}
for name, v in vital_rules["vitals"].items():
    feat = v["feature"]
    lo, hi = v["normal"]
    check((lo, hi) == CORE_NORMAL[feat], f"{name} normal {v['normal']} == core NORMAL {list(CORE_NORMAL[feat])}")
# F->C rule sanity
c = (98.6 - 32) * 5 / 9
check(abs(c - 37.0) < 1e-9, f"F->C: 98.6F -> {c:.4f}C ~= 37.0")

print("\n[4] Cluster groupings (for review):")
for name, cl in cc_vocab["clusters"].items():
    mem = ", ".join(f"{m['token']}({m['count']})" + ("*" if m["conditioned"] else "") for m in cl["members"])
    print(f"\n  {name}  [prefer: {cl['preferred_emit']}]")
    print(f"    {cl['description']}")
    print(f"    {mem}")

print("\n  emittable bases (LLM may emit these; * = no plain cc_ column, code must pick a form):")
for b, e in cc_vocab["emittable_bases"].items():
    star = "" if e["base_in_vocab"] else "*"
    print(f"    {b}{star:2s} ({e['requires']}) -> {', '.join(e['resolves_to'])}")

print("\n" + ("ALL GATE-0 CHECKS PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
