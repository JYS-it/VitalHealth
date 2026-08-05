"""Gate 0 builder — writes vocab/{cc_vocab.json, arrivalmode_map.json, vital_rules.json}.

Every prevalence number and every arrivalmode string is DERIVED from the real
artefacts (feature_cols.pkl, X_train.npy, ordinal_encoder.pkl). Only the semantic
groupings (clusters) and the conditioned-token derivation rules are authored — and
those are validated against the real token list before writing.
"""
import json, os
import numpy as np
import joblib

VOCAB = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(VOCAB)
os.makedirs(VOCAB, exist_ok=True)

feature_cols = list(joblib.load(APP + "/feature_cols.pkl"))
cat_cols = list(joblib.load(APP + "/categorical_cols.pkl"))
enc = joblib.load(APP + "/ordinal_encoder.pkl")
X_train = np.load(APP + "/X_train.npy")

COL = {c: i for i, c in enumerate(feature_cols)}
cc_cols = [c for c in feature_cols if c.startswith("cc_")]
tokens = [c[3:] for c in cc_cols]
ccset = set(tokens)
N = len(X_train)

# ---- prevalence (count + share of training rows where cc_<t> == 1) ----
prev = {}
for c in cc_cols:
    n1 = int(np.nansum(X_train[:, COL[c]] == 1))
    prev[c[3:]] = {"count": n1, "share": round(n1 / N, 5)}

# ---------------------------------------------------------------------------
# The 17 conditioned tokens: base + derivation rule. requires ∈
# {age,intent,history,symptoms,substance,visit_context}. base_in_vocab is
# computed, not asserted: it flags whether code can fall back to a real base cc_
# column (fall/fever/headache/cellulitis) or MUST pick a conditioned form
# because no base column exists (overdose/seizure/withdrawal/bloodsugar/wound).
# ---------------------------------------------------------------------------
CONDITIONED = {
    "fall>65": dict(base="fall", requires="age",
        rule="LLM emits base 'fall'; code sets fall>65 when age>65, else keeps cc_fall."),
    "fever-75yearsorolder": dict(base="fever", requires="age",
        rule="LLM emits base 'fever'; code sets fever-75yearsorolder when age>=75."),
    "fever-9weeksto74years": dict(base="fever", requires="age",
        rule="LLM emits base 'fever'; code sets fever-9weeksto74years when 9 weeks<=age<75 (all adults 18-74)."),
    "overdose-intentional": dict(base="overdose", requires="intent",
        rule="LLM emits base 'overdose' + intent span; code sets -intentional when intent is deliberate/self-harm."),
    "overdose-accidental": dict(base="overdose", requires="intent",
        rule="LLM emits base 'overdose' + intent span; code sets -accidental when intent is accidental OR unstated (default). No plain cc_overdose column exists, so one of the two forms MUST be chosen."),
    "seizure-newonset": dict(base="seizure", requires="history",
        rule="LLM emits base 'seizure' + history span; code sets -newonset when NO prior seizure history is stated."),
    "seizure-priorhxof": dict(base="seizure", requires="history",
        rule="LLM emits base 'seizure' + history span; code sets -priorhxof when a prior seizure history is stated. No singular cc_seizure column exists (only plural cc_seizures, n=127); code must choose a form or the rare plural fallback."),
    "headache-newonsetornewsymptoms": dict(base="headache", requires="history",
        rule="LLM emits base 'headache' + onset/history span; code sets -newonsetornewsymptoms for new onset or new/changed symptoms."),
    "headache-recurrentorknowndxmigraines": dict(base="headache", requires="history",
        rule="LLM emits base 'headache' + history span; code sets -recurrentorknowndxmigraines when recurrent or a known migraine dx is stated."),
    "headachere-evaluation": dict(base="headache", requires="visit_context",
        rule="Re-evaluation visit for a prior headache. Visit-context token (cf. woundre-evaluation), not age/history-conditioned; code sets it when the note frames this as a headache re-check/re-evaluation."),
    "elevatedbloodsugar-symptomatic": dict(base="elevatedbloodsugar", requires="symptoms",
        rule="LLM emits base 'elevatedbloodsugar' (a.k.a. hyperglycemia) + symptom evidence; code sets -symptomatic when symptoms accompany the high glucose. No plain base column exists."),
    "elevatedbloodsugar-nosymptoms": dict(base="elevatedbloodsugar", requires="symptoms",
        rule="LLM emits base 'elevatedbloodsugar'; code sets -nosymptoms when high glucose is reported without symptoms. No plain base column exists."),
    "decreasedbloodsugar-symptomatic": dict(base="decreasedbloodsugar", requires="symptoms",
        rule="LLM emits base 'decreasedbloodsugar' (hypoglycemia); code sets -symptomatic (the only low-blood-sugar form present). No plain base column exists."),
    "withdrawal-alcohol": dict(base="withdrawal", requires="substance",
        rule="LLM emits base 'withdrawal' + substance span; code sets -alcohol when substance is alcohol (the only withdrawal form present). No plain base column exists."),
    "post-opproblem": dict(base="post-opproblem", requires="visit_context",
        rule="Visit-context compound (recent surgery / post-operative complication), NOT age/intent/history-conditioned. Self-based: emitted only when the note frames the visit as a post-op problem; code owns it so the LLM does not guess post-op status."),
    "follow-upcellulitis": dict(base="cellulitis", requires="visit_context",
        rule="Follow-up visit for known cellulitis; code sets it when the note frames a cellulitis re-check, else keeps base cc_cellulitis."),
    "woundre-evaluation": dict(base="wound", requires="visit_context",
        rule="Re-evaluation of a prior wound; code sets it for a wound re-check visit. Base concept 'wound' has no plain column (nearest: cc_woundcheck)."),
}
assert len(CONDITIONED) == 17, len(CONDITIONED)
# every conditioned key must be a real token
missing = [t for t in CONDITIONED if t not in ccset]
assert not missing, f"conditioned tokens not in real vocab: {missing}"

# emittable base concepts the LLM MAY emit that are NOT literal cc_ columns but
# resolve via derivation. base_in_vocab tells the resolver whether a fallback
# column exists.
BASES = {}
for tok, meta in CONDITIONED.items():
    b = meta["base"]
    BASES.setdefault(b, {"forms": [], "base_in_vocab": (b in ccset), "requires": meta["requires"]})
    BASES[b]["forms"].append(tok)

# ---------------------------------------------------------------------------
# Clusters — authored near-synonym groupings. Membership validated vs real
# tokens; order + preferred emit derived from computed prevalence.
# ---------------------------------------------------------------------------
CLUSTER_DEFS = {
    "respiratory": ("Breathing / dyspnoea complaints. Prefer the common token; the rest are rare alternates.",
        ["shortnessofbreath","dyspnea","breathingdifficulty","breathingproblem","respiratorydistress","wheezing","asthma","hemoptysis"]),
    "chest": ("Chest complaints.",
        ["chestpain","chesttightness"]),
    "cardiac_rhythm": ("Fast/irregular heartbeat near-synonyms.",
        ["palpitations","tachycardia","rapidheartrate","irregularheartbeat"]),
    "syncope_altered_mental": ("Collapse / reduced or altered consciousness.",
        ["alteredmentalstatus","syncope","nearsyncope","lethargy","lossofconsciousness","unresponsive","confusion"]),
    "abdominal": ("Abdominal pain near-synonyms.",
        ["abdominalpain","giproblem","epigastricpain","abdominaldistention","abdominalcramping"]),
    "headache": ("Headache complaints. Common forms are age/history-conditioned; LLM emits base 'headache'.",
        ["headache-newonsetornewsymptoms","headachere-evaluation","headache-recurrentorknowndxmigraines","migraine","headache"]),
    "seizure": ("Seizure complaints. LLM emits base 'seizure'; code derives new-onset vs prior-hx.",
        ["seizure-priorhxof","seizure-newonset","seizures"]),
    "fever": ("Fever complaints. Common forms are age-conditioned; LLM emits base 'fever'.",
        ["fever-9weeksto74years","fever-75yearsorolder","feverimmunocompromised","chills","fever"]),
    "alcohol_substance": ("Alcohol / drug / overdose / withdrawal cluster (behavioural + substance).",
        ["alcoholintoxication","drugproblem","alcoholproblem","drug/alcoholassessment","detoxevaluation","poisoning","medicationproblem","addictionproblem","overdose-accidental","withdrawal-alcohol","overdose-intentional","ingestion"]),
    "blood_sugar": ("High/low blood-sugar complaints. Forms are symptom-conditioned; no plain base column.",
        ["elevatedbloodsugar-symptomatic","elevatedbloodsugar-nosymptoms","decreasedbloodsugar-symptomatic","hyperglycemia"]),
    "wound_skin": ("Wound / cellulitis / skin complaints incl. follow-up & re-evaluation visit forms.",
        ["rash","laceration","abscess","woundcheck","woundinfection","skinproblem","skinirritation","follow-upcellulitis","woundre-evaluation","cellulitis"]),
    "fall_trauma": ("Falls and trauma. fall>65 is age-conditioned on base 'fall'.",
        ["fall","motorvehiclecrash","fall>65","assaultvictim","modifiedtrauma","motorcyclecrash","multiplefalls","fulltrauma","trauma"]),
    "psych": ("Psychiatric / behavioural-safety complaints (protocol-triaged).",
        ["suicidal","psychiatricevaluation","anxiety","depression","hallucinations","panicattack","agitation","psychoticsymptoms","homicidal"]),
}

clusters = {}
tok2cluster = {}
for name, (desc, members) in CLUSTER_DEFS.items():
    bad = [m for m in members if m not in ccset]
    assert not bad, f"cluster {name} references non-tokens: {bad}"
    ordered = sorted(members, key=lambda m: -prev[m]["count"])
    # preferred emit = what the LLM should emit for the DOMINANT (most common)
    # presentation of this cluster. If the most common member is a conditioned
    # token, the LLM emits its base concept (code derives the form); otherwise it
    # emits the member token directly. This avoids preferring a rare stray synonym
    # (e.g. 'migraine'/'feverimmunocompromised') over the true base.
    top = ordered[0]
    preferred = CONDITIONED[top]["base"] if top in CONDITIONED else top
    clusters[name] = {
        "description": desc,
        "preferred_emit": preferred,
        "members": [{"token": m, "count": prev[m]["count"], "share": prev[m]["share"],
                     "conditioned": m in CONDITIONED} for m in ordered],
    }
    for m in members:
        tok2cluster[m] = name

# ---------------------------------------------------------------------------
# Assemble cc_vocab.json — `tokens` holds EXACTLY the 200 real tokens.
# ---------------------------------------------------------------------------
tokens_obj = {}
for t in sorted(tokens):
    entry = {"count": prev[t]["count"], "share": prev[t]["share"],
             "cluster": tok2cluster.get(t), "conditioned": t in CONDITIONED}
    if t in CONDITIONED:
        c = CONDITIONED[t]
        entry["base_token"] = c["base"]
        entry["base_in_vocab"] = (c["base"] in ccset)
        entry["requires"] = c["requires"]
        entry["derivation_rule"] = c["rule"]
    tokens_obj[t] = entry

cc_vocab = {
    "_meta": {
        "source": "feature_cols.pkl + X_train.npy (fitted NB1 artefacts)",
        "n_train_rows": N,
        "n_tokens": len(tokens_obj),
        "n_feature_cols_total": len(feature_cols),
        "prevalence_basis": "share = count / n_train_rows, count = rows where cc_<token>==1 in X_train",
        "cap_complaints": 2,
        "other_is_valid_fallback": True,
        "notes": [
            "The 17 conditioned tokens must be DERIVED in code (see §7), never emitted by the LLM (§8 guardrail #3).",
            "The LLM emits a base concept (see emittable_bases) + evidence span; code resolves the conditioned form.",
            "emittable_bases whose base_in_vocab=false have NO plain cc_ column: code MUST pick a conditioned form (with the documented default), it cannot fall back to a base column.",
            "feature_cols total is 552 (200 cc_ + 281 CCS + 48 med + 13 categorical + 3 utilisation + 7 vital), not 553 as the spec prose rounds it; Phase 3 must assert length == len(feature_cols) == 552.",
        ],
    },
    "tokens": tokens_obj,
    "clusters": clusters,
    "conditioned_tokens": {t: {"base_token": CONDITIONED[t]["base"],
                               "base_in_vocab": CONDITIONED[t]["base"] in ccset,
                               "requires": CONDITIONED[t]["requires"],
                               "count": prev[t]["count"], "share": prev[t]["share"],
                               "derivation_rule": CONDITIONED[t]["rule"]}
                           for t in CONDITIONED},
    "emittable_bases": {b: {"base_in_vocab": v["base_in_vocab"],
                            "requires": v["requires"],
                            "resolves_to": sorted(v["forms"], key=lambda m: -prev[m]["count"])}
                        for b, v in BASES.items()},
}

with open(os.path.join(VOCAB, "cc_vocab.json"), "w", encoding="utf-8") as f:
    json.dump(cc_vocab, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# arrivalmode_map.json — enum -> exact training string. Strings pulled from the
# fitted encoder's categories; each verified to round-trip to a non-NaN code.
# ---------------------------------------------------------------------------
cat_pos = {c: k for k, c in enumerate(cat_cols)}
am_cats = [str(x) for x in enc.categories_[cat_pos["arrivalmode"]]]
ENUM_MAP = {
    "ambulance": "ambulance",
    "car": "Car",
    "walk_in": "Walk-in",
    "public_transport": "Public Transportation",
    "wheelchair": "Wheelchair",
    "other": "Other",
}
for v in ENUM_MAP.values():
    assert v in am_cats, f"mapped string {v!r} not a real arrivalmode category"
unmapped = [c for c in am_cats if c not in ENUM_MAP.values() and c != "<NA>"]
arrivalmode_map = {
    "_meta": {
        "source": "ordinal_encoder.pkl categories_ for 'arrivalmode'",
        "purpose": "controlled extractor enum -> EXACT training string. A wrong-case string (e.g. 'Ambulance') encodes to NaN and silently drops the model's 4th-strongest feature.",
        "case_quirk": "'ambulance' is lowercase; Car/Walk-in/Public Transportation/Wheelchair/Other are capitalised.",
        "all_encoder_categories": am_cats,
        "not_stated": "null / omit -> pass 'unknown' downstream (unknown_value=NaN); do NOT map to a real category.",
        "unmapped_real_categories": unmapped,
        "unmapped_note": "'Police' is a real training arrivalmode with no enum value in the §4 controlled set; police-transport arrivals should map to 'other'. '<NA>' is the encoder's missing sentinel (code 0), not an arrival mode.",
    },
    "enum_to_training_string": ENUM_MAP,
}
with open(os.path.join(VOCAB, "arrivalmode_map.json"), "w", encoding="utf-8") as f:
    json.dump(arrivalmode_map, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# vital_rules.json — NORMAL bands (incl. dbp), F->C rule, plausible bounds.
# NORMAL bands mirror ctrse_core.NORMAL exactly (single source of truth check
# done in the validator).
# ---------------------------------------------------------------------------
vital_rules = {
    "_meta": {
        "source": "ctrse_core.NORMAL (normal bands, incl. the dbp band) + NB1 unit facts",
        "model_space_units": "triage_vital_temp is CELSIUS in model space (NB1 converts raw Fahrenheit). All other vitals are in the units below.",
        "typed_not_extracted": "Vitals are typed with explicit units in the UI (§5); never parsed from prose. temp carries a unit toggle; o2 carries a device.",
        "missing": "not recorded -> NaN in the feature vector (HGB handles natively).",
    },
    "vitals": {
        "hr":  {"feature": "triage_vital_hr",  "unit": "bpm",          "normal": [50, 110],  "plausible": [20, 250]},
        "sbp": {"feature": "triage_vital_sbp", "unit": "mmHg",         "normal": [90, 180],  "plausible": [40, 300]},
        "dbp": {"feature": "triage_vital_dbp", "unit": "mmHg",         "normal": [60, 90],   "plausible": [20, 200]},
        "rr":  {"feature": "triage_vital_rr",  "unit": "breaths/min",  "normal": [10, 24],   "plausible": [4, 60]},
        "o2":  {"feature": "triage_vital_o2",  "unit": "%",            "normal": [94, 100],  "plausible": [50, 100],
                "device_feature": "triage_vital_o2_device", "device_note": "device (RA/NC/etc) is a separate categorical, not numeric"},
        "temp": {"feature": "triage_vital_temp", "unit": "C", "normal": [35.5, 38.0], "plausible_c": [30.0, 43.0],
                 "plausible_f": [86.0, 109.4]},
    },
    "temp_conversion": {
        "model_unit": "C",
        "f_to_c": "C = (F - 32) * 5 / 9",
        "rule": "If entered in F, convert to C BEFORE writing triage_vital_temp and before the plausible-range check (use plausible_c).",
        "example": {"98.6F": 37.0, "100.4F": 38.0},
    },
}
with open(os.path.join(VOCAB, "vital_rules.json"), "w", encoding="utf-8") as f:
    json.dump(vital_rules, f, indent=2, ensure_ascii=False)

print("WROTE:")
print("  vocab/cc_vocab.json      tokens=%d clusters=%d conditioned=%d bases=%d"
      % (len(tokens_obj), len(clusters), len(CONDITIONED), len(BASES)))
print("  vocab/arrivalmode_map.json  enum=%d unmapped_real=%s" % (len(ENUM_MAP), unmapped))
print("  vocab/vital_rules.json   vitals=%d" % len(vital_rules["vitals"]))
