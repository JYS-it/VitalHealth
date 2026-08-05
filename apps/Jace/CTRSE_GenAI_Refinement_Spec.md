# CTRSE Gen-AI Refinement — Build Spec (Claude Code-ready)

**Goal:** make the two Gen-AI outputs read as credible, clinical-grade communication *and* make the underlying sophistication visible and probe-proof in the PR2 demo. Every feature is specced as a demo moment with a viva-defense line.

**The governing principle (repeat in `CLAUDE.md`):** the model predicts the triage nurse's ESI assignment. Every generated sentence describes **what the model weighted**, never **what is clinically wrong with the patient**. Each clause must survive: *could the model know this from triage-time data?* If not, it is either code-owned (a fact) or forbidden (an inference the model cannot make).

---

## 1. `escalation_basis` — the headline feature (code-owned)

### 1.1 What it is
A deterministic tag, computed in `ctrse_core.py`, that classifies **why the model landed on this level**, derived from payload facts the model actually used. It is an annotation of **model attribution and label provenance**, not a clinical judgment. This is the field that lets a protocol-driven P2 and a physiology-driven P2 generate completely different explanations from the same predicted level.

### 1.2 Where it's computed
Inside `explain()`, as two new payload fields (making it 16 keys total). It must be code-owned — never inferred by the LLM — because it *controls* the LLM's framing.

- `escalation_basis`: one of `"red_flag" | "protocol" | "physiology" | "complaint" | "mixed" | "routine"`
- `threshold_sensitive`: `bool` — True when `predicted_level == "P1"` and `probabilities["P1"] < 0.5` (i.e. P1 only because the operating threshold is low, not because P1 dominates). Surfaces operating-point honesty.

### 1.3 Classification rules (priority order, first match wins)
Compute from payload facts already present. Determine the **dominant driver** as: the top `shap_top_contributors` feature when SHAP is available; else the first `high_importance_features_present`; else `active_chief_complaints[0]`.

```
1. red_flag_triggered is True
     -> "red_flag"          (the red-flag floor forced the level)

2. predicted_level in {"P1","P2"}:
     dominant driver ∈ PROTOCOL_COMPLAINTS  AND  abnormal_vitals is empty
       -> "protocol"        (safety-policy complaint, vitals normal)
     abnormal_vitals non-empty AND dominant driver is a vital / physiological feature
       -> "physiology"      (model leaned on out-of-range vitals)
     dominant driver is a clinical (non-protocol) complaint AND abnormal_vitals empty
       -> "complaint"       (complaint-driven, e.g. chestpain)
     abnormal_vitals non-empty AND a driving complaint also present
       -> "mixed"
     else
       -> "complaint"       (safe fallback for escalations)

3. predicted_level in {"P3","P4"}
     -> "routine"           (no escalation to explain)
```

`threshold_sensitive` is computed independently and can co-occur with any basis (most often `complaint`/`threshold` P1s).

### 1.4 `PROTOCOL_COMPLAINTS` set — **must be validated against real `cc_` tokens**
Candidate set (behavioural/safety complaints triaged high by policy, not physiology):
```
suicidal, alcoholintoxication, drugoverdose, substanceabuse, psychiatricevaluation,
emotionaldisorder, behavioralproblem, intoxication, overdose, detox, homicidal
```
**Build step (do not skip):** print the actual `cc_*` column names from `feature_cols` and keep only tokens that exist; log the final set. A guessed token that isn't in the data silently breaks the `protocol` branch. This validation is itself a viva point ("the set is data-verified, not assumed").

### 1.5 How it threads into the prompt (per-request framing directive)
`build_user_prompt()` appends a **basis directive** chosen by `escalation_basis` (+ a threshold note if `threshold_sensitive`). Exact directive text:

```
red_flag:   "This level was forced by a red-flag safety rule triggered by {red_flag_complaint}.
             State that the level reflects a rule-based safety override on that complaint. Do not
             imply independent physiological assessment."

protocol:   "This escalation is protocol-driven: {complaint} is triaged high-acuity by safety
             policy in the training data, not by physiological instability, and vitals are within
             normal limits. State explicitly that the prediction reflects protocol-based triage
             priority, NOT a physiological emergency. Do not use physiological-emergency language."

physiology: "The model weighted out-of-range vital signs ({abnormal_vitals}). You may cite these
             recorded values as the basis, framed as observations the model weighted — not as a
             diagnosis or a claim of clinical deterioration."

complaint:  "This escalation is driven mainly by the chief complaint of {complaint}. Reference the
             complaint and its historical high-acuity base rate as the basis. Do not invent
             physiological findings; none are recorded as abnormal."

mixed:      "Multiple recorded factors contribute ({drivers}). Reference them together without
             implying one caused another or asserting a unifying diagnosis."

routine:    "This is a non-urgent prediction. State the basis plainly and do not manufacture
             concern or urgency."

threshold note (appended when threshold_sensitive):
            "Note: P1 was assigned because the model's P1 probability ({p1_prob}) exceeded the
             sensitivity-tuned alert threshold ({thr}), not because P1 is the model's dominant
             prediction. Convey this as a low-confidence, safety-biased flag."
```
All `{…}` are filled by code from the payload before the prompt is sent — the LLM receives resolved text, never placeholders.

**Viva defense:** *"`escalation_basis` is derived deterministically from the model's own attribution and the payload; it annotates why the model escalated, including when the escalation reflects a safety protocol the labels encode rather than physiology. It's a statement about model behaviour and label provenance, provable from the payload — not a clinical claim."*

---

## 2. Prompt changes (exact)

### 2.1 `COMMON_RULES` — replace the framing standard
Remove the "Based on the available physiological risk indicators…" mandate. Replace the rule block with:

```
1. Ground every statement in the payload only. Name no driver, number, symptom, or finding
   that is not present in the payload.
2. Frame statements as what the MODEL weighted at triage, not as clinical facts about the
   patient. Prefer openings like "Based on the information recorded at triage…" or
   "The model weighted…". Do NOT use "physiological indicators" as a blanket frame — vitals
   are frequently not the driver.
3. Describe model behaviour, not patient acuity. Every clause must survive: could the model
   know this from triage-time data? If it asserts something about the patient's body that is
   not in the payload, it is forbidden.
4. Never soften or reassure on a P1 or P2 escalation, and never reassure on any level.
5. State honest uncertainty plainly where the prediction rests on thin data, is near the
   threshold, or is protocol-driven.
6. No generic advice ("seek medical attention"), no restated confidence percentages, no filler
   connectives ("it is worth noting", "taken together", "given the above").
7. Obey the basis directive supplied for this patient.
```
Keep the mandatory closing disclaimer unchanged.

### 2.2 `SYSTEM_PROMPT_JUSTIFY` (Use Case A) — register: readable clinical prose
Append:
```
Write 2–3 sentences of clean, readable clinical prose with uneven sentence length. Moderate
abbreviation is fine (SpO₂, HR, RA). Structure: lead with the driver(s) the model weighted;
support with the historical base rate as a frequency; close with a calibration clause only if
warranted. No headers, no bullets, no restated probabilities, no generic escalation advice.
```

### 2.3 `SYSTEM_PROMPT_HANDOVER` (Use Case B) — register: telegraphic SBAR
Append:
```
Write in telegraphic clinical-handover register — denser and more clipped than the justification.
Standard abbreviations expected (c/o, hx, pt, amb, RA, WNL). Produce exactly two lines:
  Synthesis: <one dense line — what the recorded information shows and what drove the acuity>
  Caveat:    <one line — the model's basis and its limits>
Fragments over full sentences wherever a clinical fragment suffices. If this reads like the
justification, it is wrong.
```

**Register test (build-time check):** generate A and B for the same patient and confirm B is measurably shorter and more abbreviated. If they're similar length/tone, the register feature failed.

---

## 3. UI demo moments

All four use the existing design tokens (§4 of the app build spec). No new colours except where noted.

### 3.1 Side-by-side `escalation_basis` contrast — the money shot
- **Sidebar:** a `Compare` toggle. When on, the patient list allows selecting two patients; main area splits into two equal columns, each rendering a full Model + Gen-AI panel.
- **Named demo presets:** `prep_sample.py` surfaces two additional named entries alongside the archetypes — `demo_physiology_p2` (P2, `escalation_basis == "physiology"`) and `demo_protocol_p2` (P2, `escalation_basis == "protocol"`). A `Demo: basis contrast` button loads both into compare view in one click.
- **The basis tag is a prominent labelled badge** on each panel, directly under the level badge: e.g. `BASIS · PROTOCOL` vs `BASIS · PHYSIOLOGY`, in a muted pill so it reads as metadata, not a second acuity signal. Same `P2 · EMERGENCY` badge on both; different basis; visibly different prose.

**Viva line:** *"Same predicted level, different explanation, because the system classifies why it escalated — protocol vs physiology — deterministically from the payload."*

### 3.2 Register contrast on screen
- The Gen-AI panel renders **both** outputs at once (not a toggle): the justification (prose, `--font-sans`, ≤62ch) and the SBAR handover (telegraphic, fielded) stacked with clear labels. The typographic difference reinforces the register difference. Keep the toggle available for focused viewing, but default the demo view to "both."

**Viva line:** *"One payload, two audiences — a readable justification and a telegraphic handover — with deliberately different registers."*

### 3.3 Guardrail status — prominent, and falsifiable (feature 1)
- Move the guardrail indicator to a **header chip on the Gen-AI panel**: `Guardrails ✓ passed` (`--pass`) / `⚑ flagged: <list>` (`--warn`). Not buried in a footer.
- **Adversarial demo control:** a small `Test guardrail` button that sends a **pre-written deliberately non-compliant candidate** (e.g. a P2 output containing "no cause for concern" + an untraceable number) through the *real* `guardrail_check()` and renders the result — the chip flips to flagged and lists the exact violations caught. A reset restores the live output.
  - Honest framing: this is the **real** guardrail running on planted text, clearly labelled `synthetic test input`. It never alters a live clinical output.

**Viva line:** *"The safety layer is falsifiable — here it is catching a planted violation live, naming the rule it tripped."*

**Build cost:** low. Reuses `guardrail_check()`; adds one button + one canned bad string per use case.

### 3.4 Code-owns-facts / LLM-owns-prose — provenance panel
- A `Provenance` expander on the Gen-AI panel with two-tone treatment: **deterministic zones** (level, probabilities, red flags, basis, all payload facts, SBAR Situation/Background) in one visual treatment; **LLM-authored zones** (the justification prose; SBAR Assessment + Caveat) in another. A legend states the split.
- **Deeper layer (if time):** highlight every number/entity in the LLM prose and, on hover, highlight the payload field it's grounded in — the visible form of the untraceable-numbers guardrail. If a number couldn't be grounded, the guardrail would already have flagged it, so this view *shows the guardrail's grounding working*.

**Viva line:** *"Every safety-critical fact is code-rendered; the model writes only the connective prose — and here's exactly which sentences those are."*

---

## 4. Additional feature 2 — model-sensitivity (occlusion) probe

- **What:** for the selected patient, mask the single dominant driver feature and re-run the model; display `model predicts {P2}; with {chestpain} masked, model predicts {P3}`. A real forward pass with one input occluded.
- **Strict framing (mandatory):** label it **"model sensitivity"**, phrase as *"the model's prediction when this input is removed"* — **never** "if the patient didn't have X." It is a statement about the model's dependence on an input, not a clinical counterfactual. Put this framing in the UI label itself, not just the docs.
- **Why it impresses:** shows live feature-dependence, connects directly to your `dep_name` ablation, and *reinforces* the honesty story (the model is complaint-driven, and here's proof) rather than threatening it.

**Viva line:** *"This is occlusion sensitivity — the model's output with one input masked. It measures what the prediction leans on; it makes no claim about the patient's counterfactual health."*

**Build cost:** medium. One extra endpoint (`POST /api/sensitivity`) that masks a feature index and re-predicts; one UI row. Recommend **after** §1–§3 land, as the depth extra.

---

## 5. Code-owned vs LLM-owned (reaffirmed — unchanged by this work)

**Code:** predicted level/label/colour; full probability distribution; threshold context; `escalation_basis`; `threshold_sensitive`; red-flag boolean + complaint; every payload fact; confidence word; SBAR Situation + Background; guardrail verdict; disclaimer; the sensitivity re-prediction.

**LLM (prose only):** the 2–3 justification sentences; the SBAR Assessment + Caveat lines. Nothing else. The basis directive constrains *how* it writes; it never hands a fact to the model.

---

## 6. Build order & gates

```
Phase R1  ctrse_core: add escalation_basis + threshold_sensitive to explain();
          validate PROTOCOL_COMPLAINTS against real cc_ tokens (log the final set).
          GATE: unit-check basis on the 3 archetypes + a known suicidal-complaint row.

Phase R2  Prompt changes (COMMON_RULES framing, A/B register, basis directive threading).
          GATE: A-vs-B register test (B shorter/denser); protocol row produces NO
          physiological-emergency language; live guardrail still PASSes.

Phase R3  prep_sample: surface demo_physiology_p2 + demo_protocol_p2 named presets.
          GATE: both exist with the intended basis tags.

Phase R4  UI: basis badge, compare view, both-registers view, prominent guardrail chip +
          adversarial control, provenance expander. Human-verified in browser.

Phase R5  (optional depth) sensitivity endpoint + UI row, with the strict "occlusion" framing.
```

Parity note: adding payload keys means `test_parity.py` must be updated to expect 16 keys and to assert `escalation_basis`/`threshold_sensitive` against regenerated pinned outputs.

---

## 7. Trap register (keep visible while building)
```
- Model narrating patient acuity instead of model behaviour   -> the cardinal failure
- "physiological indicators" framing on complaint/protocol cases -> the vitals-weak trap
- Physiological-emergency language on a protocol-driven case   -> dishonest + clinically wrong
- Occlusion probe worded as clinical counterfactual            -> causal overclaim
- Guardrail chip that only ever shows green                    -> looks like theatre; make it falsifiable
- Over-abbreviating A / under-abbreviating B                   -> wastes the register feature
- Clinically wrong severity words (RR 28 "failure", SpO₂ 88% "mild") -> assessor catches instantly
- Base rate cited as a patient probability rather than a cohort frequency -> subtle overclaim
```
