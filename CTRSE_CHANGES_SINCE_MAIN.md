# CTRSE (apps/Jace) — what changed since `main`

Re-context brief. Branch `Clinician-Patient-Yada-yada`, **33 commits ahead of `main`**.
CTRSE scope: **~5,600 lines added / ~560 removed across 20 files.**

---

## 0. The headline change

On `main`, CTRSE was a **single-audience clinician tool**: one SPA, one intake, one model,
two GenAI registers (justify / handover). There was no concept of a user role.

It is now a **two-audience tool on one model**. A patient and a clinician get genuinely
separate surfaces — separate pages, separate API routes, separate GenAI prompts, separate
persisted record types — but both are driven by the same trained model and the same
`ctrse_core.py` safety logic. **Nothing was retrained and no second model was added.**

The architectural invariant from the build spec still holds and got *stronger*:
`ctrse_core.py` remains the single source of truth for every clinical fact and safety
decision. All the new patient logic (urgency banding, redaction, patient prompts) was added
**inside core**, not in `api.py` and not in the frontend.

---

## 1. `ctrse_core.py` (+865 lines) — the clinical logic

### 1.1 Patient urgency banding — `patient_urgency_band(result)`

A deterministic, code-owned function that turns a model result into one of four action
bands: `EMERGENCY_NOW` / `URGENT_TODAY` / `SEE_CLINICIAN` / `UNDETERMINED`.

Design: a **max-lattice**, not an if/elif chain. Four rules each independently propose a
"floor" rank, and the band is the **maximum** floor:

| Rule | Source | Floor | Basis |
|---|---|---|---|
| red flag triggered | `RED_FLAGS` (from model bundle) | 3 | `red_flag` |
| protocol complaint | `PROTOCOL_COMPLAINTS` (validated in `init()`) | 3 | `protocol` |
| abnormal vitals | out of `NORMAL` band | 2 | `physiology` |
| predicted level | P1→3, P2→2, P3/P4→1 | varies | `model_level` |

Because it is a `max()`, it is **structurally incapable of de-escalating** — no code path
ever assigns downward. There is deliberately **no rank 0**: the floor of the lattice is
"see a clinician soon", so no combination of inputs can ever produce "you don't need care".

Also returns `escalated_above_model` (did a safety rule push above what the model alone
said), `vitals_checked`, `reasons[]`, a code-owned `action_line`, and `safety_netting` —
which is appended **at every band including the lowest**.

Mental-health split: a protocol-basis band keeps rank 3 but takes **crisis-line wording,
never resuscitation language** — mirroring the split `SYSTEM_PROMPT_HANDOVER` already drew
for the clinician register.

### 1.2 Patient-safe projection — `patient_view(result)`

Builds a **new dict** (never `dict(result)` then `del`) so an unrecognised future key
defaults to hidden. Deliberately drops: `probabilities`, `threshold_context`,
`threshold_sensitive`, `escalation_basis`, `shap_top_contributors`, `complaint_base_rates`,
`provenance`, raw `cc_*` tokens, `department`, `utilisation_history`, and — the biggest leak
surface — `payload` (the 16-key `explain()` dict, which re-contains most of the above).

### 1.3 New code-owned patient copy tables

`EMERGENCY_CONTACTS` (Singapore 995 / SOS 1767), `PATIENT_SCOPE_NOTE`, `SAFETY_NETTING`,
`PATIENT_WATCH_FOR`, `PATIENT_COMPLAINT_LABELS` (token → plain language),
`PATIENT_LEVEL_HEADLINES`, `PATIENT_CONFIDENCE`, `PATIENT_REASSURING_WORDS`.

`PATIENT_SYMPTOM_OPTIONS` is validated against real `cc_` columns in `init()` — the same
validate-against-the-bundle pattern `PROTOCOL_COMPLAINTS` already used.

### 1.4 A THIRD GenAI use case — `SYSTEM_PROMPT_PATIENT` (use case "C")

`_USE_CASE_MAP` grew: `justify→A`, `handover→B`, **`patient_guidance→C`**.

This is the only GenAI surface written in a **patient register** — A and B are both
explicitly "for a CLINICIAN". Use case C has its own guardrail path,
`_guardrail_check_patient()`, plus a **self-care suppression** mechanism
(`_self_care_suppressed()` + `_SELF_CARE_PATTERNS`) that strips "take paracetamol"-style
advice on paths where it would be dangerous (e.g. an overdose presentation).

---

## 2. `api.py` (+333 lines) — four new patient routes

`main` had 10 routes, all clinician. Now:

| New route | Purpose |
|---|---|
| `POST /api/self-check/extract` | free-text → structured fields (span-or-silence extractor) |
| `GET  /api/self-check/options` | vocabulary + emergency contacts + scope note |
| `POST /api/self-check` | runs the model, returns `patient_view` only |
| `POST /api/self-check/explain` | use case C patient guidance (supplementary) |

**Authorization.** `PATIENT_API_PREFIX = "/api/self-check"` — the middleware
`require_clinician_for_clinical_api` blocks patient sessions from every `/api/*` route
except `/api/dashboard/*` and this prefix. `/api/vocab` and `/api/explain` stay
clinician-gated; `/api/self-check/options` serves the same vocabulary rather than widening
`/api/vocab`'s authz. `/api/self-check/explain` is a **separate route** from `/api/explain`
on purpose — the patient/clinician boundary lives in the routing table, not in a string
parameter.

**Red-flag retention (the key safety decision).** The patient confirm screen is fully
editable against the full vocabulary — same as the clinician's. That opens one path that
must stay closed: editing away an extracted red flag to get a calmer answer. So
`POST /api/self-check` **re-derives red flags server-side from the extraction and unions any
missing ones back in** — never dropped, only added. Adding a token only ever over-triages,
which the max-lattice tolerates by design.

**`other`-only rejection.** If a submission resolves to nothing but the `other` fallback
token, it 422s rather than scoring — otherwise "we didn't understand you" would become a
potentially reassuring P-code.

**Persistence.** Self-checks are `record_type="triage_self_check"`,
`status="PATIENT_SELF_CHECK"` — deliberately **not** `PENDING_REVIEW`, because nothing is
expected to action a self-check and queueing it would be a false safety promise. The
response body is `patient_view` + `record_id` + `confirmed_complaint_tokens` and nothing
else. Only the prepared/redacted note is persisted, never raw browser text.

---

## 3. Frontend — the patient page and the shared intake

### New: `static/self-check.html` + `self-check.js` (+739 lines)

Its own Alpine root, own file — a patient never loads `app.js`. It runs the **same
three-stage interface the clinician uses**: input → confirm extracted fields → result. The
confirm screen is fully editable against the same controlled vocabulary. What differs is
downstream of the model, not the interface: no justification/handover blocks, replaced by
patient guidance (use case C).

Stages: persistent non-dismissible safety notice → free-text input (with 6 first-person demo
seeds, one per path through the guidance model) → skeleton loader → confirm screen with span
highlighting → result (band headline first, model P-code demoted to a collapsed disclosure).

### New: `static/intake-shared.js` (+175 lines)

Pure helper functions shared by `app.js` and `self-check.js` (`buildConfirmFields`,
`highlightedNote`, `friendlyFlag`, `vitalsSummary`, …). Plain global IIFE (`window.VHIntake`)
— this repo has no build step. Every function is pure and computes no clinical fact, which
is what makes it safe to share across the two audiences.

### `index.html` (+712) / `app.js` (+691) / `style.css` (+373)

- **Role branching**: the clinical workspace is wrapped in `<template x-if="!isPatient">` —
  a patient's DOM **genuinely does not contain** the intake form (not merely CSS-hidden).
- New portal dashboard: patient tiles, clinician patient table, pending-review queue.
- **SBAR fix**: the handover adapter was reading `.synthesis` / `.caveat`, but core emits
  `.assessment` / `.recommendation` — Assessment and Recommendation had been rendering as two
  blank boxes and silently dropping out of Copy and Print. Fixed, plus a
  "Draft — requires clinician review" chip on both GenAI registers.
- Fixed an unreachable error banner (`ctrseApp.error` was shadowed by a nested Alpine scope,
  so API failures rendered nothing at all).

---

## 4. New supporting modules

- **`dashboard_api.py`** (new, 329 lines) — 6 read-only dashboard routes, per-route authz.
- **`dashboard_summaries.py`** (new, 467 lines) — one stored record → one dashboard tile, for
  all three modules. `audience="patient"` is a redaction boundary, not a styling hint. A
  patient's own self-check shows its band; a clinician viewing that same record gets a
  first-position fact: **"Patient self-check — self-reported, not clinician-verified."**

---

## 5. Tests — 5 new files (+1,217 lines)

| File | What it locks down |
|---|---|
| `test_patient_view.py` | 32-combination monotonicity (band never falls below model-alone), forbidden-substring leak check, never-reassure scan over every copy table, refusal shape |
| `test_register.py` | The A/B register separation gate the GenAI spec called for but was never built — measures article rate, not length |
| `test_patient_self_check_free_text.py` | Two-step extract→confirm contract, `other`-only rejection, red-flag-cannot-be-edited-away |
| `test_self_check_allowlist.py` | Enumerates the real route table so a future patient route can't silently 403 |
| `test_self_check_explain.py` | Guardrail-failed generations degrade to "unavailable", never shown with a warning; redaction-bypass regression |

**Known pre-existing failure:** `test_parity.py` has 2 failures (`escalation_basis` /
`shap_top_contributors` drift against the frozen `pinned.json` fixture). Reproducible on a
clean checkout with no local changes — unrelated to any of the above; likely needs the
fixture regenerated against current library versions.

---

## 6. Cross-cutting (outside CTRSE, but CTRSE depends on it)

- `vitalhealth_storage/identity.py` (new) — signed session cookies, roles, subject resolution.
- `vitalhealth_storage/settings.py` (new) — shared env loading across all 4 processes.
- `vitalhealth_storage/store.py` (+507) — roles, record ownership, `list_pending_review`.
- `apps/gateway/main.py` (+477) — login/register with roles, plus a patient proxy allowlist
  that **mirrors CTRSE's own middleware** — defence in depth, both must agree independently.
