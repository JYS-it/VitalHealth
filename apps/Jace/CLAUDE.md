## Intake pipeline (current work)
Spec: CTRSE_Pipeline_Build_Spec.md — read before any phase.

Hard rules:
- The extractor is a new front door. The model, threshold, red-flag floor,
  payload, escalation_basis, and both explanation use cases are REUSED UNTOUCHED.
- No second model. No auto-coding of CCS history flags or med classes.
- Every extracted value must quote a literal substring of the note (span-or-silence).
  This is both the anti-hallucination and anti-injection mechanism. Protect it.
- arrivalmode must map to EXACT training strings: Car / ambulance (lowercase) /
  Walk-in / Other / Public Transportation / Wheelchair
- The OrdinalEncoder needs all 13 categoricals; pass "unknown" for the 11
  a note can't supply.
- Vitals are TYPED with explicit units, never extracted from prose.