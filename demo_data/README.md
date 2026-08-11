# Demo data

Eight mock patients ("characters"), each with a scenario in all three apps
(triage, stroke risk, EMC), for demonstrating VitalHealth. Two deliverables:

- **`CHEAT_SHEET.md`** — values to paste into each app's live web form
  during a demo, with the real model's expected result for each scenario.
  Already generated and committed; regenerate after editing `characters.py`.
- **The shared database** — the same characters seeded as real
  `patients`/`clinical_records`/`audit_events` rows, so the audit trail is
  genuinely populated, not just talked about.

`characters.py` is the single source of truth. Everything else is derived
from it.

## Why this needs multiple Python environments

Jace, Jeslyn, and YS pin conflicting `scikit-learn` versions and live in
separate venvs (see the root `README.md`, "Why separate environments") — no
one Python process can load all three apps' models together. So getting
*real* model output (not hand-guessed) takes three separate compute steps,
each run with its own app's venv, before a final step writes everything to
the database with any venv that has `vitalhealth_storage` installed.

## Run order

From the repo root, in order:

```powershell
apps\Jace\.venv\Scripts\python.exe demo_data\compute_triage.py
apps\Jeslyn\.venv\Scripts\python.exe demo_data\compute_stroke.py
apps\YS\.venv\Scripts\python.exe demo_data\compute_emc.py
apps\gateway\.venv\Scripts\python.exe demo_data\seed_db.py
python demo_data\render_cheat_sheet.py
python demo_data\sync_jace_seeds.py
```

- The three `compute_*.py` scripts call each app's real prediction logic
  directly (no server needs to be running, no HTTP) and write
  `demo_data/output/{triage,stroke,emc}.json` — gitignored, regenerate
  anytime.
- `seed_db.py` needs `DATABASE_URL` set (e.g. via `apps/gateway/.env`) and
  writes to the same Postgres database the three apps already share. It's
  always safe to re-run: patients are upserted by `external_id`, and each
  character's per-app record is updated in place rather than duplicated.
  **Exception:** `audit_events` are database-enforced append-only (see the
  `vitalhealth_prevent_audit_mutation` trigger in
  `vitalhealth_storage/store.py`) — re-seeding appends a fresh audit event
  each time instead of replacing the old one. That's intentional: it's the
  same thing a real re-assessment would do, not something this script tries
  to undo.
- `render_cheat_sheet.py` is stdlib-only and always safe to run with any
  Python. Run it after the compute scripts so `CHEAT_SHEET.md`'s "Expect"
  lines reflect real output; running it before they exist still produces a
  valid cheat sheet, just without the "Expect" lines.
- `sync_jace_seeds.py` is stdlib-only and writes
  `apps/Jace/static/demo_character_seeds.js`, which Jace's own frontend
  loads directly (`index.html`) to show a one-click "Demo characters" row
  in its Triage intake tab, alongside its existing built-in "Demo seeds"
  row. **This generated file is committed**, not gitignored — a fresh
  clone needs it present for Jace's UI to work, not only after someone
  thinks to run this script.

## Demoing Jace's note extractor

Each character also has a `clinician_note` — free text you can paste into
Jace's note box, or (after running `sync_jace_seeds.py`) load with one
click via the "Demo characters" row. Unlike everything else in this
pipeline, **this runs live against the real Gemini API each time** —
deliberately not pre-captured/pinned like Jace's own 10 built-in seeds
(those keep their pinned offline fallback in
`apps/Jace/sample/pinned_extractions.json`; the 8 characters' notes don't
have one). Make sure `GEMINI_API_KEY` is reachable in the environment Jace
runs in before demoing this part — `run_all.py` and standalone Jace both
just inherit it from your shell/profile if it's already set there.

## Editing the roster

Add/edit a character in `characters.py`, then re-run the steps above that
depend on what changed (e.g. only touched EMC symptoms → just
`compute_emc.py`, `seed_db.py`, `render_cheat_sheet.py`; only touched
`clinician_note` → just `sync_jace_seeds.py` and `render_cheat_sheet.py`).

`external_id` doubles as the shared-DB `patients.external_id` and YS's
`patient_id`, so keep it matching `[A-Za-z0-9-]{3,32}` and prefixed
`demo-` (makes demo rows easy to spot in the database:
`SELECT * FROM patients WHERE external_id LIKE 'demo-%'`).

## Scope note

The seeded EMC `output_payload` captures the ML diagnosis-classifier step
only (`run_prediction()` in `apps/YS/app.py`), not YS's LLM-drafted
certificate text or Jeslyn's RAG care plan — those are the most interactive
parts of a live demo anyway, so they belong in the cheat-sheet-driven live
walkthrough rather than baked into seed data (this also avoids burning
OpenRouter/Gemini API calls just to seed a database).
