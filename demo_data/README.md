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

For normal local development, simply run `py -3.12 run_all.py` from the
repository root. Once the four environments are ready, the launcher creates
missing output snapshots and runs `seed_db.py` automatically whenever
`DATABASE_URL` is configured. The commands below remain useful when you want
to regenerate model snapshots deliberately after changing the demo roster or
a model.

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
  always safe to re-run: patients are upserted by `external_id`, users by
  email, and each character's per-app record is updated in place rather than
  duplicated. It must run with the **gateway's** venv, which is the only one
  carrying `bcrypt` (needed to hash the demo logins); the script says so and
  exits cleanly if you use another.
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
  in its Triage intake tab. That row is now the *only* seed row — Jace's
  built-in "Demo seeds" row was removed from the UI, so these 8 characters
  are the one-click demo path. **This generated file is committed**, not gitignored — a fresh
  clone needs it present for Jace's UI to work, not only after someone
  thinks to run this script.

## Demo logins

`seed_db.py` also creates a login for every character plus one clinician, and
marks each character's seeded records as owned by their own account. That
means both dashboards have real content the first time you log in, instead of
being empty until someone re-runs all three modules by hand.

| Account | Email | Password |
| --- | --- | --- |
| Patient (×8) | `<first>.<last>@demo.vitalhealth.local` — e.g. `grace.lim@demo.vitalhealth.local` | `demo1234` |
| Clinician | `dr.alex.lee@demo.vitalhealth.local` | `demo1234` |

Emails are derived from `external_id` by `demo_email()` in `characters.py`,
so the roster stays the single source of truth. `.local` is reserved by
RFC 6762 and cannot resolve publicly, so these can never reach a real mailbox.

Log in as a character to see the patient dashboard already at 3 of 3 —
**Timothy Ng** is the interesting one, since he is 15 and outside the triage
model's validated range, so his triage tile reads "Model declined to assess"
rather than showing a priority. Log in as the clinician to see all eight in
one table.

These are demo credentials in a gitignored seed script; they are not a
production auth story.

## Demoing Jace's note extractor

Each character also has a `clinician_note` — free text you can paste into
Jace's note box, or (after running `sync_jace_seeds.py`) load with one
click via the "Demo characters" row. Unlike everything else in this
pipeline, **this runs live against the real Gemini API each time** —
deliberately not pre-captured/pinned like Jace's own 10 built-in seed
notes (those keep their pinned offline fallback in
`apps/Jace/sample/pinned_extractions.json`, generated from — and now
solely owned by — `apps/Jace/prep_pinned_extractions.py`; they no longer
have a button row in the UI, so paste them if you need them. The 8
characters' notes have no pinned fallback). Make sure `GEMINI_API_KEY` is reachable in the environment Jace
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
