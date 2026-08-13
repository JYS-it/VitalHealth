# VitalHealth

A monorepo of three independently runnable healthcare ML applications, unified
behind a single reverse-proxy gateway. Each app keeps its own dependencies,
its own virtual environment, and its own run command — they are **not**
merged into a single codebase and do not share a Python environment.

## Quickstart (new clone)

For login, dashboards, and demo accounts, PostgreSQL must also be running.
Before the first launch, copy `.env.example` to `.env` at the repository root
and set `DATABASE_URL` to your own local PostgreSQL connection. Each teammate
needs their own `.env`; it is deliberately ignored by Git and never arrives
with a clone. The default local pattern is:

```text
DATABASE_URL=postgresql+psycopg://vitalhealth:<your-postgres-password>@127.0.0.1:5432/vitalhealth
```

**Prerequisites:** Python 3.11 (tested; Jace also works on 3.9, its original
target — anything in that range is fine) and Git.

```bash
git clone <this-repo-url>
cd Integration
python run_all.py
```

That one command sets up everything on a fresh clone: it creates a venv for
each of the four processes, installs each one's `requirements.txt`, starts
Jace, Jeslyn, and YS, then the gateway once all three backends are reachable,
and opens **http://127.0.0.1:8080/** in your browser automatically. First run
takes a few minutes (installing dependencies); later runs start in seconds
since the venvs already exist.

When `DATABASE_URL` is configured, the launcher also creates the mock-patient
roster and Dr Alex Lee demo clinician in the shared database before starting
the web services. Missing demo model snapshots are generated on that first
run only; later launches synchronise the same records without duplicating
unchanged audit data.

Want the AI-drafted-text features (care plans, EMC drafts, note extraction),
not just the ML predictions? Add API keys before running — see
[Environment variables](#environment-variables) below; the predictions work
fine without them either way.

Stop everything with **Ctrl+C** in the terminal running `run_all.py` — it
tears down all four processes cleanly, including Flask's debug-reloader
child process (which orphans itself on its port if killed the naive way).

**Troubleshooting:**
- *Port already in use* — `run_all.py` checks all four ports up front and
  tells you exactly which app/port conflicts before doing anything; stop
  whatever's already listening (`netstat -ano | findstr :5000` on Windows)
  and re-run.
- *A dependency install fails* — that app's own `requirements.txt` is the
  source of truth; check its pinned versions (see
  [Why separate environments](#why-separate-environments)) against your
  Python version.
- Prefer to run one app at a time instead of the combined script? See
  [Manual / advanced](#manual--advanced-run-one-app-at-a-time) below.

## Applications

| App | Framework | Purpose |
| --- | --- | --- |
| [apps/Jace](apps/Jace) | FastAPI | CTRSE — clinical triage acuity prediction with clinician-note extraction |
| [apps/Jeslyn](apps/Jeslyn) | Flask | Stroke risk prediction with PDF-grounded (RAG) care plan generation |
| [apps/YS](apps/YS) | Flask | IIP-EMC Clinical Copilot — AI-drafted electronic medical certificates |
| [apps/gateway](apps/gateway) | FastAPI | Reverse proxy — the single public entry point that routes to the three apps above |

## Why separate environments

The apps pin conflicting versions of the same libraries, so a single shared
environment will break at least one of them:

| Package | Jace | Jeslyn | YS |
| --- | --- | --- | --- |
| scikit-learn | `==1.5.1` | `>=1.3.0` | `==1.6.1` |
| Flask / FastAPI | FastAPI `0.128.8` | Flask `>=3.0.0` | Flask `3.1.2` |
| GenAI SDK | `google-genai` | `google-generativeai` | `openai` (via OpenRouter) |

The `scikit-learn` pin in Jace is load-bearing: a version mismatch breaks
unpickling of `ctrse_p1p4_model.pkl`. Always install each app into its own
venv. This is also why "one web app" here means a **gateway in front of four
independent processes**, not one merged codebase.

## How the gateway works

`python run_all.py` (or `apps/gateway` started manually — see
[Manual / advanced](#manual--advanced-run-one-app-at-a-time)) puts a single
URL, `http://127.0.0.1:8080/`, in front of all three apps:

- `/triage/` → Jace
- `/stroke/` → Jeslyn
- `/emc/` → YS

The gateway forwards HTTP requests and contains no *clinical* business
logic — the one exception is login (see [Authentication](#authentication)
below), since it's the natural place for a single sign-on across all three
apps. See
[apps/gateway/main.py](apps/gateway/main.py) for how routing and prefix
headers work, and the `PrefixMiddleware` class near the top of
[apps/Jeslyn/app.py](apps/Jeslyn/app.py) and
[apps/YS/app.py](apps/YS/app.py) for how each Flask app generates
correctly-prefixed links when proxied (`url_for()` reads `SCRIPT_NAME`,
which the middleware sets from the gateway's `X-Forwarded-Prefix` header —
a no-op when that header is absent, so standalone runs are unaffected).

Jace's static frontend uses paths relative to its own page rather than
root-absolute ones (a deliberate, minimal change — see git history on
`apps/Jace/static/`), which is what lets it work both standalone at `/` and
proxied at `/triage/` with no backend routing changes.

## Authentication

The gateway owns login for all three apps: log in once at `/login` and the
session works for `/triage/`, `/stroke/`, and `/emc/` alike, since they're
all only reachable through the gateway's one origin
(`http://127.0.0.1:8080/`).

- `/register` — self-service signup (name, email, password, and a role).
- `/login` / `/logout` — session start/end.
- Sessions are a signed, expiring cookie (`itsdangerous`, 12-hour default),
  verified statelessly on every proxied request — the database is only
  queried at login and register, not on every request.
- Passwords are hashed with `bcrypt`.
- Unauthenticated browser requests to a proxied route redirect to `/login`;
  unauthenticated API-style requests get a `401` instead.
- **Caveat:** this only protects requests that go through the gateway.
  Running a backend standalone (see
  [Manual / advanced](#manual--advanced-run-one-app-at-a-time) below) is not
  authenticated — the same class of limitation as `X-Forwarded-Prefix`
  already being a no-op when a backend is run outside the gateway.

### Roles

There are two: **patient** and **clinician**. Both land on `/triage/`, which
renders a different dashboard for each (see
[Dashboards](#dashboards)).

- A **patient** account is linked to a row in `patients`, created at
  registration. The current patient surface is their read-only portal
  dashboard. Clinical triage, stroke assessment, EMC drafting, review, and
  issuance are rejected server-side until dedicated patient submission routes
  are introduced.
- A **clinician** account can read every patient's records. Registration
  therefore requires `CLINICIAN_ACCESS_CODE`, and **fails closed**: if that
  variable is unset, the Clinician option is rejected outright.
- Triage and stroke have no patient field on their forms, so a clinician
  picks a patient with **Work on** in the dashboard. That choice is a second
  signed cookie (`vh_subject`) which all three backends read, and it is
  cleared on logout.

Backends authenticate off the **signed session cookie**, which the proxy
forwards along with every other header. The `X-Vitalhealth-User` /
`-User-Email` / `-Role` headers are still sent, but only for logging — the
backends bind `127.0.0.1`, so anything running locally could forge a header,
whereas the cookie's signature cannot be forged without `SESSION_SECRET`.
That is also why all four processes must share one `SESSION_SECRET`;
`run_all.py` copies the gateway's value into the three backends for you.

### Dashboards

`/triage/` is the portal home for both roles, served by the Jace SPA
(`apps/Jace/static/dashboard.js`) and backed by read-only endpoints in
`apps/Jace/dashboard_api.py`. Authorisation is enforced there, server-side;
the SPA's role branch only picks a layout.

- **Patient** — one tile per module showing progress (*n* of 3), safe workflow
  status, their submitted fields, and a chronological activity list. Triage
  priority, stroke probabilities, model drivers, and clinician identities are
  never returned to this view.
- **Clinician** — every patient in one table with each module's latest result
  and last activity, a search box, a drill-down into one patient's full
  history, and an "Unassigned records" bucket for assessments run with no
  patient selected.

Tiles are formatted server-side in `apps/Jace/dashboard_summaries.py`, which
enforces patient redaction for all three modules. The gateway is the public
role boundary; each clinical backend also rejects requests that carry a
patient session cookie.

## Manual / advanced (run one app at a time)

`run_all.py` is the recommended path (see Quickstart above), but every app
can still be set up and launched on its own — useful for developing or
debugging one app in isolation, or in an environment where running four
processes from one script isn't wanted. Each app needs its own venv:

```bash
cd apps/<name>       # Jace, Jeslyn, YS, or gateway
python -m venv .venv && .venv/Scripts/activate   # Windows; use .venv/bin/activate on POSIX
pip install -r requirements.txt
```

Jace, Jeslyn, and YS each ship a `.env.example` — copy it to `.env` and fill
in real keys if you want the GenAI features (optional; see
[Environment variables](#environment-variables)). Note that only Jeslyn and
YS load `.env` themselves; Jace reads `GEMINI_API_KEY` straight from the
process environment, so export it in your shell before launching Jace
standalone (`run_all.py` handles this for you automatically).

### apps/Jace (FastAPI)

```bash
uvicorn api:app --reload
```

Serves the SPA at `http://127.0.0.1:8000/`. Tests: `pytest` from `apps/Jace`.

### apps/Jeslyn (Flask)

```bash
python app.py
```

Defaults to `http://127.0.0.1:5000/`.

### apps/YS (Flask)

```bash
PORT=5001 python app.py
```

Also defaults to port 5000, so set `PORT` if Jeslyn is already running.

### apps/gateway (FastAPI)

Copy `apps/gateway/.env.example` to `apps/gateway/.env` and set `DATABASE_URL`
first (required for login — see [Authentication](#authentication)), then:

```bash
uvicorn main:app --port 8080
```

Only useful once at least one backend is also running; otherwise every
proxied route 404s while the landing page at `/` still works.

## Shared PostgreSQL records

VitalHealth can write durable, cross-module records to one PostgreSQL database
without coupling the apps' model code. The shared schema contains `patients`,
`clinical_records`, `users`, and append-only `audit_events` tables. It stores:

- EMC workflow snapshots, approval/rejection events, and redacted audit data;
- stroke risk assessments and generated care plans; and
- triage assessments and clinician extraction corrections.

Every record carries two separate identities, because they answer different
questions and often point at different people:

- `patient_id` — who the record is **about** (the subject);
- `owner_user_id` — which account **produced** it (the actor).

For a patient running their own assessment these coincide. For a clinician
assessing someone else they do not, and the dashboards depend on the
distinction.

The three clinical apps remain usable without a database: `DATABASE_URL` is
optional for `apps/Jace/.env`, `apps/Jeslyn/.env`, and `apps/YS/.env`, and
predictions still work without it (persistence writes just no-op). Use the
same URL in all three files. Note that Jace additionally needs it to serve
the dashboards; `run_all.py` copies the gateway's value into any backend
that does not set its own.

The gateway is different: `DATABASE_URL` is a **hard requirement** for it
(`apps/gateway/.env`), because it uses the shared database for exactly one
thing — the `users` table backing login (see
[Authentication](#authentication) below). It still has no clinical-record
business logic and never touches `clinical_records` or `audit_events`.
Without `DATABASE_URL` set, nobody can log in and every proxied route
(`/triage/*`, `/stroke/*`, `/emc/*`) becomes permanently inaccessible, even
though the landing page at `/` still renders.

For a local PostgreSQL installation, add the same URL to the three existing
app `.env` files. Once `run_all.py` has built the app environments, initialise
or upgrade the schema after a storage-package update:

```powershell
apps\Jace\.venv\Scripts\python.exe -m vitalhealth_storage
```

This command is idempotent. It also creates the record-history indexes and
database-enforced append-only audit-event trigger. For a production deployment,
use a managed PostgreSQL service, encrypted connections, backups, and a proper
migration review before changing the schema.

## Demo data

`demo_data/` has an 8-character mock-patient roster for demonstrating all
three apps — a copy-paste cheat sheet (`demo_data/CHEAT_SHEET.md`) for
live-form demos, plus a script that seeds the same characters into the
shared database with real (not hand-guessed) model output. See
[demo_data/README.md](demo_data/README.md) for the run order.

## Environment variables

Each app reads its own secrets from the environment (or a local `.env`). No
keys are committed; `.env` is gitignored. Jace, Jeslyn, and YS each ship a
`.env.example` — copy it to `.env` and fill in real values. `run_all.py`
reads all three automatically (see [Manual / advanced](#manual--advanced-run-one-app-at-a-time)
for the one exception: Jace's own code doesn't load `.env` itself).

| Variable | Used by | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | Jace, Jeslyn | Same variable name, but two different SDKs — the apps are not interchangeable |
| `OPENROUTER_API_KEY` | YS | OpenRouter-hosted models via the `openai` client |
| `OPENROUTER_MODEL` | YS | Optional; defaults to `openai/gpt-4o-mini` |
| `SECRET_KEY` | Jeslyn | Flask session secret; defaults to a placeholder in development |
| `DATABASE_URL` | Jace, Jeslyn, YS, gateway | One shared PostgreSQL URL. Optional for the three clinical apps (durable clinical records/audit events); required for the gateway (the `users` table backing login) |
| `SESSION_SECRET` | gateway, Jace, Jeslyn, YS | Signs the login session cookie. **All four must share one value** — the gateway signs, the backends verify to decide whose record a result is. `run_all.py` copies the gateway's value into the backends; set it yourself if you start them by hand. Defaults to a placeholder in development — set a long random value in production |
| `CLINICIAN_ACCESS_CODE` | gateway | Required to register a clinician account. Unset means clinician registration is refused outright (clinicians can read every patient's records) |
| `SESSION_COOKIE_SECURE` | gateway | Optional; set to `true` once the gateway is served over HTTPS |
| `TRIAGE_UPSTREAM` / `STROKE_UPSTREAM` / `EMC_UPSTREAM` | gateway | Optional; override where each prefix proxies to (default `127.0.0.1:8000` / `:5000` / `:5001`) |
