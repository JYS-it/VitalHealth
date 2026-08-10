# VitalHealth

A monorepo of three independently runnable healthcare ML applications, unified
behind a single reverse-proxy gateway. Each app keeps its own dependencies,
its own virtual environment, and its own run command — they are **not**
merged into a single codebase and do not share a Python environment.

## Quickstart (new clone)

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

The gateway only forwards HTTP requests; it contains no business logic. See
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

```bash
uvicorn main:app --port 8080
```

Only useful once at least one backend is also running; otherwise every
proxied route 404s while the landing page at `/` still works.

## Shared PostgreSQL records

VitalHealth can write durable, cross-module records to one PostgreSQL database
without coupling the apps' model code. The shared schema contains `patients`,
`clinical_records`, and append-only `audit_events` tables. It stores:

- EMC workflow snapshots, approval/rejection events, and redacted audit data;
- stroke risk assessments and generated care plans; and
- triage assessments and clinician extraction corrections.

The apps remain usable without a database. Set `DATABASE_URL` in each of
`apps/Jace/.env`, `apps/Jeslyn/.env`, and `apps/YS/.env` to enable persistent
records. Use the same URL in all three files. Do not put it in the gateway;
the gateway has no clinical business logic and does not access records.

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
| `DATABASE_URL` | Jace, Jeslyn, YS | One shared PostgreSQL URL for durable clinical records and audit events |
| `TRIAGE_UPSTREAM` / `STROKE_UPSTREAM` / `EMC_UPSTREAM` | gateway | Optional; override where each prefix proxies to (default `127.0.0.1:8000` / `:5000` / `:5001`) |
