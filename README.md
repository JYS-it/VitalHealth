# VitalHealth

A monorepo of three independently runnable healthcare ML applications, unified
behind a single reverse-proxy gateway. Each app keeps its own dependencies,
its own virtual environment, and its own run command — they are **not**
merged into a single codebase and do not share a Python environment.

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

## Running behind the gateway (recommended)

Open four terminals, one per process, each `cd`'d into that app's directory
with its venv activated:

```bash
# Terminal 1 — Jace (FastAPI), port 8000
cd apps/Jace && uvicorn api:app --host 127.0.0.1 --port 8000

# Terminal 2 — Jeslyn (Flask), port 5000
cd apps/Jeslyn && python app.py

# Terminal 3 — YS (Flask), port 5001
cd apps/YS && PORT=5001 python app.py

# Terminal 4 — gateway, port 8080 (the one URL users hit)
cd apps/gateway && uvicorn main:app --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080/` — a landing page links to:

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

## Running each app standalone

Every app can still be launched on its own, exactly as before the gateway
existed — useful for developing or debugging one app in isolation.

### apps/Jace (FastAPI)

```bash
cd apps/Jace
python -m venv .venv && .venv/Scripts/activate   # Windows; use .venv/bin/activate on POSIX
pip install -r requirements.txt
uvicorn api:app --reload
```

Serves the SPA at `http://127.0.0.1:8000/`. Tests: `pytest` from `apps/Jace`.

### apps/Jeslyn (Flask)

```bash
cd apps/Jeslyn
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python app.py
```

Defaults to `http://127.0.0.1:5000/`.

### apps/YS (Flask)

```bash
cd apps/YS
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
PORT=5001 python app.py
```

Also defaults to port 5000, so set `PORT` if Jeslyn is already running.

### apps/gateway (FastAPI)

```bash
cd apps/gateway
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
uvicorn main:app --port 8080
```

Only useful once at least one backend is also running; otherwise every
proxied route 404s while the landing page at `/` still works.

## Environment variables

Each app reads its own secrets from the environment (or a local `.env`). No
keys are committed; `.env` is gitignored.

| Variable | Used by | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | Jace, Jeslyn | Same variable name, but two different SDKs — the apps are not interchangeable |
| `OPENROUTER_API_KEY` | YS | OpenRouter-hosted models via the `openai` client |
| `OPENROUTER_MODEL` | YS | Optional; defaults to `openai/gpt-4o-mini` |
| `SECRET_KEY` | Jeslyn | Flask session secret; defaults to a placeholder in development |
| `TRIAGE_UPSTREAM` / `STROKE_UPSTREAM` / `EMC_UPSTREAM` | gateway | Optional; override where each prefix proxies to (default `127.0.0.1:8000` / `:5000` / `:5001`) |
