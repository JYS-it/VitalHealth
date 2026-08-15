"""Reverse-proxy gateway — the single public entry point for all three apps.

This process forwards HTTP requests to whichever backend a path prefix names,
then relays the response back unchanged. Each backend keeps running as its own
process so the three apps can keep their own dependencies while sharing one URL.

Routing:
    /                          -> gateway landing page
    /login, /register, /logout -> session management
    /triage/*                  -> Jace   (FastAPI, CTRSE triage acuity)
    /stroke/*                  -> Jeslyn (Flask, stroke risk + care plan)
    /emc/*                     -> YS     (Flask, EMC copilot)

Authentication:
    The gateway owns login/session handling. Requests to /triage/*, /stroke/*,
    and /emc/* require a valid session cookie. The gateway forwards the logged-in
    user's identity using X-Vitalhealth-User and X-Vitalhealth-User-Email.
"""

from __future__ import annotations

import html
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from vitalhealth_storage import get_store

try:
    from sqlalchemy.exc import IntegrityError
except Exception:
    class IntegrityError(Exception):
        """Fallback when SQLAlchemy is not installed in the active gateway environment."""

        pass

import auth
from vitalhealth_storage import identity, load_shared_env, missing_shared_keys

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BASE_DIR / ".env")
# Same shared-settings resolution the three backends use, so the gateway
# signs cookies with the value they verify against no matter which .env a
# given machine actually has filled in.
load_shared_env()
for _key in missing_shared_keys():
    print(f"[gateway] WARNING: {_key} is not set - login and sessions will not work.")

app = FastAPI(title="VitalHealth gateway")

app.mount("/vh-assets", StaticFiles(directory=BASE_DIR / "static"), name="vh-assets")

SHARED_STORE = get_store()
DATABASE_ERROR: Optional[str] = None

try:
    SHARED_STORE.initialize()
except RuntimeError as exc:
    DATABASE_ERROR = str(exc)
except Exception as exc:
    DATABASE_ERROR = f"{type(exc).__name__}: {exc}"

BACKENDS = {
    "triage": os.environ.get("TRIAGE_UPSTREAM", "http://127.0.0.1:8000"),
    "stroke": os.environ.get("STROKE_UPSTREAM", "http://127.0.0.1:5000"),
    "emc": os.environ.get("EMC_UPSTREAM", "http://127.0.0.1:5001"),
}

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
}

VITA_FALLBACK = (
    "Vita is currently unavailable. Please try again later. "
    "VitalHealth outputs are educational decision-support only and should be reviewed "
    "by a qualified healthcare professional."
)


class VitaContext(BaseModel):
    active_section: Optional[str] = Field(default="dashboard")


class VitaChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    context: Optional[VitaContext] = None


def _vita_section_guidance(active_section: str) -> str:
    section = (active_section or "").strip().lower()

    if "stroke" in section:
        return "Focus on helping the user use Stroke Assessment features and result interpretation limits."
    if "emc" in section:
        return "Focus on helping the user use EMC workflow and where to find related tools."
    if "result" in section:
        return "Focus on helping the user interpret where results appear and what they mean at a high level."
    if "intake" in section:
        return "Focus on helping the user complete clinical triage intake fields and next steps."
    if "triage" in section:
        return "Focus on helping the user navigate triage workspace actions."

    return "Focus on guiding overall dashboard navigation and module selection."


def _vita_local_reply(message: str, active_section: str) -> str:
    text = (message or "").strip().lower()
    section = (active_section or "dashboard").strip().lower()

    if any(k in text for k in ["where", "open", "start", "go to", "navigate"]):
        if "stroke" in text or "stroke" in section:
            return (
                "Open Stroke Assessment from the top navigation, then complete the patient intake form and submit to view risk output. "
                "You can continue to Care Plan from the result area. "
                "This is educational decision-support, not a diagnosis."
            )

        if "emc" in text or "certificate" in text or "leave" in text or "emc" in section:
            return (
                "Open EMC Workflow from the top navigation, complete the intake fields, and submit to generate a clinician review draft. "
                "Use Approve or Reject actions after reviewing safety and policy checks. "
                "This is educational decision-support, not a diagnosis."
            )

        if "triage" in text or "intake" in text or "dashboard" in section:
            return (
                "From Home, choose Clinical Triage to begin intake or browse existing patient entries. "
                "Use the workspace controls to switch between browser, intake, and results views. "
                "This is educational decision-support, not a diagnosis."
            )

    if any(k in text for k in ["result", "risk", "score", "output"]):
        return (
            "Results indicate model-supported estimates and workflow status, and should be interpreted by a clinician in context. "
            "I can guide you to the relevant output section for each module. "
            "This is educational decision-support, not a diagnosis."
        )

    if any(k in text for k in ["diagnose", "diagnosis", "treat", "treatment", "medicine", "prescribe"]):
        return (
            "I can help with workflow navigation and output interpretation, but I cannot provide diagnosis or treatment instructions. "
            "Please consult a licensed clinician for medical decisions. "
            "This is educational decision-support, not a diagnosis."
        )

    return (
        "I can help you navigate Home, Clinical Triage, Stroke Assessment, or EMC Workflow and explain where each output appears. "
        "Tell me what step you are on and I will give concise next actions. "
        "This is educational decision-support, not a diagnosis."
    )


async def _vita_live_reply(message: str, active_section: str) -> Optional[str]:
    system_prompt = (
        "You are Vita, a concise assistant for the VitalHealth educational platform. "
        "Never diagnose, prescribe, or provide emergency directives. "
        "If asked for diagnosis/treatment, redirect to licensed clinicians. "
        "Keep responses to 2-4 short sentences. "
        "Always include this clause naturally once per reply: "
        "'This is educational decision-support, not a diagnosis.' "
        + _vita_section_guidance(active_section)
    )

    provider = (os.getenv("VITA_PROVIDER", "") or "").strip().lower()
    gemini_key = (os.getenv("VITA_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()

    should_try_gemini = provider in {"gemini", "google"} or (not provider and bool(gemini_key))

    if should_try_gemini and gemini_key:
        gemini_model = os.getenv("VITA_GEMINI_MODEL", "gemini-2.5-flash")
        gemini_base = os.getenv(
            "VITA_GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        )

        gemini_payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": message}]}],
            "generationConfig": {"temperature": 0.3},
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                gemini_resp = await client.post(
                    f"{gemini_base.rstrip('/')}/models/{gemini_model}:generateContent",
                    params={"key": gemini_key},
                    json=gemini_payload,
                )

            if gemini_resp.status_code < 400:
                gemini_data = gemini_resp.json()

                for candidate in gemini_data.get("candidates") or []:
                    content = candidate.get("content") or {}

                    for part in content.get("parts") or []:
                        text = part.get("text")

                        if isinstance(text, str) and text.strip():
                            return text.strip()

        except Exception:
            pass

        if provider in {"gemini", "google"}:
            return None

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("VITA_MODEL", "gpt-4o-mini")
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not api_key:
        return None

    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{api_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )

        if resp.status_code >= 400:
            return None

        data = resp.json()
        choices = data.get("choices") or []

        if not choices:
            return None

        content = ((choices[0] or {}).get("message") or {}).get("content")

        if not content or not isinstance(content, str):
            return None

        return content.strip()

    except Exception:
        return None


_client = httpx.AsyncClient(follow_redirects=False, timeout=30.0)

_SHARED_CSS = """
:root {
    --vh-blue: #1f6feb;
    --vh-blue-dark: #1557b0;
    --vh-orange: #f59e0b;
    --vh-orange-dark: #d97706;
    --vh-navy: #14213d;
    --vh-muted: #5f6f82;
    --vh-border: #d8e1ec;
    --vh-card: #ffffff;
    --vh-soft-blue: #eaf2ff;
    --vh-soft-orange: #fff4df;
}
* {
    box-sizing: border-box;
}
body {
    min-height: 100vh;
    margin: 0;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--vh-navy);
    background:
        radial-gradient(circle at top left, rgba(255, 244, 223, 0.95), transparent 34rem),
        linear-gradient(135deg, #f59e0b 0%, #ffb238 42%, #f7fbff 42%, #f5f7fa 100%);
}
.auth-page {
    min-height: 100vh;
    display: grid;
    place-items: center;
    padding: 2rem 1rem;
}
.auth-card {
    width: min(460px, 100%);
    background: var(--vh-card);
    border: 1px solid rgba(216, 225, 236, 0.95);
    border-radius: 28px;
    padding: 2.25rem;
    box-shadow: 0 24px 60px rgba(20, 33, 61, 0.18);
}
.auth-logo {
    width: min(260px, 82%);
    height: auto;
    display: block;
    margin: 0 auto 1.25rem;
    border-radius: 18px;
}
.auth-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: var(--vh-soft-orange);
    color: #9a5d00;
    font-weight: 700;
    font-size: 0.8rem;
    margin-bottom: 0.85rem;
}
h1 {
    margin: 0 0 0.4rem;
    font-size: clamp(1.9rem, 4vw, 2.4rem);
    line-height: 1.05;
}
p.sub {
    color: var(--vh-muted);
    margin: 0 0 1.5rem;
    line-height: 1.55;
}
form {
    margin-top: 1.25rem;
}
label {
    display: block;
    margin: 1rem 0 0.35rem;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--vh-navy);
}
input[type=email],
input[type=password] {
    width: 100%;
    padding: 0.8rem 0.9rem;
    border: 1.5px solid var(--vh-border);
    border-radius: 12px;
    background: #ffffff;
    color: var(--vh-navy);
    font-size: 1rem;
    outline: none;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
input[type=email]:focus,
input[type=password]:focus {
    border-color: var(--vh-blue);
    box-shadow: 0 0 0 4px rgba(31, 111, 235, 0.13);
}
button {
    width: 100%;
    margin-top: 1.45rem;
    padding: 0.85rem 1.25rem;
    border: none;
    border-radius: 14px;
    background: var(--vh-blue);
    color: #fff;
    font-size: 1rem;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 0 12px 28px rgba(31, 111, 235, 0.28);
    transition: transform 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
}
button:hover {
    background: var(--vh-blue-dark);
    transform: translateY(-1px);
    box-shadow: 0 16px 34px rgba(31, 111, 235, 0.33);
}
.error {
    background: #fdecec;
    color: #b91c1c;
    border: 1px solid #f4b4b4;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin: 1rem 0;
    font-size: 0.92rem;
}
.hint {
    margin: 1.25rem 0 0;
    font-size: 0.93rem;
    color: var(--vh-muted);
    text-align: center;
}
.hint a {
    color: var(--vh-blue);
    font-weight: 800;
}
.topbar {
    width: min(960px, calc(100% - 2rem));
    margin: 1.5rem auto 0;
    text-align: right;
    font-size: 0.9rem;
    color: var(--vh-muted);
}
.topbar a {
    color: var(--vh-blue);
    font-weight: 800;
}
.landing-page {
    min-height: 100vh;
    padding: 1.5rem 1rem 3rem;
}
.landing-shell {
    width: min(900px, 100%);
    margin: 2rem auto 0;
}
.landing-card {
    background: var(--vh-card);
    border: 1px solid var(--vh-border);
    border-radius: 28px;
    padding: 2rem;
    box-shadow: 0 24px 60px rgba(20, 33, 61, 0.12);
}
.landing-logo {
    width: min(300px, 82%);
    height: auto;
    display: block;
    margin-bottom: 1rem;
    border-radius: 18px;
}
.workflow-links {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1rem;
    margin-top: 1.5rem;
}
.card {
    display: block;
    border: 1px solid var(--vh-border);
    border-radius: 18px;
    padding: 1.2rem;
    text-decoration: none;
    color: inherit;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(20, 33, 61, 0.06);
}
.card:hover {
    border-color: var(--vh-blue);
}
.card h2 {
    margin: 0 0 0.35rem 0;
    font-size: 1.1rem;
}
.card p {
    margin: 0;
    color: var(--vh-muted);
    font-size: 0.95rem;
    line-height: 1.5;
}
.footer {
    margin-top: 2rem;
    font-size: 0.82rem;
    color: var(--vh-muted);
    border-top: 1px solid var(--vh-border);
    padding-top: 1rem;
}
.role-choice {
    border: 1px solid var(--vh-border);
    border-radius: 14px;
    padding: 0.9rem 1rem 1rem;
    margin: 1.1rem 0 0.4rem;
}
.role-choice legend {
    padding: 0 0.4rem;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--vh-muted);
}
.role-option {
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
    margin: 0.55rem 0 0;
    font-weight: 400;
    line-height: 1.4;
    cursor: pointer;
}
.role-option input {
    width: auto;
    margin: 0.25rem 0 0;
    flex: none;
}
.role-option span {
    font-size: 0.9rem;
    color: var(--vh-muted);
}
.role-option strong {
    color: inherit;
    font-size: 0.95rem;
}
.hint--inline {
    margin-top: 0.35rem;
    font-size: 0.8rem;
}
@media (max-width: 760px) {
    .workflow-links {
        grid-template-columns: 1fr;
    }
    .auth-card,
    .landing-card {
        padding: 1.5rem;
        border-radius: 22px;
    }
}
"""


def _authenticated_user(request: Request) -> dict | None:
    return auth.read_session_token(request.cookies.get(auth.COOKIE_NAME))


def _current_actor(request: Request) -> identity.Actor | None:
    return identity.actor_from_cookies(request.cookies)


def _forbidden_for_role(request: Request, message: str) -> Response:
    """Deny a role before the request reaches an upstream clinical app."""
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/triage/", status_code=303)
    return JSONResponse({"detail": message}, status_code=403)


_PATIENT_SUBMIT_FIRST_SEGMENTS = {"submit", "submitted", "status", "download", "static"}


def _patient_may_use_proxy_path(prefix: str, path: str) -> bool:
    """What a patient account may reach through the proxy.

    Triage's clinical *workspace* stays clinician-only and instant — out of
    scope for the review workflow. A patient may instead reach the
    self-check surface: its own static page plus its two explicit API
    routes (`api/self-check`, `api/self-check/options`), which run the same
    model but return only a patient-safe projection (see
    ctrse_core.patient_view) and persist as a distinct, non-queued status.
    Stroke and EMC allow exactly three patient-facing routes each:
    self-submission, its waiting page, and the `status/` poll target that
    waiting page reads to reveal a result once a clinician approves it. Each
    Flask app's own `static/` folder (CSS/JS) is allowed too so those pages
    don't render unstyled — it's Flask's built-in static file serving, no
    clinical data lives there. Everything else, including the
    review/edit/approve pages, stays clinician-only. Jace's static assets are
    allowed so the portal can load, while only its explicit patient-safe
    dashboard and self-check APIs can be called.
    """
    normalized = path.strip("/")
    if prefix == "triage":
        if not normalized.startswith("api/"):
            return True
        return normalized.startswith("api/dashboard/") or normalized in (
            "api/self-check", "api/self-check/options",
        )
    if prefix in ("stroke", "emc"):
        first_segment = normalized.split("/", 1)[0] if normalized else ""
        return first_segment in _PATIENT_SUBMIT_FIRST_SEGMENTS
    return False


def _patient_pending_destination(actor: identity.Actor | None, prefix: str) -> str | None:
    """Return a patient's newest pending request for a module, if any.

    A patient returning to a module should resume an open request rather than
    receive a blank form that can create a duplicate submission. The backend
    still verifies record ownership before disclosing its status or outcome.
    """
    if actor is None or not actor.is_patient or not SHARED_STORE.enabled:
        return None
    try:
        pending = SHARED_STORE.list_records(
            owner_user_id=actor.user_id,
            source_app=prefix,
            status="PENDING_REVIEW",
            limit=1,
        )
    except Exception:
        return None
    if not pending:
        return None
    return f"/{prefix}/submitted/{pending[0]['id']}"


def _name_from_email(email: str) -> str:
    """Fallback display name for accounts registered before names were asked for."""
    local_part = str(email or "").split("@", 1)[0]
    return local_part.replace(".", " ").replace("_", " ").replace("-", " ").strip().title()


def _safe_next_path(next_path: str) -> str:
    if next_path.startswith("/") and not next_path.startswith("//"):
        return next_path

    return "/triage/"


def _database_error_message() -> str | None:
    if not DATABASE_ERROR:
        return None

    return (
        "Shared database is not configured. Please check DATABASE_URL, PostgreSQL, "
        "and whether the VitalHealth database tables have been initialised."
    )


def _render_landing(user: dict | None) -> str:
    if user:
        topbar = f'Logged in as {html.escape(user["email"])} &middot; <a href="/logout">Log out</a>'
    else:
        topbar = '<a href="/login">Log in</a> &middot; <a href="/register">Register</a>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VitalHealth</title>
  <link rel="icon" type="image/png" href="/vh-assets/brand/login.png">
  <style>{_SHARED_CSS}</style>
</head>

<body class="landing-page">
  <div class="topbar">{topbar}</div>

  <main class="landing-shell">
    <section class="landing-card">
      <img class="landing-logo" src="/vh-assets/brand/login.png" alt="VitalHealth">

      <span class="auth-kicker">Integrated healthcare workflow support</span>
      <h1>Welcome to VitalHealth</h1>
      <p class="sub">
        Access clinical triage, stroke assessment, and electronic medical certificate workflows from one portal.
      </p>

      <div class="workflow-links">
        <a class="card" href="/triage/">
          <h2>Clinical Triage</h2>
          <p>Assess triage priority from intake details and clinician notes.</p>
        </a>

        <a class="card" href="/stroke/prediction">
          <h2>Stroke Assessment</h2>
          <p>Estimate stroke risk and generate educational care planning guidance.</p>
        </a>

        <a class="card" href="/emc/">
          <h2>EMC Workflow</h2>
          <p>Prepare electronic medical certificate drafts for clinician review.</p>
        </a>
      </div>

      <p class="footer">
        VitalHealth is an educational clinical decision-support prototype. It does not provide a medical diagnosis and does not replace review by a qualified healthcare professional.
      </p>
    </section>
  </main>
</body>
</html>
"""


def _render_login(error: str | None, next_path: str, email: str) -> str:
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    next_field = html.escape(next_path)
    email_value = html.escape(email)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Log in - VitalHealth</title>
  <link rel="icon" type="image/png" href="/vh-assets/brand/login.png">
  <style>{_SHARED_CSS}</style>
</head>

<body class="auth-page">
  <main class="auth-card">
    <img class="auth-logo" src="/vh-assets/brand/login.png" alt="VitalHealth">

    <span class="auth-kicker">Secure sign in</span>
    <h1>Welcome back</h1>
    <p class="sub">Sign in to continue to VitalHealth workflows.</p>

    {error_html}

    <form method="post" action="/login">
      <input type="hidden" name="next" value="{next_field}">

      <label for="email">Email</label>
      <input type="email" id="email" name="email" value="{email_value}" required autofocus>

      <label for="password">Password</label>
      <input type="password" id="password" name="password" required>

      <button type="submit">Log in</button>
    </form>

    <p class="hint">No account yet? <a href="/register">Register</a></p>
  </main>
</body>
</html>
"""


def _render_register(
    error: str | None,
    email: str,
    display_name: str = "",
    role: str = identity.ROLE_PATIENT,
) -> str:
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    email_value = html.escape(email)
    name_value = html.escape(display_name)
    clinician_selected = " checked" if role == identity.ROLE_CLINICIAN else ""
    patient_selected = "" if role == identity.ROLE_CLINICIAN else " checked"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Register - VitalHealth</title>
  <link rel="icon" type="image/png" href="/vh-assets/brand/login.png">
  <style>{_SHARED_CSS}</style>
</head>

<body class="auth-page">
  <main class="auth-card">
    <img class="auth-logo" src="/vh-assets/brand/login.png" alt="VitalHealth">

    <span class="auth-kicker">Patient and clinician access</span>
    <h1>Create account</h1>
    <p class="sub">Register to access VitalHealth workflows.</p>

    {error_html}

    <form method="post" action="/register">
      <label for="display_name">Full name</label>
      <input type="text" id="display_name" name="display_name" value="{name_value}" required maxlength="120" autofocus>

      <label for="email">Email</label>
      <input type="email" id="email" name="email" value="{email_value}" required>

      <label for="password">Password</label>
      <input type="password" id="password" name="password" required minlength="8">

      <label for="confirm">Confirm password</label>
      <input type="password" id="confirm" name="confirm" required minlength="8">

      <fieldset class="role-choice">
        <legend>This account is for</legend>

        <label class="role-option">
          <input type="radio" name="role" value="patient"{patient_selected}>
          <span><strong>Patient</strong><br>See your own assessments and care guidance.</span>
        </label>

        <label class="role-option">
          <input type="radio" name="role" value="clinician"{clinician_selected}>
          <span><strong>Clinician</strong><br>Review every patient's records. Requires an access code.</span>
        </label>
      </fieldset>

      <label for="clinician_code">Clinician access code</label>
      <input type="password" id="clinician_code" name="clinician_code" autocomplete="off">
      <p class="hint hint--inline">Leave blank when registering as a patient.</p>

      <button type="submit">Register</button>
    </form>

    <p class="hint">Already have an account? <a href="/login">Log in</a></p>
  </main>
</body>
</html>
"""


def _cookie_is_secure() -> bool:
    return os.environ.get("SESSION_COOKIE_SECURE", "").lower() == "true"


def _set_session_cookie(
    response: Response,
    *,
    user_id: str,
    email: str,
    role: str,
    patient_external_id: str | None = None,
    display_name: str | None = None,
) -> None:
    token = auth.create_session_token(
        user_id=user_id,
        email=email,
        role=role,
        patient_external_id=patient_external_id,
        display_name=display_name,
    )

    response.set_cookie(
        auth.COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=_cookie_is_secure(),
        max_age=auth.SESSION_MAX_AGE_SECONDS,
        path="/",
    )


def _patient_external_id_for(user: Any) -> str | None:
    """The patients row a login speaks for, or None for clinicians."""
    patient_id = user.get("patient_id") if hasattr(user, "get") else None
    if not patient_id:
        return None
    try:
        patient = SHARED_STORE.get_patient(patient_id)
    except Exception:
        return None
    return patient["external_id"] if patient else None


@app.get("/")
async def landing(request: Request):
    user = _authenticated_user(request)

    if user:
        return RedirectResponse(url="/triage/", status_code=303)

    return HTMLResponse(_render_landing(user))


@app.get("/login")
async def login_form(request: Request) -> HTMLResponse:
    error = request.query_params.get("error", "")
    next_path = _safe_next_path(request.query_params.get("next", "/triage/"))
    email = request.query_params.get("email", "")

    return HTMLResponse(_render_login(error or None, next_path, email))


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()

    email = str(form.get("email", "")).strip()
    password = str(form.get("password", ""))
    next_path = _safe_next_path(str(form.get("next", "") or "/triage/"))

    db_error = _database_error_message()

    if db_error:
        query = f"error={quote(db_error)}&next={quote(next_path)}&email={quote(email)}"
        return RedirectResponse(url=f"/login?{query}", status_code=303)

    try:
        user = SHARED_STORE.get_user_by_email(email) if email else None
    except Exception:
        query = f"error={quote('Could not connect to the shared database.')}&next={quote(next_path)}&email={quote(email)}"
        return RedirectResponse(url=f"/login?{query}", status_code=303)

    if not user or not auth.verify_password(password, user["password_hash"]):
        query = f"error={quote('Incorrect email or password.')}&next={quote(next_path)}&email={quote(email)}"
        return RedirectResponse(url=f"/login?{query}", status_code=303)

    response = RedirectResponse(url=next_path, status_code=303)
    _set_session_cookie(
        response,
        user_id=user["id"],
        email=user["email"],
        role=identity.normalise_role(user["role"]),
        patient_external_id=_patient_external_id_for(user),
        display_name=user["display_name"],
    )

    return response


@app.get("/register")
async def register_form(request: Request) -> HTMLResponse:
    error = request.query_params.get("error", "")
    email = request.query_params.get("email", "")
    display_name = request.query_params.get("display_name", "")
    role = identity.normalise_role(request.query_params.get("role", ""))

    return HTMLResponse(_render_register(error or None, email, display_name, role))


@app.post("/register")
async def register_submit(request: Request):
    form = await request.form()

    email = str(form.get("email", "")).strip()
    password = str(form.get("password", ""))
    confirm = str(form.get("confirm", ""))
    display_name = str(form.get("display_name", "")).strip()
    role = identity.normalise_role(form.get("role", ""))
    clinician_code = str(form.get("clinician_code", "")).strip()

    def fail(message: str):
        query = (
            f"error={quote(message)}&email={quote(email)}"
            f"&display_name={quote(display_name)}&role={quote(role)}"
        )
        return RedirectResponse(url=f"/register?{query}", status_code=303)

    db_error = _database_error_message()

    if db_error:
        return fail(db_error)

    if not display_name:
        return fail("Enter your full name.")

    if len(display_name) > 120:
        return fail("Name must be 120 characters or fewer.")

    if not email or "@" not in email:
        return fail("Enter a valid email address.")

    if len(password) < 8:
        return fail("Password must be at least 8 characters.")

    if password != confirm:
        return fail("Passwords do not match.")

    # A clinician account can read every patient's records, so this gate fails
    # closed: with no CLINICIAN_ACCESS_CODE configured, the role is simply not
    # available for self-service registration.
    if role == identity.ROLE_CLINICIAN:
        expected_code = os.environ.get("CLINICIAN_ACCESS_CODE", "").strip()

        if not expected_code:
            return fail("Clinician registration is not enabled on this deployment.")

        if clinician_code != expected_code:
            return fail("That clinician access code is not valid.")

    patient_external_id = None
    patient_id = None

    try:
        if role == identity.ROLE_PATIENT:
            # Doubles as YS's patient_id, which only accepts [A-Za-z0-9-]{3,32}
            # (apps/YS/app.py), so this must not be derived from the email.
            patient_external_id = f"u-{uuid.uuid4().hex[:16]}"
            patient_id = SHARED_STORE.upsert_patient(patient_external_id, display_name)

        user_id = SHARED_STORE.create_user(
            email=email,
            password_hash=auth.hash_password(password),
            role=role,
            display_name=display_name,
            patient_id=patient_id,
        )
    except IntegrityError:
        return fail("Email already registered.")
    except Exception:
        return fail("Could not create the account. Please check the shared database setup.")

    response = RedirectResponse(url="/triage/", status_code=303)
    _set_session_cookie(
        response,
        user_id=user_id,
        email=email.strip().lower(),
        role=role,
        patient_external_id=patient_external_id,
        display_name=display_name,
    )

    return response


@app.api_route("/logout", methods=["GET", "POST"])
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    # Otherwise the next clinician to sign in on this browser inherits whichever
    # patient the previous one had selected.
    response.delete_cookie(auth.SUBJECT_COOKIE_NAME, path="/")

    return response

@app.get("/api/me")
async def current_user(request: Request):
    actor = _current_actor(request)

    if not actor:
        return {
            "authenticated": False,
            "email": None,
            "display_name": "guest",
            "role": None,
            "patient_external_id": None,
        }

    return {
        "authenticated": True,
        "email": actor.email,
        # Registered names beat the email-prefix guess, which is only a
        # fallback for accounts created before names were collected.
        "display_name": actor.display_name or _name_from_email(actor.email) or "user",
        "role": actor.role,
        "patient_external_id": actor.patient_external_id,
    }


@app.post("/api/vita/chat")
async def vita_chat(payload: VitaChatRequest):
    message = payload.message.strip()

    if not message:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "data": {"reply": VITA_FALLBACK},
                "error": "Message is required",
            },
        )

    active_section = "dashboard"

    if payload.context and payload.context.active_section:
        active_section = payload.context.active_section

    live_reply = await _vita_live_reply(
        message=message,
        active_section=active_section,
    )

    if live_reply:
        return {
            "success": True,
            "data": {"reply": live_reply},
            "error": None,
        }

    local_reply = _vita_local_reply(
        message=message,
        active_section=active_section,
    )

    return {
        "success": True,
        "data": {"reply": local_reply or VITA_FALLBACK},
        "error": None,
    }


class ClinicianContextRequest(BaseModel):
    patient_external_id: Optional[str] = None
    display_name: Optional[str] = None


# Declared above the /{prefix} catch-alls below: Starlette matches routes in
# registration order, so anything added after them is swallowed by the proxy.
@app.post("/api/clinician/context")
async def set_clinician_context(payload: ClinicianContextRequest, request: Request):
    """Choose which patient a clinician's next assessment is about.

    Neither the triage nor the stroke form has a patient field, so without a
    selected subject a clinician's assessments could never be filed against
    anyone. The choice lives in a signed cookie rather than a form field so all
    three backends read it the same way and none of them can be lied to.
    """
    actor = _current_actor(request)

    if actor is None:
        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    if not actor.is_clinician:
        return JSONResponse({"detail": "Clinician access required"}, status_code=403)

    external_id = (payload.patient_external_id or "").strip()
    response = JSONResponse({
        "patient_external_id": external_id or None,
        "display_name": payload.display_name if external_id else None,
    })

    if not external_id:
        response.delete_cookie(auth.SUBJECT_COOKIE_NAME, path="/")
        return response

    response.set_cookie(
        auth.SUBJECT_COOKIE_NAME,
        auth.create_subject_token(
            external_id=external_id,
            display_name=payload.display_name,
        ),
        httponly=True,
        samesite="lax",
        secure=_cookie_is_secure(),
        max_age=auth.SESSION_MAX_AGE_SECONDS,
        path="/",
    )

    return response


@app.api_route("/{prefix}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def redirect_bare_prefix(prefix: str, request: Request):
    if prefix == "stroke":
        # Bare /stroke used to always mean "the clinician-instant form," but
        # that route is clinician-only now — a patient landing here needs the
        # submission form instead, or they'd just bounce off a 403.
        actor = _current_actor(request)
        target = _patient_pending_destination(actor, "stroke") or "/stroke/submit"
        if not actor or not actor.is_patient:
            target = "/stroke/prediction"
        return RedirectResponse(url=target, status_code=307)

    if prefix in BACKENDS:
        if prefix == "emc":
            actor = _current_actor(request)
            if actor and actor.is_patient:
                target = _patient_pending_destination(actor, "emc") or "/emc/submit"
                return RedirectResponse(url=target, status_code=307)
        return RedirectResponse(url=f"/{prefix}/", status_code=307)

    return Response(status_code=404)


@app.api_route("/{prefix}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(prefix: str, path: str, request: Request):
    upstream = BACKENDS.get(prefix)

    if upstream is None:
        return Response(status_code=404)

    actor = _current_actor(request)

    # The nav links to /stroke/ and /emc/ (trailing slash — a distinct route
    # from bare /stroke and /emc, which redirect_bare_prefix() handles) always
    # meant "the clinician-instant entry point" until patient submission
    # existed. For a patient that now clinician-only landing 403s and bounces
    # them back to the dashboard with no explanation — send them to the
    # submission form instead, same as redirect_bare_prefix() already does
    # for the bare form.
    if prefix == "stroke" and path.strip("/") == "":
        target = _patient_pending_destination(actor, "stroke") or "/stroke/submit"
        if not actor or not actor.is_patient:
            target = "/stroke/prediction"
        return RedirectResponse(url=target, status_code=307)

    if prefix == "emc" and path.strip("/") == "" and actor and actor.is_patient:
        target = _patient_pending_destination(actor, "emc") or "/emc/submit"
        return RedirectResponse(url=target, status_code=307)

    if actor is None:
        if "text/html" in request.headers.get("accept", ""):
            next_path_value = f"/{prefix}/{path}"
            if request.query_params:
                next_path_value += f"?{request.query_params}"

            next_path = quote(next_path_value)
            return RedirectResponse(url=f"/login?next={next_path}", status_code=303)

        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    if actor.is_patient and not _patient_may_use_proxy_path(prefix, path):
        return _forbidden_for_role(
            request,
            "This workflow is available to clinician accounts only.",
        )

    url = f"{upstream}/{path}"

    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    forward_headers["X-Forwarded-Prefix"] = f"/{prefix}"
    forward_headers["X-Forwarded-Host"] = request.headers.get("host", "")
    forward_headers["X-Forwarded-Proto"] = request.url.scheme
    # Useful in backend logs, but NOT the basis for any authorisation decision.
    # The backends bind 127.0.0.1, so anything running locally can forge these;
    # they authenticate off the signed vh_session cookie (forwarded above with
    # the rest of the headers) instead.
    forward_headers["X-Vitalhealth-User"] = actor.user_id
    forward_headers["X-Vitalhealth-User-Email"] = actor.email
    forward_headers["X-Vitalhealth-Role"] = actor.role

    body = await request.body()

    request_timeout = None
    if prefix == "stroke" and path.strip("/").startswith(("care-plan", "submit")):
        # Care-plan generation, and now patient self-submission (which
        # generates the care plan eagerly too), both call an LLM and can take
        # longer than the default timeout.
        request_timeout = httpx.Timeout(180.0, connect=10.0, read=180.0)

    try:
        upstream_response = await _client.request(
            request.method,
            url,
            params=request.query_params,
            headers=forward_headers,
            content=body,
            timeout=request_timeout,
        )
    except httpx.RequestError:
        if "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(
                "<h1>Service temporarily unavailable</h1>"
                "<p>The selected workflow service is currently unreachable. "
                "Please try again shortly.</p>",
                status_code=502,
            )

        return JSONResponse(
            {"detail": "Upstream workflow service is currently unavailable."},
            status_code=502,
        )

    response_headers = {
        k: v for k, v in upstream_response.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    content = upstream_response.content
    content_type = upstream_response.headers.get("content-type", "").lower()

    if prefix == "stroke" and "text/html" in content_type:
        stroke_html = content.decode("utf-8", errors="replace")

        stroke_html = re.sub(
            r'(href|src|action)="/(?!(?:stroke/|triage/|emc/|api/|vh-assets/|"))',
            r'\1="/stroke/',
            stroke_html,
        )

        content = stroke_html.encode("utf-8")

    if prefix == "emc" and "text/html" in content_type:
        emc_html = content.decode("utf-8", errors="replace")

        if "brand-logo" not in emc_html:
            emc_html = re.sub(
                r'<a\s+class="brand"\s+href="/triage/">\s*VitalHealth\s*</a>',
                '<a class="brand brand-logo" href="/triage/" aria-label="VitalHealth Home">'
                '<img src="/vh-assets/brand/vitalhealth-logo-full.png" alt="VitalHealth">'
                '</a>',
                emc_html,
                flags=re.IGNORECASE,
            )

        emc_html = re.sub(
            r'\s*<a\s+href="/stroke/care-plan">\s*Results\s*</a>\s*',
            "\n",
            emc_html,
            flags=re.IGNORECASE,
        )

        emc_html = emc_html.replace(
            'href="/emc/static/styles.css"',
            'href="/emc/static/styles.css?v=vh-emc-ui-20260811"',
        )

        loading_patch = """
<script id="vh-emc-loading-patch">
(() => {
    const forms = document.querySelectorAll('.workflow-form');
    if (!forms.length) return;

    forms.forEach((form) => {
        form.addEventListener('submit', () => {
            if (!form.checkValidity()) return;

            const overlay = document.getElementById('loading-overlay');
            if (overlay) overlay.hidden = false;

            const button = form.querySelector('button[type="submit"]');
            if (!button) return;

            const original = button.dataset.loadingText || button.textContent.trim();
            const loadingText = /reject/i.test(original) ? 'Processing rejection...' : 'Generating...';

            button.dataset.loadingText = original;
            button.textContent = loadingText;
            button.classList.add('btn-loading', 'is-generating');
            button.disabled = true;
            button.setAttribute('aria-busy', 'true');
        }, { once: true });
    });
})();
</script>
"""

        if "vh-emc-loading-patch" not in emc_html and "</body>" in emc_html:
            emc_html = emc_html.replace("</body>", f"{loading_patch}\n</body>")

        content = emc_html.encode("utf-8")

    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
