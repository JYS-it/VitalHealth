"""Reverse-proxy gateway — the single public entry point for all three apps.

This process forwards HTTP requests to whichever backend a path prefix names,
then relays the response back unchanged. Each backend keeps running as its own
process so the three apps can keep their own dependencies while sharing one URL.

Routing:
    /                         -> gateway landing page
    /login, /register, /logout -> session management
    /triage/*                  -> Jace   (FastAPI, CTRSE triage acuity)
    /stroke/*                  -> Jeslyn (Flask, stroke risk + care plan)
    /emc/*                     -> YS     (Flask, EMC copilot)

Authentication:
    The gateway owns login/session handling. Requests to /triage/*, /stroke/*,
    and /emc/* require a valid session cookie. The gateway forwards the logged-in
    user's identity using X-Vitalhealth-User and X-Vitalhealth-User-Email.
"""

import html
import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from vitalhealth_storage import get_store

import auth

load_dotenv()

app = FastAPI(title="VitalHealth gateway")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/vh-assets", StaticFiles(directory=BASE_DIR / "static"), name="vh-assets")

SHARED_STORE = get_store()

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
    """Return a deterministic local Vita reply when no model provider is available."""
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
        gemini_base = os.getenv("VITA_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

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
body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 720px;
    margin: 4rem auto;
    padding: 0 1.5rem;
    color: #14213d;
    background: #f5f7fa;
}
h1 {
    margin-bottom: 0.25rem;
}
p.sub {
    color: #5f6f82;
    margin-top: 0;
}
.card {
    display: block;
    border: 1px solid #d8e1ec;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    text-decoration: none;
    color: inherit;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(20, 33, 61, 0.06);
}
.card:hover {
    border-color: #1f6feb;
}
.card h2 {
    margin: 0 0 0.25rem 0;
    font-size: 1.1rem;
}
.card p {
    margin: 0;
    color: #5f6f82;
    font-size: 0.95rem;
}
form {
    margin-top: 1.5rem;
}
label {
    display: block;
    margin: 1rem 0 0.25rem;
    font-size: 0.9rem;
    color: #14213d;
}
input[type=email],
input[type=password] {
    width: 100%;
    padding: 0.65rem;
    border: 1px solid #d8e1ec;
    border-radius: 8px;
    font-size: 1rem;
    box-sizing: border-box;
}
button {
    margin-top: 1.5rem;
    padding: 0.7rem 1.25rem;
    border: none;
    border-radius: 8px;
    background: #1f6feb;
    color: #fff;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
}
button:hover {
    background: #1557b0;
}
.error {
    background: #fdecec;
    color: #b91c1c;
    border: 1px solid #f4b4b4;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 1rem;
    font-size: 0.9rem;
}
.hint {
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #5f6f82;
}
.hint a {
    color: #1f6feb;
    font-weight: 700;
}
.topbar {
    text-align: right;
    font-size: 0.9rem;
    color: #5f6f82;
    margin-bottom: 1rem;
}
.topbar a {
    color: #1f6feb;
    font-weight: 700;
}
.footer {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #5f6f82;
    border-top: 1px solid #d8e1ec;
    padding-top: 1rem;
}
"""


def _authenticated_user(request: Request) -> dict | None:
    return auth.read_session_token(request.cookies.get(auth.COOKIE_NAME))


def _safe_next_path(next_path: str) -> str:
    """Only redirect to a same-origin path."""
    if next_path.startswith("/") and not next_path.startswith("//"):
        return next_path

    return "/"


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
  <style>{_SHARED_CSS}</style>
</head>

<body>
  <div class="topbar">{topbar}</div>
  <h1>VitalHealth</h1>
  <p class="sub">Integrated educational clinical workflow support platform.</p>

  <a class="card" href="/triage/">
    <h2>Clinical Triage</h2>
    <p>Supports triage priority assessment from intake details and clinician notes.</p>
  </a>

  <a class="card" href="/stroke/">
    <h2>Stroke Assessment</h2>
    <p>Estimates stroke risk and generates educational care planning guidance.</p>
  </a>

  <a class="card" href="/emc/">
    <h2>EMC Workflow</h2>
    <p>Prepares electronic medical certificate drafts for clinician review and approval.</p>
  </a>

  <p class="footer">
    VitalHealth is an educational clinical decision-support prototype. It does not provide a medical diagnosis and does not replace review by a qualified healthcare professional.
  </p>
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
  <style>{_SHARED_CSS}</style>
</head>

<body>
  <h1>Log in</h1>
  <p class="sub">One login for triage, stroke assessment, and EMC workflow.</p>

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
</body>
</html>
"""


def _render_register(error: str | None, email: str) -> str:
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    email_value = html.escape(email)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Register - VitalHealth</title>
  <style>{_SHARED_CSS}</style>
</head>

<body>
  <h1>Register</h1>
  <p class="sub">Create an account to use VitalHealth.</p>

  {error_html}

  <form method="post" action="/register">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" value="{email_value}" required autofocus>

    <label for="password">Password</label>
    <input type="password" id="password" name="password" required minlength="8">

    <label for="confirm">Confirm password</label>
    <input type="password" id="confirm" name="confirm" required minlength="8">

    <button type="submit">Register</button>
  </form>

  <p class="hint">Already have an account? <a href="/login">Log in</a></p>
</body>
</html>
"""


def _set_session_cookie(response: Response, user_id: str, email: str) -> None:
    token = auth.create_session_token(user_id, email)

    response.set_cookie(
        auth.COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=os.environ.get("SESSION_COOKIE_SECURE", "").lower() == "true",
        max_age=auth.SESSION_MAX_AGE_SECONDS,
        path="/",
    )


@app.get("/")
async def landing(request: Request) -> HTMLResponse:
    return HTMLResponse(_render_landing(_authenticated_user(request)))


@app.get("/login")
async def login_form(request: Request) -> HTMLResponse:
    error = request.query_params.get("error", "")
    next_path = _safe_next_path(request.query_params.get("next", "/"))
    email = request.query_params.get("email", "")

    return HTMLResponse(_render_login(error or None, next_path, email))


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()

    email = str(form.get("email", "")).strip()
    password = str(form.get("password", ""))
    next_path = _safe_next_path(str(form.get("next", "") or "/"))

    user = SHARED_STORE.get_user_by_email(email) if email else None

    if not user or not auth.verify_password(password, user["password_hash"]):
        query = f"error={quote('Incorrect email or password.')}&next={quote(next_path)}&email={quote(email)}"
        return RedirectResponse(url=f"/login?{query}", status_code=303)

    response = RedirectResponse(url=next_path, status_code=303)
    _set_session_cookie(response, user["id"], user["email"])

    return response


@app.get("/register")
async def register_form(request: Request) -> HTMLResponse:
    error = request.query_params.get("error", "")
    email = request.query_params.get("email", "")

    return HTMLResponse(_render_register(error or None, email))


@app.post("/register")
async def register_submit(request: Request):
    form = await request.form()

    email = str(form.get("email", "")).strip()
    password = str(form.get("password", ""))
    confirm = str(form.get("confirm", ""))

    def fail(message: str):
        query = f"error={quote(message)}&email={quote(email)}"
        return RedirectResponse(url=f"/register?{query}", status_code=303)

    if not email or "@" not in email:
        return fail("Enter a valid email address.")

    if len(password) < 8:
        return fail("Password must be at least 8 characters.")

    if password != confirm:
        return fail("Passwords do not match.")

    try:
        user_id = SHARED_STORE.create_user(
            email=email,
            password_hash=auth.hash_password(password),
        )
    except IntegrityError:
        return fail("Email already registered.")

    response = RedirectResponse(url="/", status_code=303)
    _set_session_cookie(response, user_id, email.strip().lower())

    return response


@app.api_route("/logout", methods=["GET", "POST"])
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(auth.COOKIE_NAME, path="/")

    return response


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

    if local_reply:
        return {
            "success": True,
            "data": {"reply": local_reply},
            "error": None,
        }

    return {
        "success": True,
        "data": {"reply": VITA_FALLBACK},
        "error": None,
    }


@app.api_route("/{prefix}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def redirect_bare_prefix(prefix: str):
    """Force a trailing slash so relative asset/API paths resolve correctly."""
    if prefix in BACKENDS:
        return RedirectResponse(url=f"/{prefix}/", status_code=307)

    return Response(status_code=404)


@app.api_route("/{prefix}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(prefix: str, path: str, request: Request):
    upstream = BACKENDS.get(prefix)

    if upstream is None:
        return Response(status_code=404)

    user = _authenticated_user(request)

    if user is None:
        if "text/html" in request.headers.get("accept", ""):
            next_path = quote(f"/{prefix}/{path}")
            return RedirectResponse(url=f"/login?next={next_path}", status_code=303)

        return JSONResponse({"detail": "Authentication required"}, status_code=401)

    url = f"{upstream}/{path}"

    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    forward_headers["X-Forwarded-Prefix"] = f"/{prefix}"
    forward_headers["X-Forwarded-Host"] = request.headers.get("host", "")
    forward_headers["X-Forwarded-Proto"] = request.url.scheme
    forward_headers["X-Vitalhealth-User"] = user["uid"]
    forward_headers["X-Vitalhealth-User-Email"] = user["email"]

    body = await request.body()

    upstream_response = await _client.request(
        request.method,
        url,
        params=request.query_params,
        headers=forward_headers,
        content=body,
    )

    response_headers = {
        k: v for k, v in upstream_response.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    content = upstream_response.content
    content_type = upstream_response.headers.get("content-type", "").lower()

    if prefix == "stroke" and "text/html" in content_type:
        stroke_html = content.decode("utf-8", errors="replace")

        # Rewrite only Stroke app internal links.
        # Keep platform-level routes untouched:
        # /triage/    = main dashboard / Clinical Triage
        # /emc/       = EMC workflow
        # /api/       = Vita/chat API
        # /vh-assets/ = shared shell assets
        # /stroke/    = already prefixed
        stroke_html = re.sub(
            r'(href|src|action)="/(?!(?:stroke/|triage/|emc/|api/|vh-assets/|"))',
            r'\1="/stroke/',
            stroke_html,
        )

        content = stroke_html.encode("utf-8")

    if prefix == "emc" and "text/html" in content_type:
        emc_html = content.decode("utf-8", errors="replace")

        # Normalize old EMC headers to use the same brand logo element as other pages.
        if "brand-logo" not in emc_html:
            emc_html = re.sub(
                r'<a\s+class="brand"\s+href="/triage/">\s*VitalHealth\s*</a>',
                '<a class="brand brand-logo" href="/triage/" aria-label="VitalHealth Home">'
                '<img src="/vh-assets/brand/vitalhealth-logo-full.png" alt="VitalHealth">'
                '</a>',
                emc_html,
                flags=re.IGNORECASE,
            )

        # Remove stale top-level Results nav item if an older EMC template is served.
        emc_html = re.sub(
            r'\s*<a\s+href="/stroke/care-plan">\s*Results\s*</a>\s*',
            "\n",
            emc_html,
            flags=re.IGNORECASE,
        )

        # Force clients to fetch the latest EMC stylesheet after UI updates.
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