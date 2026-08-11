"""Reverse-proxy gateway — the single public entry point for all three apps.

This process forwards HTTP requests to whichever backend a path prefix
names, then relays the response back unchanged. Each backend keeps running
as its own process in its own virtualenv (they pin conflicting dependency
versions — see the root README — and cannot share one Python environment),
so this is the only way to give users one URL without merging the three
codebases.

Routing:
    /            -> landing page (this process)
    /login, /register, /logout -> session management (this process)
    /triage/*    -> Jace   (FastAPI, CTRSE triage acuity)
    /stroke/*    -> Jeslyn (Flask, stroke risk + care plan)
    /emc/*       -> YS     (Flask, EMC copilot)

Jace's frontend already assumes it owns the page it's served from and
uses paths relative to that page, so no `root_path`/`SCRIPT_NAME`
awareness is needed on its side. The two Flask apps generate links with
`url_for()`, which respects `SCRIPT_NAME` — the `X-Forwarded-Prefix`
header set below is read by a small PrefixMiddleware in each Flask app
to make `url_for()` emit correctly-prefixed URLs.

Authentication (single sign-on, owned entirely by this process): every
`/{prefix}/{path:path}` request must carry a valid `vh_session` cookie
(see auth.py) or it is redirected to /login (browser navigation) / gets a
401 (API-style requests). Because all three backends are only reachable
through this one origin, one login here covers all of them. Backends
require no auth code of their own — the gateway forwards the caller's
identity as X-Vitalhealth-User / X-Vitalhealth-User-Email headers, which
they may optionally read. Running a backend standalone (bypassing this
gateway, e.g. `uvicorn api:app` directly) is NOT protected by this layer —
the same class of caveat as X-Forwarded-Prefix already being a no-op when
absent.
"""
import html
import os
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from sqlalchemy.exc import IntegrityError
from vitalhealth_storage import get_store

import auth

load_dotenv()
SHARED_STORE = get_store()

app = FastAPI(title="VitalHealth gateway")

BACKENDS = {
    "triage": os.environ.get("TRIAGE_UPSTREAM", "http://127.0.0.1:8000"),
    "stroke": os.environ.get("STROKE_UPSTREAM", "http://127.0.0.1:5000"),
    "emc": os.environ.get("EMC_UPSTREAM", "http://127.0.0.1:5001"),
}

# Headers that describe this specific hop and must not be relayed verbatim
# (RFC 7230 §6.1, plus Host/Content-Length which httpx recomputes for us).
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host",
}

_client = httpx.AsyncClient(follow_redirects=False, timeout=30.0)

_SHARED_CSS = """
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 4rem auto; padding: 0 1.5rem; color: #1a1a1a; }
    h1 { margin-bottom: 0.25rem; }
    p.sub { color: #555; margin-top: 0; }
    .card { display: block; border: 1px solid #ddd; border-radius: 8px; padding: 1.25rem 1.5rem; margin: 1rem 0; text-decoration: none; color: inherit; }
    .card:hover { border-color: #888; }
    .card h2 { margin: 0 0 0.25rem 0; font-size: 1.1rem; }
    .card p { margin: 0; color: #555; font-size: 0.95rem; }
    form { margin-top: 1.5rem; }
    label { display: block; margin: 1rem 0 0.25rem; font-size: 0.9rem; color: #333; }
    input[type=email], input[type=password] { width: 100%; padding: 0.5rem; border: 1px solid #ccc; border-radius: 6px; font-size: 1rem; box-sizing: border-box; }
    button { margin-top: 1.5rem; padding: 0.6rem 1.25rem; border: none; border-radius: 6px; background: #1a1a1a; color: #fff; font-size: 1rem; cursor: pointer; }
    button:hover { background: #333; }
    .error { background: #fdecea; color: #a33; border: 1px solid #f3c2bd; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 1rem; font-size: 0.9rem; }
    .hint { margin-top: 1rem; font-size: 0.9rem; color: #555; }
    .hint a { color: #1a1a1a; }
    .topbar { text-align: right; font-size: 0.9rem; color: #555; margin-bottom: 1rem; }
    .topbar a { color: #1a1a1a; }
"""


def _authenticated_user(request: Request) -> dict | None:
    return auth.read_session_token(request.cookies.get(auth.COOKIE_NAME))


def _safe_next_path(next_path: str) -> str:
    """Only ever redirect to a same-origin path. `next` round-trips through
    a query param and a hidden form field, so treat it as untrusted input —
    otherwise a crafted /login?next=https://evil.com link becomes an open
    redirect once the victim submits valid credentials."""
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
  <p class="sub">Three independent clinical tools behind one address.</p>

  <a class="card" href="/triage/">
    <h2>Triage acuity (CTRSE)</h2>
    <p>Predicts a triage assignment from patient data or a clinician note.</p>
  </a>

  <a class="card" href="/stroke/">
    <h2>Stroke risk assessment</h2>
    <p>Stroke risk prediction with a grounded, source-cited care plan.</p>
  </a>

  <a class="card" href="/emc/">
    <h2>IIP-EMC Clinical Copilot</h2>
    <p>AI-drafted electronic medical certificates with a clinician review gate.</p>
  </a>
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
  <p class="sub">One login for triage, stroke risk, and EMC drafting.</p>
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
        user_id = SHARED_STORE.create_user(email=email, password_hash=auth.hash_password(password))
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


@app.api_route("/{prefix}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def redirect_bare_prefix(prefix: str):
    """Force a trailing slash so relative asset/API paths in the backend's
    own HTML resolve against the right base (e.g. /triage, not /)."""
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
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
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
        k: v for k, v in upstream_response.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
