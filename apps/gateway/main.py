"""Reverse-proxy gateway — the single public entry point for all three apps.

This process owns no business logic. It only forwards HTTP requests to
whichever backend a path prefix names, then relays the response back
unchanged. Each backend keeps running as its own process in its own
virtualenv (they pin conflicting dependency versions — see the root
README — and cannot share one Python environment), so this is the only
way to give users one URL without merging the three codebases.

Routing:
    /            -> landing page (this process)
    /triage/*    -> Jace   (FastAPI, CTRSE triage acuity)
    /stroke/*    -> Jeslyn (Flask, stroke risk + care plan)
    /emc/*       -> YS     (Flask, EMC copilot)

Jace's frontend already assumes it owns the page it's served from and
uses paths relative to that page, so no `root_path`/`SCRIPT_NAME`
awareness is needed on its side. The two Flask apps generate links with
`url_for()`, which respects `SCRIPT_NAME` — the `X-Forwarded-Prefix`
header set below is read by a small PrefixMiddleware in each Flask app
to make `url_for()` emit correctly-prefixed URLs.
"""
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

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

LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VitalHealth</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 4rem auto; padding: 0 1.5rem; color: #1a1a1a; }
    h1 { margin-bottom: 0.25rem; }
    p.sub { color: #555; margin-top: 0; }
    .card { display: block; border: 1px solid #ddd; border-radius: 8px; padding: 1.25rem 1.5rem; margin: 1rem 0; text-decoration: none; color: inherit; }
    .card:hover { border-color: #888; }
    .card h2 { margin: 0 0 0.25rem 0; font-size: 1.1rem; }
    .card p { margin: 0; color: #555; font-size: 0.95rem; }
  </style>
</head>
<body>
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


@app.get("/")
async def landing() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML)


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

    url = f"{upstream}/{path}"
    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    forward_headers["X-Forwarded-Prefix"] = f"/{prefix}"
    forward_headers["X-Forwarded-Host"] = request.headers.get("host", "")
    forward_headers["X-Forwarded-Proto"] = request.url.scheme

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
