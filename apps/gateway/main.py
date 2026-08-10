"""Reverse-proxy gateway — the single public entry point for all three apps.

This process owns no business logic. It only forwards HTTP requests to
whichever backend a path prefix names, then relays the response back
unchanged.

Routing:
    /            -> landing page
    /triage/*    -> Jace   (FastAPI, CTRSE triage acuity)
    /stroke/*    -> Jeslyn (Flask, stroke risk + care plan)
    /emc/*       -> YS     (Flask, EMC copilot)
"""

import os
import re
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

app = FastAPI(title="VitalHealth gateway")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/vh-assets", StaticFiles(directory=BASE_DIR / "static"), name="vh-assets")

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

    # Prefer Gemini when explicitly configured, or when no provider is set and
    # a Gemini key is available.
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

        # If Gemini was explicitly requested, do not try other providers.
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

LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VitalHealth</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 760px;
      margin: 4rem auto;
      padding: 0 1.5rem;
      color: #1a1a1a;
      background: #f6f7f9;
    }

    h1 {
      margin-bottom: 0.25rem;
      color: #143447;
    }

    p.sub {
      color: #555;
      margin-top: 0;
      margin-bottom: 2rem;
    }

    .card {
      display: block;
      border: 1px solid #dfe5eb;
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin: 1rem 0;
      text-decoration: none;
      color: inherit;
      background: #ffffff;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }

    .card:hover {
      border-color: #0f6075;
    }

    .card h2 {
      margin: 0 0 0.25rem 0;
      font-size: 1.1rem;
      color: #17324a;
    }

    .card p {
      margin: 0;
      color: #555;
      font-size: 0.95rem;
    }

    .footer {
      margin-top: 2rem;
      font-size: 0.8rem;
      color: #5a6472;
      border-top: 1px solid #dfe5eb;
      padding-top: 1rem;
    }
  </style>
</head>

<body>
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


@app.get("/")
async def landing() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML)


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

    url = f"{upstream}/{path}"

    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
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
        k: v for k, v in upstream_response.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    content = upstream_response.content
    content_type = upstream_response.headers.get("content-type", "").lower()

    if prefix == "stroke" and "text/html" in content_type:
        html = content.decode("utf-8", errors="replace")

        # Rewrite only Stroke app internal links.
        # Keep platform-level routes untouched:
        # /triage/    = main working dashboard
        # /emc/       = EMC workflow
        # /api/       = Vita/chat API
        # /vh-assets/ = shared shell assets
        # /stroke/    = already prefixed
        html = re.sub(
            r'(href|src|action)="/(?!(?:stroke/|triage/|emc/|api/|vh-assets/|"))',
            r'\1="/stroke/',
            html,
        )

        content = html.encode("utf-8")

    return Response(
        content=content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )