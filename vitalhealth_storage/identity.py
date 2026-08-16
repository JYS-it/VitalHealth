"""Who is making this request — shared by all four VitalHealth processes.

The gateway mints a signed session cookie at login; every other app reads it.
Because the cookie is signed with a secret the browser never sees, a backend
can trust it without asking the gateway or the database anything, which keeps
the stateless design the gateway already committed to.

This matters more than it looks. The gateway also forwards X-Vitalhealth-User
headers, but the three backends listen on 127.0.0.1 and anything local can
forge a header. The cookie's signature is the part that cannot be forged, so
the cookie — not the header — decides who the actor is.

Deliberately dependency-light: stdlib plus itsdangerous, and *no import of
store.py*. Identity is not persistence, and keeping the two apart means this
module can be exercised without a database.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

LOGGER = logging.getLogger(__name__)

COOKIE_NAME = "vh_session"
SUBJECT_COOKIE_NAME = "vh_subject"
SESSION_MAX_AGE_SECONDS = int(os.environ.get("SESSION_MAX_AGE_SECONDS", 60 * 60 * 12))

ROLE_PATIENT = "patient"
ROLE_CLINICIAN = "clinician"
VALID_ROLES = frozenset({ROLE_PATIENT, ROLE_CLINICIAN})
DEFAULT_ROLE = ROLE_PATIENT

_INSECURE_DEFAULT_SECRET = "change-this-in-production"

# Bumped from "vitalhealth-session" when roles were introduced. Sessions issued
# before roles existed carry no role and there is no safe way to guess one, so
# they are invalidated by signature instead of being interpreted. Everyone logs
# in once and comes back with a role-bearing cookie.
_SESSION_SALT = "vitalhealth-session-v2"
_SUBJECT_SALT = "vitalhealth-subject-v1"


def _secret() -> str:
    secret = os.environ.get("SESSION_SECRET", "").strip()
    if secret:
        return secret
    # A backend silently falling back to the default while the gateway uses a
    # real secret produces no error anywhere — every signature check just fails
    # and identity quietly resolves to None. Say so out loud, once.
    global _warned_about_secret
    if not _warned_about_secret:
        _warned_about_secret = True
        LOGGER.warning(
            "SESSION_SECRET is not set; falling back to the insecure default. "
            "Sessions signed by another process will not verify."
        )
    return _INSECURE_DEFAULT_SECRET


_warned_about_secret = False
_serializers: dict[tuple[str, str], URLSafeTimedSerializer] = {}


def _serializer(salt: str) -> URLSafeTimedSerializer:
    """Resolved on first use, not at import.

    Callers load their .env after importing this module (apps/gateway/main.py
    does exactly that), so reading the secret at import time would silently
    capture the insecure default and every cross-process signature check would
    fail. Keyed by secret as well as salt so a late-arriving value still wins.
    """
    key = (_secret(), salt)
    if key not in _serializers:
        _serializers[key] = URLSafeTimedSerializer(key[0], salt=salt)
    return _serializers[key]


@dataclass(frozen=True)
class Actor:
    """The authenticated human behind a request."""

    user_id: str
    email: str
    role: str
    patient_external_id: str | None = None
    display_name: str | None = None

    @property
    def is_patient(self) -> bool:
        return self.role == ROLE_PATIENT

    @property
    def is_clinician(self) -> bool:
        return self.role == ROLE_CLINICIAN


def create_session_token(
    *,
    user_id: str,
    email: str,
    role: str,
    patient_external_id: str | None = None,
    display_name: str | None = None,
) -> str:
    """Sign the session payload. `pref` is the patient's external reference and
    is only meaningful for patient accounts."""
    return _serializer(_SESSION_SALT).dumps({
        "uid": user_id,
        "email": email,
        "role": role if role in VALID_ROLES else DEFAULT_ROLE,
        "pref": patient_external_id,
        "name": display_name,
    })


def read_session_token(token: str | None) -> dict | None:
    """The raw payload, or None when the signature is bad, expired, or absent."""
    if not token:
        return None
    try:
        payload = _serializer(_SESSION_SALT).loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    return payload if isinstance(payload, dict) else None


def actor_from_token(token: str | None) -> Actor | None:
    payload = read_session_token(token)
    if not payload:
        return None

    user_id = payload.get("uid")
    role = payload.get("role")
    # A token without a recognised role is not usable: guessing "clinician"
    # would leak every patient's records, and guessing "patient" would silently
    # demote staff. Treat it as no session at all.
    if not user_id or role not in VALID_ROLES:
        return None

    return Actor(
        user_id=str(user_id),
        email=str(payload.get("email") or ""),
        role=role,
        patient_external_id=payload.get("pref") or None,
        display_name=payload.get("name") or None,
    )


def actor_from_cookies(cookies: Mapping[str, str] | None) -> Actor | None:
    """Works unchanged in FastAPI and Flask — both expose request.cookies as a
    string mapping."""
    if not cookies:
        return None
    return actor_from_token(cookies.get(COOKIE_NAME))


def create_subject_token(*, external_id: str, display_name: str | None = None) -> str:
    """The patient a clinician is currently working on.

    Neither the triage nor the stroke form has a patient field, so without this
    a clinician's assessments would have no subject and could never be filed
    against anyone.
    """
    return _serializer(_SUBJECT_SALT).dumps({"pref": external_id, "name": display_name})


def read_subject_token(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        payload = _serializer(_SUBJECT_SALT).loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_subject(
    cookies: Mapping[str, str] | None,
    actor: Actor | None,
) -> tuple[str | None, str | None]:
    """The patient this request is *about*, as (external_id, display_name).

    One uniform answer for every persistence site: a patient is always their
    own subject regardless of what any form field claims, a clinician's subject
    is whichever patient they selected, and an unauthenticated request has none.
    """
    if actor is None:
        return None, None

    if actor.is_patient:
        return actor.patient_external_id, actor.display_name

    payload = read_subject_token((cookies or {}).get(SUBJECT_COOKIE_NAME))
    if not payload:
        return None, None

    return payload.get("pref") or None, payload.get("name") or None


def normalise_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return role if role in VALID_ROLES else DEFAULT_ROLE
