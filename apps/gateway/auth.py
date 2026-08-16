"""Credential mechanics for the gateway's login system.

Only the gateway ever handles a password, so bcrypt lives here rather than in
the shared package — there is no reason for the three clinical backends to
carry a credential-hashing dependency.

Session tokens moved to vitalhealth_storage.identity when roles were added:
the backends need to read the same cookie the gateway writes, and two copies
of the signing logic would eventually disagree. The re-exports below keep
main.py's existing `auth.COOKIE_NAME` / `auth.read_session_token` references
working.
"""
from __future__ import annotations

import bcrypt

from vitalhealth_storage.identity import (  # noqa: F401  (re-exported for main.py)
    COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    SUBJECT_COOKIE_NAME,
    create_session_token,
    create_subject_token,
    read_session_token,
)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False
