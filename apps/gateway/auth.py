"""Session/credential mechanics for the gateway's login system.

Kept separate from main.py so proxy/routing logic doesn't get tangled with
credential and cookie handling. The gateway validates sessions statelessly:
the database is only touched at login (verify password) and register
(create user) — every subsequent proxied request is authenticated purely by
verifying the session cookie's signature and expiry in-process.
"""
import os

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

COOKIE_NAME = "vh_session"
SESSION_MAX_AGE_SECONDS = int(os.environ.get("SESSION_MAX_AGE_SECONDS", 60 * 60 * 12))

_serializer = URLSafeTimedSerializer(
    os.environ.get("SESSION_SECRET", "change-this-in-production"),
    salt="vitalhealth-session",
)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_session_token(user_id: str, email: str) -> str:
    return _serializer.dumps({"uid": user_id, "email": email})


def read_session_token(token: str | None) -> dict | None:
    if not token:
        return None
    try:
        return _serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
