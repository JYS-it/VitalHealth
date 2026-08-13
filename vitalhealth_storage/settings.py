"""Settings every VitalHealth process must agree on, resolved identically.

SESSION_SECRET and DATABASE_URL are not per-app configuration — they are
shared platform state. The gateway signs a session cookie that all three
backends verify, and all four read the same database. If any process
disagrees on either value it fails *silently*: signature checks return
"no session" rather than an error, so a logged-in user simply gets 401/403
from a backend and their records look like they vanished.

That is exactly what happens when the apps are started by hand instead of
through run_all.py, because each one previously resolved these differently:

    gateway   load_dotenv(<repo root>/.env)
    Jeslyn    load_dotenv()   -> apps/Jeslyn/.env, where SESSION_SECRET= is
    YS        load_dotenv()   -> apps/YS/.env,     blank and clobbers it
    Jace      nothing at all  -> no secret, no database

This module gives all four one answer. Deliberately stdlib-only: Jace does
not depend on python-dotenv, and a shared settings loader should not be the
thing that forces a new dependency into an app's pinned requirements.
"""

from __future__ import annotations

import os
from pathlib import Path

# Resolved from this file so it does not depend on the working directory a
# process happens to be started from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Both are read in order; the first non-empty value wins. The repo root is
# what the gateway already loads, and apps/gateway/.env is what run_all.py
# treats as the canonical shared source.
SHARED_ENV_FILES = (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "apps" / "gateway" / ".env",
)

SHARED_KEYS = ("SESSION_SECRET", "DATABASE_URL")


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader — enough for the shared keys, no dependency."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def load_shared_env(keys: tuple[str, ...] = SHARED_KEYS) -> dict[str, str]:
    """Fill any shared key this process is missing, and report what it set.

    An **empty** value counts as missing. That is the whole point: a local
    .env carrying `SESSION_SECRET=` with nothing after it is not a choice to
    run without a secret, it is an unfilled placeholder, and letting it win
    over the real shared value is the bug this exists to prevent.

    A value already present and non-empty in the environment is never
    overridden, so run_all.py's injection and any real deployment's
    environment still take precedence.
    """
    filled: dict[str, str] = {}
    file_cache: dict[Path, dict[str, str]] = {}

    for key in keys:
        if os.environ.get(key, "").strip():
            continue
        for env_file in SHARED_ENV_FILES:
            if env_file not in file_cache:
                file_cache[env_file] = _parse_env_file(env_file)
            value = file_cache[env_file].get(key, "").strip()
            if value:
                os.environ[key] = value
                filled[key] = str(env_file)
                break

    return filled


def missing_shared_keys(keys: tuple[str, ...] = SHARED_KEYS) -> list[str]:
    """Shared keys still unset after loading — worth warning about at startup."""
    return [key for key in keys if not os.environ.get(key, "").strip()]
