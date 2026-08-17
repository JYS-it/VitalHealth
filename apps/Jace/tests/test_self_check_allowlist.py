"""tests/test_self_check_allowlist.py — regression guard for the bug this session's
plan (§Unify the patient intake interface with the clinician's) fixed: a prior
enumerated-set version of PATIENT_API_PREFIX silently 403'd POST /api/self-check/explain
for every real patient session, because it was never added to that set (the same
mistake was independently made in apps/gateway/main.py's mirrored allowlist).

This test enumerates api.app's actual route table so a FUTURE patient route is
caught the same way: any route the patient page calls that isn't reachable under
the prefix fails here, at commit time, instead of 403ing silently in production
for anyone with a real session cookie (the same reason the bug shipped unnoticed —
TestClient sends no cookie, so the middleware's `if token:` gate never fires in a
route-level test that doesn't check this directly).

Run:  python -m pytest tests/test_self_check_allowlist.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api

# Every route the patient self-check page (static/self-check.js) actually calls.
# If a future route is added to that page without being added here too, this list
# — not the allowlist itself — is what should be extended, so the mismatch is a
# one-line diff instead of a rediscovered bug.
_PATIENT_PAGE_ROUTES = {
    "/api/self-check/options",
    "/api/self-check/extract",
    "/api/self-check",
    "/api/self-check/describe-help",
    "/api/self-check/explain",
}


def test_every_patient_page_route_is_covered_by_the_prefix():
    for path in _PATIENT_PAGE_ROUTES:
        assert path.startswith(api.PATIENT_API_PREFIX), (
            f"{path} is not reachable under PATIENT_API_PREFIX "
            f"({api.PATIENT_API_PREFIX!r}) — a real patient session would 403 on it."
        )


def test_every_registered_self_check_route_is_covered_by_the_prefix():
    """The other direction: every route actually registered on the app under
    /api/self-check must itself satisfy the prefix (trivially true today since the
    prefix IS "/api/self-check", but this is what would catch a route registered
    under a sibling path like /api/selfcheck/... or /api/patient-check/... that a
    future refactor might introduce without updating the prefix to match)."""
    self_check_paths = {
        route.path for route in api.app.routes
        if getattr(route, "path", "").startswith("/api/self")
    }
    assert self_check_paths, "no /api/self* routes found — has the route table moved?"
    for path in self_check_paths:
        assert path.startswith(api.PATIENT_API_PREFIX), (
            f"registered route {path} does not satisfy PATIENT_API_PREFIX "
            f"({api.PATIENT_API_PREFIX!r})"
        )


def test_the_prefix_is_not_vacuously_wide():
    """Proves the test above is meaningful, not trivially true: a known
    clinician-only route must NOT satisfy the same prefix check."""
    clinician_only = {"/api/predict", "/api/vocab", "/api/extract", "/api/explain",
                       "/api/patients", "/api/guardrail-test"}
    for path in clinician_only:
        assert not path.startswith(api.PATIENT_API_PREFIX), (
            f"{path} unexpectedly satisfies PATIENT_API_PREFIX — the prefix has "
            "become too wide and would let a patient session reach a clinical route."
        )
