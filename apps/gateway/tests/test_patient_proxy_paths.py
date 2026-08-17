"""apps/gateway/tests/test_patient_proxy_paths.py — regression guard for
_patient_may_use_proxy_path's triage allowlist, mirroring
apps/Jace/tests/test_self_check_allowlist.py.

Both allowlists — this one and Jace's own api.py:PATIENT_API_PREFIX — must agree
on what a patient session may reach, or the gateway 403s a route Jace itself would
happily serve (the bug this test exists to catch: a prior enumerated-set version
of this function silently 403'd POST /api/self-check/explain for every real
patient session, because the tuple was never updated when that route was added).

Run:  python -m pytest apps/gateway/tests/test_patient_proxy_paths.py -v

Note: this repo has no established test scaffolding for apps/gateway (no prior
tests/ directory existed here) — this file establishes the same layout
apps/Jace/tests already uses.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import _patient_may_use_proxy_path  # noqa: E402  (needs sys.path insert above)

# Every route the patient self-check page (apps/Jace/static/self-check.js) actually
# calls, through the gateway's triage proxy prefix. Kept in sync with the same list
# in apps/Jace/tests/test_self_check_allowlist.py — if that list grows, this one
# should grow with it.
_PATIENT_PAGE_ROUTES = {
    "api/self-check/options",
    "api/self-check/extract",
    "api/self-check",
    "api/self-check/describe-help",
    "api/self-check/explain",
}


def test_every_patient_page_route_is_reachable_through_the_triage_proxy():
    for path in _PATIENT_PAGE_ROUTES:
        assert _patient_may_use_proxy_path("triage", path), (
            f"{path} is not reachable by a patient session through the gateway — "
            "a real patient would 403 on it even if Jace's own api.py allows it."
        )


def test_the_triage_allowlist_is_not_vacuously_wide():
    """Proves the test above is meaningful: a known clinician-only triage route must
    still be rejected."""
    clinician_only = {"api/predict", "api/vocab", "api/extract", "api/explain",
                       "api/patients", "api/guardrail-test"}
    for path in clinician_only:
        assert not _patient_may_use_proxy_path("triage", path), (
            f"{path} unexpectedly reachable by a patient session — the triage "
            "allowlist has become too wide."
        )


def test_static_and_dashboard_paths_stay_reachable():
    """The self-check prefix change must not have narrowed what was already open:
    static assets (any non-api/ path) and the dashboard prefix."""
    assert _patient_may_use_proxy_path("triage", "self-check.html")
    assert _patient_may_use_proxy_path("triage", "api/dashboard/summary")
