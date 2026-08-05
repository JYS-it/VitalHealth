"""precompute_stats.py — one-off builder for vocab/precomputed_stats.json.

ctrse_core.init() loads X_train.npy (~985 MB) + X_test.npy (~246 MB) ONLY to derive two
static summaries: BASE_RATES (train-only complaint emergency rates, n>=300 floor) and
TOP_FEATURES (the permutation-importance top-15). This script runs that exact derivation
once — by calling core.init() on the arrays path — and pins the results to a small JSON so
the deployed app can init() without the arrays (KB, not ~1.2 GB). No logic is reimplemented
here: the numbers come straight out of init(), so they cannot drift from the app.

Run (where the .npy arrays exist):  python precompute_stats.py
Re-run whenever the NB1/NB2 artefacts (X_*.npy, feature_cols.pkl, the model bundle) change.
"""

import json
import os
from datetime import datetime, timezone

import sklearn

import ctrse_core as core

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_REL = os.path.join("vocab", "precomputed_stats.json")
OUT_PATH = os.path.join(APP_DIR, OUT_REL)

REQUIRED_ARRAYS = ("X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy")


def main():
    missing = [n for n in REQUIRED_ARRAYS if not os.path.exists(os.path.join(APP_DIR, n))]
    if missing:
        raise SystemExit(f"ERROR: precompute needs the source arrays; missing {missing} in {APP_DIR}")

    # init() short-circuits to the precomputed path if the JSON already exists; move any
    # existing copy aside so this run derives fresh from the arrays, then discard the backup.
    backup = None
    if os.path.exists(OUT_PATH):
        backup = OUT_PATH + ".bak"
        os.replace(OUT_PATH, backup)
    try:
        core.init(APP_DIR)                       # arrays path -> core.BASE_RATES + core.TOP_FEATURES
        if core.X_train is None:
            raise SystemExit("ERROR: init() did not take the arrays path (arrays were not loaded).")
        out = {
            "base_rates": core.BASE_RATES,
            "top_features": sorted(core.TOP_FEATURES),
            "provenance": {
                "source_artefacts": list(REQUIRED_ARRAYS) + ["ctrse_p1p4_model.pkl", "feature_cols.pkl"],
                "n_train_rows": int(len(core.X_train)),
                "n_test_rows": int(len(core.X_test)),
                "random_state": core.RANDOM_STATE,
                "sklearn_version": sklearn.__version__,
                "computed_utc": datetime.now(timezone.utc).isoformat(),
                "note": ("Derived by precompute_stats.py from the NB1/NB2 artefacts so "
                         "ctrse_core.init() can run without X_*/y_*.npy. See init() for the logic."),
            },
        }
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    except Exception:
        if backup is not None and not os.path.exists(OUT_PATH):
            os.replace(backup, OUT_PATH)         # restore the prior JSON on failure
        raise
    else:
        if backup is not None:
            os.remove(backup)

    n_rated = sum(1 for v in out["base_rates"].values() if v["n"] >= 300)
    print(f"Wrote {OUT_REL}: {len(out['base_rates'])} base_rates "
          f"({n_rated} with n>=300), {len(out['top_features'])} top_features.")
    print(f"  provenance: {out['provenance']['n_train_rows']} train rows, "
          f"{out['provenance']['n_test_rows']} test rows, sklearn {out['provenance']['sklearn_version']}")


if __name__ == "__main__":
    main()
