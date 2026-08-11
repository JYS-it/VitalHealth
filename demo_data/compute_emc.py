"""Compute real EMC diagnosis-classifier predictions for the demo roster.

Run with YS's own venv (needs YS's pinned scikit-learn to unpickle the
model bundle):

    apps\\YS\\.venv\\Scripts\\python.exe demo_data\\compute_emc.py

Calls apps/YS/app.py's real `run_prediction()` (app.py:206) directly —
importing app.py only loads joblib model artifacts and registers Flask
routes (it never calls app.run() or constructs the OpenAI client at import
time), so this is safe without a running server or API keys.

Deliberately captures the ML classification step only, not YS's LLM-drafted
certificate text (see demo_data/README.md for why). Writes
demo_data/output/emc.json.
"""
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YS_DIR = ROOT / "apps" / "YS"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "emc.json"

sys.path.insert(0, str(ROOT / "demo_data"))
sys.path.insert(0, str(YS_DIR))

from characters import CHARACTERS  # noqa: E402

# app.py loads its model bundle from a path relative to its own directory
# (MODEL_PATH = "iip_emc_production_model.pkl"), same as running `python
# app.py` from within apps/YS — so match that working directory here too.
os.chdir(YS_DIR)
import app as ys_app  # noqa: E402


def full_metadata(character: dict) -> dict:
    today = date.today().isoformat()
    meta = dict(character["emc"]["metadata"])
    leave_days = meta["authorized_medical_leave_days"]
    return {
        **meta,
        "patient_id": character["external_id"],
        "consultation_date": today,
        "medical_leave_start_date": today,
        "medical_leave_end_date": (date.today() + timedelta(days=leave_days - 1)).isoformat(),
        "clinician_review_status": "PENDING_REVIEW",
    }


def full_features(symptoms: list[str]) -> dict:
    selected = set(symptoms)
    return {feature: (1 if feature in selected else 0) for feature in ys_app.FEATURE_LAYOUT}


def main():
    results = []
    for character in CHARACTERS:
        metadata = full_metadata(character)
        features = full_features(character["emc"]["symptoms"])
        prediction_result = ys_app.run_prediction(features)

        results.append({
            "external_id": character["external_id"],
            "display_name": character["display_name"],
            "input_payload": {"metadata": metadata, "features": features},
            "output_payload": {"prediction": prediction_result},
        })
        print(f"  {character['display_name']}: {prediction_result['primary_predicted_diagnosis']} "
              f"({prediction_result['prediction_confidence_percentage']}%)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} EMC results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
