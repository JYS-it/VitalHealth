"""Compute real stroke-risk predictions for the demo roster.

Run with Jeslyn's own venv (needs Jeslyn's pinned scikit-learn/joblib
artifacts):

    apps\\Jeslyn\\.venv\\Scripts\\python.exe demo_data\\compute_stroke.py

Jeslyn's prediction logic lives inline in its `/prediction` Flask route
(apps/Jeslyn/app.py:471-485) rather than a reusable function, so this
script deliberately mirrors those exact lines against the same pickled
model/scaler artifacts, rather than duplicating a divergent copy. If that
route's logic ever changes, update this script to match. Writes
demo_data/output/stroke.json.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JESLYN_DIR = ROOT / "apps" / "Jeslyn"
MODELS_DIR = JESLYN_DIR / "models"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "stroke.json"

sys.path.insert(0, str(ROOT / "demo_data"))
from characters import CHARACTERS  # noqa: E402

model = joblib.load(MODELS_DIR / "stroke_logistic_model.pkl")
scaler = joblib.load(MODELS_DIR / "stroke_scaler.pkl")
feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
num_cols = joblib.load(MODELS_DIR / "num_cols.pkl")
risk_threshold = float(joblib.load(MODELS_DIR / "risk_threshold.pkl"))


def predict(patient_data: dict) -> dict:
    # Mirrors apps/Jeslyn/app.py:471-485 exactly.
    model_input_df = pd.DataFrame([patient_data])[feature_cols].copy()
    model_input_df["Gender"] = model_input_df["Gender"].map({"Male": 1, "Female": 0})
    model_input_df[num_cols] = scaler.transform(model_input_df[num_cols])
    model_input_df = model_input_df.apply(pd.to_numeric, errors="coerce")

    risk_probability = float(model.predict_proba(model_input_df)[:, 1][0])
    risk_category = "High risk" if risk_probability >= risk_threshold else "Low risk"
    return {
        "risk_category": risk_category,
        "risk_probability_percent": round(risk_probability * 100, 2),
        "threshold_used": float(risk_threshold),
    }


def main():
    results = []
    for character in CHARACTERS:
        patient_data = character["stroke"]
        prediction_result = predict(patient_data)
        results.append({
            "external_id": character["external_id"],
            "display_name": character["display_name"],
            "input_payload": patient_data,
            "output_payload": {"prediction": prediction_result},
        })
        print(f"  {character['display_name']}: {prediction_result['risk_category']} "
              f"({prediction_result['risk_probability_percent']}%)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} stroke results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
