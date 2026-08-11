"""Compute real CTRSE triage predictions for the demo roster.

Run with Jace's own venv (it needs Jace's pinned scikit-learn to unpickle
the model bundle):

    apps\\Jace\\.venv\\Scripts\\python.exe demo_data\\compute_triage.py

Calls the same `core.predict_from_fields()` orchestrator apps/Jace/api.py's
POST /api/predict uses (api.py:299) — no HTTP, no server needed. Writes
demo_data/output/triage.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JACE_DIR = ROOT / "apps" / "Jace"
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "triage.json"

sys.path.insert(0, str(ROOT / "demo_data"))
sys.path.insert(0, str(JACE_DIR))

from characters import CHARACTERS  # noqa: E402
import ctrse_core as core  # noqa: E402


def main():
    core.init(str(JACE_DIR))
    core.init_extraction(str(JACE_DIR))

    results = []
    for character in CHARACTERS:
        fields = character["triage"]
        output = core.predict_from_fields(fields)
        results.append({
            "external_id": character["external_id"],
            "display_name": character["display_name"],
            "input_payload": fields,
            "output_payload": output,
        })
        refused = output.get("model_refused")
        level = output.get("predicted_level")
        print(f"  {character['display_name']}: "
              f"{'REFUSED (' + output.get('refusal_reason', '') + ')' if refused else level}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} triage results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
