"""Seed the shared VitalHealth database with the demo roster's real model
outputs (already computed into demo_data/output/*.json by compute_triage.py
/ compute_stroke.py / compute_emc.py -- run those three first, each with
its own app's venv).

Run with any venv that has vitalhealth_storage installed (e.g. the
gateway's):

    apps\\gateway\\.venv\\Scripts\\python.exe demo_data\\seed_db.py

Always safe to re-run: patients are upserted by external_id, and each
character's per-app clinical_record is updated in place rather than
duplicated if one already exists (looked up by patient + source_app +
record_type). audit_events are intentionally append-only at the database
level (see vitalhealth_storage/store.py's
vitalhealth_prevent_audit_mutation trigger -- it blocks UPDATE and DELETE
outright), so re-seeding appends a fresh audit event each time instead of
replacing old ones. That's not a limitation to work around: it's the same
invariant a real re-assessment would produce, and this script deliberately
doesn't fight it.
"""
import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
OUTPUT_DIR = DEMO_DIR / "output"

sys.path.insert(0, str(DEMO_DIR))
sys.path.insert(0, str(ROOT))

from characters import CHARACTERS  # noqa: E402
from vitalhealth_storage import get_store  # noqa: E402
from vitalhealth_storage.store import records as records_table  # noqa: E402

APP_CONFIG = {
    "triage": {
        "file": "triage.json",
        "record_type": "triage_assessment",
        "model_version": "ctrse_p1p4",
        "event_type": "triage_assessed",
        "status": lambda out: "REFUSED" if out.get("model_refused") else "ASSESSED",
        "audit_payload": lambda out: {"predicted_level": out.get("predicted_level")},
    },
    "stroke": {
        "file": "stroke.json",
        "record_type": "stroke_risk_assessment",
        "model_version": "stroke_logistic_model",
        "event_type": "stroke_assessed",
        "status": lambda out: "ASSESSED",
        "audit_payload": lambda out: {"risk_category": out["prediction"]["risk_category"]},
    },
    "emc": {
        "file": "emc.json",
        "record_type": "electronic_medical_certificate",
        "model_version": "webapp_genai_emc_v4.0-grounded",
        "event_type": "emc_draft_created",
        "status": lambda out: "PENDING_REVIEW",
        "audit_payload": lambda out: {
            "predicted_diagnosis": out["prediction"]["primary_predicted_diagnosis"]
        },
    },
}


def load_results(source_app: str) -> dict:
    path = OUTPUT_DIR / APP_CONFIG[source_app]["file"]
    if not path.exists():
        raise SystemExit(
            f"Missing {path} -- run demo_data/compute_{source_app}.py "
            f"with the {source_app} app's venv first."
        )
    return {row["external_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))}


def find_existing_record(store, patient_id: str, source_app: str, record_type: str) -> str | None:
    with store.engine.begin() as connection:
        row = connection.execute(
            records_table.select()
            .where(records_table.c.patient_id == patient_id)
            .where(records_table.c.source_app == source_app)
            .where(records_table.c.record_type == record_type)
            .order_by(records_table.c.created_at.desc())
            .limit(1)
        ).mappings().first()
    return row["id"] if row else None


def main():
    store = get_store()
    if not store.enabled:
        raise SystemExit("DATABASE_URL is not configured -- set it before seeding.")

    results_by_app = {app: load_results(app) for app in APP_CONFIG}

    for character in CHARACTERS:
        external_id = character["external_id"]
        display_name = character["display_name"]
        patient_id = store.upsert_patient(external_id, display_name)
        print(f"{display_name} ({external_id}) -> patient {patient_id}")

        for source_app, config in APP_CONFIG.items():
            row = results_by_app[source_app].get(external_id)
            if row is None:
                print(f"  [{source_app}] no computed result -- skipped")
                continue

            status = config["status"](row["output_payload"])
            existing_id = find_existing_record(store, patient_id, source_app, config["record_type"])
            if existing_id:
                store.update_record(
                    existing_id,
                    status=status,
                    input_payload=row["input_payload"],
                    output_payload=row["output_payload"],
                )
                record_id, verb = existing_id, "updated"
            else:
                record_id = store.create_record(
                    source_app=source_app,
                    record_type=config["record_type"],
                    status=status,
                    input_payload=row["input_payload"],
                    output_payload=row["output_payload"],
                    model_version=config["model_version"],
                    patient_external_id=external_id,
                    patient_name=display_name,
                )
                verb = "created"

            store.append_audit_event(
                source_app=source_app,
                event_type=config["event_type"],
                record_id=record_id,
                actor_reference="demo_data/seed_db.py",
                payload=config["audit_payload"](row["output_payload"]),
            )
            print(f"  [{source_app}] {verb} record {record_id} -> {status}")

    print("Done.")


if __name__ == "__main__":
    main()
