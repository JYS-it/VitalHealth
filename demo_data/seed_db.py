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

from characters import CHARACTERS, DEMO_CLINICIAN, DEMO_PASSWORD, demo_email  # noqa: E402
from vitalhealth_storage import get_store, load_shared_env  # noqa: E402
from vitalhealth_storage.store import records as records_table, users as users_table  # noqa: E402

# Same shared-settings resolution the four apps use, so this works whether
# run_all.py injects DATABASE_URL or you run it straight from a shell.
load_shared_env()


def hash_password(password: str) -> str:
    """bcrypt lives only in the gateway's venv — it is the only process that
    handles credentials, and pulling it into all four would be gratuitous."""
    try:
        import bcrypt
    except ImportError:
        raise SystemExit(
            "bcrypt is required to create the demo logins. Run this script with "
            "the gateway's venv:\n"
            "    apps\\gateway\\.venv\\Scripts\\python.exe demo_data\\seed_db.py"
        )
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def ensure_account(store, *, email, display_name, role, patient_id=None) -> str:
    """Create or reconcile one documented demo login.

    Demo credentials are published in CHEAT_SHEET.md, so an account left over
    from an earlier schema/password/role must be repaired on re-seed instead
    of silently making the documented login unusable. This function is used
    only for the fixed demo roster, never for real user accounts.
    """
    existing = store.get_user_by_email(email)
    if existing:
        with store.engine.begin() as connection:
            connection.execute(
                users_table.update()
                .where(users_table.c.id == existing["id"])
                .values(
                    password_hash=hash_password(DEMO_PASSWORD),
                    role=role,
                    display_name=display_name,
                    patient_id=patient_id,
                )
            )
        return existing["id"]

    return store.create_user(
        email=email,
        password_hash=hash_password(DEMO_PASSWORD),
        role=role,
        display_name=display_name,
        patient_id=patient_id,
    )

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


# Stamped into every seeded output_payload so re-seeding can recognise its own
# rows. Without it this script matched on patient + app + record_type and
# updated the *newest* match — which, once patients could submit for real, was
# a live submission rather than the seeded row, silently reverting an approved
# result to PENDING_REVIEW on every launch. Demo data must never overwrite
# clinical work.
SEED_MARKER_KEY = "seeded_by"
SEED_MARKER_VALUE = "demo_data/seed_db.py"


def seeded_payload(output_payload: dict) -> dict:
    return {**output_payload, SEED_MARKER_KEY: SEED_MARKER_VALUE}


def _is_seeded(record: dict) -> bool:
    payload = record.get("output_payload")
    return isinstance(payload, dict) and payload.get(SEED_MARKER_KEY) == SEED_MARKER_VALUE


def find_existing_record(
    store, patient_id: str, source_app: str, record_type: str, legacy_output: dict | None = None
) -> str | None:
    """The seeded row for this character/app, or None.

    Only ever returns a row this script created. A patient's real submission
    for the same character and module is deliberately invisible here, so
    re-seeding leaves it untouched.

    `legacy_output` adopts rows seeded before the marker existed: a row whose
    payload is byte-identical to the fixture we are about to write is one of
    ours by definition, and adopting it avoids duplicating the whole roster
    once. A real submission never matches, because it carries the patient's
    own answers and a full workflow snapshot.
    """
    with store.engine.begin() as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                records_table.select()
                .where(records_table.c.patient_id == patient_id)
                .where(records_table.c.source_app == source_app)
                .where(records_table.c.record_type == record_type)
                .order_by(records_table.c.created_at.desc())
            ).mappings()
        ]

    for row in rows:
        if _is_seeded(row):
            return row["id"]

    if legacy_output is not None:
        for row in rows:
            if row.get("output_payload") == legacy_output:
                return row["id"]
    return None


def main():
    store = get_store()
    if not store.enabled:
        raise SystemExit("DATABASE_URL is not configured -- set it before seeding.")
    # run_all.py seeds before the gateway process starts. Initialise here so a
    # brand-new PostgreSQL database has its tables and safeguards before the
    # first demo user or record is queried.
    store.initialize()

    results_by_app = {app: load_results(app) for app in APP_CONFIG}

    clinician_id = ensure_account(
        store,
        email=DEMO_CLINICIAN["email"],
        display_name=DEMO_CLINICIAN["display_name"],
        role="clinician",
    )
    print(f"{DEMO_CLINICIAN['display_name']} <{DEMO_CLINICIAN['email']}> -> clinician {clinician_id}")

    for character in CHARACTERS:
        external_id = character["external_id"]
        display_name = character["display_name"]
        patient_id = store.upsert_patient(external_id, display_name)
        email = demo_email(external_id)
        # Owning the records makes the patient dashboard populated on first
        # login rather than empty until they re-run all three modules.
        owner_user_id = ensure_account(
            store,
            email=email,
            display_name=display_name,
            role="patient",
            patient_id=patient_id,
        )
        print(f"{display_name} ({external_id}) -> patient {patient_id}, login <{email}>")

        for source_app, config in APP_CONFIG.items():
            row = results_by_app[source_app].get(external_id)
            if row is None:
                print(f"  [{source_app}] no computed result -- skipped")
                continue

            status = config["status"](row["output_payload"])
            output_payload = seeded_payload(row["output_payload"])
            existing_id = find_existing_record(
                store, patient_id, source_app, config["record_type"],
                legacy_output=row["output_payload"],
            )
            if existing_id:
                existing_record = store.get_record(existing_id) or {}
                unchanged = (
                    existing_record.get("status") == status
                    and existing_record.get("input_payload") == row["input_payload"]
                    and existing_record.get("output_payload") == output_payload
                    and existing_record.get("owner_user_id") == owner_user_id
                )
                if unchanged:
                    print(f"  [{source_app}] already seeded record {existing_id}")
                    continue
                store.update_record(
                    existing_id,
                    status=status,
                    input_payload=row["input_payload"],
                    output_payload=output_payload,
                    # Also set on the update path, or re-seeding a database
                    # created before ownership existed leaves rows unowned.
                    owner_user_id=owner_user_id,
                )
                record_id, verb = existing_id, "updated"
            else:
                record_id = store.create_record(
                    source_app=source_app,
                    record_type=config["record_type"],
                    status=status,
                    input_payload=row["input_payload"],
                    output_payload=output_payload,
                    model_version=config["model_version"],
                    patient_external_id=external_id,
                    patient_name=display_name,
                    owner_user_id=owner_user_id,
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
