"""Database records shared by VitalHealth's independently deployed apps.

The package deliberately stores a small, versioned clinical-record envelope
instead of making one app's internal model tables visible to another app.
Each record keeps its source module, input/output snapshot, and an append-only
audit trail.  The only cross-module identifier is an optional patient external
reference supplied by a clinician-facing workflow.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import lru_cache
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    MetaData,
    String,
    Table,
    create_engine,
    or_,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError


metadata = MetaData()
LOGGER = logging.getLogger(__name__)

# The three clinical modules, in the order a dashboard should present them.
# `source_app` on every record is one of these.
SOURCE_APPS = ("triage", "stroke", "emc")

# ``demo_data.seed_db`` stamps its fixture output with this marker. Fixtures
# are useful on an empty database, but they are not patient submissions.
DEMO_SEED_MARKER_KEY = "seeded_by"
DEMO_SEED_MARKER_VALUE = "demo_data/seed_db.py"

patients = Table(
    "patients",
    metadata,
    # UUIDs are represented as strings for portability between PostgreSQL and
    # local SQLite test databases. PostgreSQL remains the production target.
    # Patient names are optional because triage and stroke inputs may not have
    # a verified patient identifier.
    Column("id", String(36), primary_key=True),
    Column("external_id", String(64), unique=True, nullable=False),
    Column("display_name", String(120), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

records = Table(
    "clinical_records",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("patient_id", String(36), ForeignKey("patients.id"), nullable=True, index=True),
    # Who the record is *about* (patient_id) and who *produced* it
    # (owner_user_id) are different questions. A clinician's assessment of
    # someone else has both, and they point at different people.
    Column("owner_user_id", String(36), ForeignKey("users.id"), nullable=True, index=True),
    Column("source_app", String(32), nullable=False, index=True),
    Column("record_type", String(64), nullable=False),
    Column("status", String(64), nullable=False, index=True),
    Column("model_version", String(128), nullable=True),
    Column("input_payload", JSON().with_variant(JSONB, "postgresql"), nullable=True),
    Column("output_payload", JSON().with_variant(JSONB, "postgresql"), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

audit_events = Table(
    "audit_events",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("record_id", String(36), ForeignKey("clinical_records.id"), nullable=True, index=True),
    Column("source_app", String(32), nullable=False, index=True),
    Column("event_type", String(64), nullable=False),
    Column("actor_reference", String(128), nullable=True),
    Column("event_payload", JSON().with_variant(JSONB, "postgresql"), nullable=True),
    Column("occurred_at", DateTime(timezone=True), nullable=False, index=True),
)

schema_migrations = Table(
    "schema_migrations",
    metadata,
    Column("version", String(64), primary_key=True),
    Column("applied_at", DateTime(timezone=True), nullable=False),
)

users = Table(
    "users",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("email", String(255), unique=True, nullable=False, index=True),
    Column("password_hash", String(255), nullable=False),
    # "patient" or "clinician". The default is the least-privileged role on
    # purpose: a forgotten role= on an insert must never mint an account that
    # can read every patient's records.
    Column("role", String(16), nullable=False, server_default="patient"),
    Column("display_name", String(120), nullable=True),
    # Set for patient accounts only — the patients row this login speaks for.
    Column("patient_id", String(36), ForeignKey("patients.id"), nullable=True, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


def _nonnegative_int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, default)))
    except ValueError:
        return default


def _normalise_database_url(database_url: str) -> str:
    """Make PostgreSQL URLs copied from Supabase work with psycopg 3.

    Supabase presents standard ``postgres://`` / ``postgresql://`` URLs in
    its Connect dialog.  This project installs psycopg 3, so SQLAlchemy needs
    the explicit ``postgresql+psycopg://`` driver prefix instead.
    """
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgres://")
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    return database_url


def _is_supabase_transaction_pooler(database_url: str) -> bool:
    """Whether this URL uses Supavisor's transaction-pooling endpoint."""
    return ".pooler.supabase.com:6543/" in database_url


def _json_value(value: Any) -> Any:
    """Convert model/numpy/date values into JSON-safe primitive values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


class ClinicalStore:
    """A thin repository layer. It is disabled when DATABASE_URL is absent."""

    def __init__(self, database_url: str | None = None):
        raw_database_url = (database_url or os.getenv("DATABASE_URL", "")).strip()
        self.database_url = _normalise_database_url(raw_database_url)
        options: dict[str, Any] = {"pool_pre_ping": True}
        if self.database_url.startswith("postgresql"):
            # Four local web processes connect to the same hosted database.
            # Keep Supabase development usage small unless a deployment
            # explicitly opts into a larger application-side pool.
            is_supabase = "supabase.com" in self.database_url
            connect_args: dict[str, Any] = {
                "connect_timeout": _positive_int_env("DATABASE_CONNECT_TIMEOUT", 10),
            }
            if _is_supabase_transaction_pooler(self.database_url):
                # Supavisor transaction mode does not support prepared
                # statements; psycopg otherwise starts preparing them after a
                # few repetitions.
                connect_args["prepare_threshold"] = None
            options.update(
                pool_size=_positive_int_env("DATABASE_POOL_SIZE", 1 if is_supabase else 5),
                max_overflow=_nonnegative_int_env("DATABASE_MAX_OVERFLOW", 0 if is_supabase else 5),
                connect_args=connect_args,
            )
        self.engine = create_engine(self.database_url, **options) if self.database_url else None

    @property
    def enabled(self) -> bool:
        return self.engine is not None

    def initialize(self) -> None:
        """Apply the idempotent initial schema and database safeguards."""
        if not self.engine:
            raise RuntimeError("DATABASE_URL is required to initialise the shared database.")
        metadata.create_all(self.engine)
        if self.engine.dialect.name != "postgresql":
            return
        with self.engine.begin() as connection:
            # These compound indexes serve the two expected retrieval paths:
            # one patient's chronology, and records emitted by one app.
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_clinical_records_patient_created "
                "ON clinical_records (patient_id, created_at DESC)"
            ))
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_clinical_records_source_created "
                "ON clinical_records (source_app, created_at DESC)"
            ))
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_audit_events_record_occurred "
                "ON audit_events (record_id, occurred_at DESC)"
            ))
            # Audit entries are intentionally immutable after insertion. The
            # application role can append events but cannot silently rewrite
            # or remove their history through ordinary SQL.
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION vitalhealth_prevent_audit_mutation()
                RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION 'audit_events are append-only';
                END;
                $$ LANGUAGE plpgsql;
            """))
            connection.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                        WHERE tgname = 'trg_audit_events_append_only'
                    ) THEN
                        CREATE TRIGGER trg_audit_events_append_only
                        BEFORE UPDATE OR DELETE ON audit_events
                        FOR EACH ROW EXECUTE FUNCTION vitalhealth_prevent_audit_mutation();
                    END IF;
                END;
                $$;
            """))
            connection.execute(text("""
                INSERT INTO schema_migrations (version, applied_at)
                VALUES ('2026_08_10_initial_hardening', CURRENT_TIMESTAMP)
                ON CONFLICT (version) DO NOTHING
            """))
            self._migrate_roles_and_ownership(connection)

    def _migrate_roles_and_ownership(self, connection: Any) -> None:
        """Add the role/ownership columns to a database created before them.

        create_all() only ever creates missing *tables*, so a database that
        already has `users` will never gain the new columns from the table
        definitions alone. These statements are the actual migration; on a
        fresh database they are all no-ops.
        """
        for statement in (
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(16) NOT NULL DEFAULT 'patient'",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS display_name VARCHAR(120)",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS patient_id VARCHAR(36)",
            "ALTER TABLE clinical_records ADD COLUMN IF NOT EXISTS owner_user_id VARCHAR(36)",
            "CREATE INDEX IF NOT EXISTS ix_users_patient ON users (patient_id)",
            "CREATE INDEX IF NOT EXISTS ix_clinical_records_owner "
            "ON clinical_records (owner_user_id, created_at DESC)",
        ):
            connection.execute(text(statement))

        # Foreign keys have no ADD CONSTRAINT IF NOT EXISTS, so they get the
        # same pg_catalog guard the audit trigger above uses.
        connection.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_users_patient') THEN
                    ALTER TABLE users ADD CONSTRAINT fk_users_patient
                    FOREIGN KEY (patient_id) REFERENCES patients (id);
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_records_owner_user') THEN
                    ALTER TABLE clinical_records ADD CONSTRAINT fk_records_owner_user
                    FOREIGN KEY (owner_user_id) REFERENCES users (id);
                END IF;
            END;
            $$;
        """))

        # Accounts that predate roles are the team's own staff logins, so they
        # become clinicians. Guarded by the migration row so it runs exactly
        # once and can never come back later to promote a patient who signed
        # up afterwards. Must precede the INSERT that records the version.
        connection.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM schema_migrations
                    WHERE version = '2026_08_12_roles_and_ownership'
                ) THEN
                    UPDATE users SET role = 'clinician' WHERE role = 'patient';
                END IF;
            END;
            $$;
        """))
        connection.execute(text("""
            INSERT INTO schema_migrations (version, applied_at)
            VALUES ('2026_08_12_roles_and_ownership', CURRENT_TIMESTAMP)
            ON CONFLICT (version) DO NOTHING
        """))

    def upsert_patient(self, external_id: str, display_name: str | None = None) -> str:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        now = _utcnow()
        with self.engine.begin() as connection:
            row = connection.execute(
                patients.select().where(patients.c.external_id == external_id)
            ).mappings().first()
            if row:
                connection.execute(
                    patients.update().where(patients.c.id == row["id"]).values(
                        display_name=display_name or row["display_name"], updated_at=now
                    )
                )
                return row["id"]
            patient_id = str(uuid.uuid4())
            connection.execute(
                patients.insert().values(
                    id=patient_id,
                    external_id=external_id,
                    display_name=display_name,
                    created_at=now,
                    updated_at=now,
                )
            )
            return patient_id

    def create_user(
        self,
        *,
        email: str,
        password_hash: str,
        role: str = "patient",
        display_name: str | None = None,
        patient_id: str | None = None,
    ) -> str:
        """Insert a new user. Raises on a duplicate email rather than
        swallowing the error — unlike the safe_* wrappers below, auth has
        no valid fallback when persistence fails."""
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        user_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                users.insert().values(
                    id=user_id,
                    email=email.strip().lower(),
                    password_hash=password_hash,
                    role=role,
                    display_name=display_name,
                    patient_id=patient_id,
                    created_at=_utcnow(),
                )
            )
        return user_id

    def get_user_by_email(self, email: str) -> Any:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        with self.engine.begin() as connection:
            return connection.execute(
                users.select().where(users.c.email == email.strip().lower())
            ).mappings().first()

    def get_user_by_id(self, user_id: str) -> dict | None:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        with self.engine.begin() as connection:
            row = connection.execute(
                users.select().where(users.c.id == user_id)
            ).mappings().first()
        return dict(row) if row else None

    def link_user_patient(self, user_id: str, patient_id: str) -> None:
        """Point a login at the patients row it speaks for."""
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        with self.engine.begin() as connection:
            connection.execute(
                users.update().where(users.c.id == user_id).values(patient_id=patient_id)
            )

    def create_record(
        self,
        *,
        source_app: str,
        record_type: str,
        status: str,
        input_payload: dict | None = None,
        output_payload: dict | None = None,
        model_version: str | None = None,
        patient_external_id: str | None = None,
        patient_name: str | None = None,
        owner_user_id: str | None = None,
    ) -> str:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        patient_id = None
        if patient_external_id:
            patient_id = self.upsert_patient(patient_external_id, patient_name)
        record_id, now = str(uuid.uuid4()), _utcnow()
        with self.engine.begin() as connection:
            connection.execute(
                records.insert().values(
                    id=record_id,
                    patient_id=patient_id,
                    owner_user_id=owner_user_id,
                    source_app=source_app,
                    record_type=record_type,
                    status=status,
                    model_version=model_version,
                    input_payload=_json_value(input_payload),
                    output_payload=_json_value(output_payload),
                    created_at=now,
                    updated_at=now,
                )
            )
        return record_id

    def update_record(
        self,
        record_id: str,
        *,
        status: str,
        input_payload: dict | None = None,
        output_payload: dict | None = None,
        owner_user_id: str | None = None,
    ) -> None:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        values: dict[str, Any] = {"status": status, "updated_at": _utcnow()}
        if input_payload is not None:
            values["input_payload"] = _json_value(input_payload)
        if output_payload is not None:
            values["output_payload"] = _json_value(output_payload)
        if owner_user_id is not None:
            values["owner_user_id"] = owner_user_id
        with self.engine.begin() as connection:
            result = connection.execute(records.update().where(records.c.id == record_id).values(**values))
            if result.rowcount != 1:
                raise KeyError(f"Unknown clinical record: {record_id}")

    def append_audit_event(
        self,
        *,
        source_app: str,
        event_type: str,
        record_id: str | None = None,
        actor_reference: str | None = None,
        payload: dict | None = None,
    ) -> str:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        event_id = str(uuid.uuid4())
        with self.engine.begin() as connection:
            connection.execute(
                audit_events.insert().values(
                    id=event_id,
                    record_id=record_id,
                    source_app=source_app,
                    event_type=event_type,
                    actor_reference=actor_reference,
                    event_payload=_json_value(payload),
                    occurred_at=_utcnow(),
                )
            )
        return event_id

    # ---- read paths (dashboards) -------------------------------------------
    #
    # These deliberately have no safe_* twin. A swallowed *write* costs one
    # record; a swallowed *read* shows a clinician an empty chart and lets them
    # conclude the patient has no history. Let SQLAlchemyError propagate and
    # have the API turn it into a visible "records unavailable" response.

    def _require_engine(self) -> Any:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        return self.engine

    def get_patient(self, patient_id: str) -> dict | None:
        with self._require_engine().begin() as connection:
            row = connection.execute(
                patients.select().where(patients.c.id == patient_id)
            ).mappings().first()
        return dict(row) if row else None

    def get_patient_by_external_id(self, external_id: str) -> dict | None:
        with self._require_engine().begin() as connection:
            row = connection.execute(
                patients.select().where(patients.c.external_id == external_id)
            ).mappings().first()
        return dict(row) if row else None

    def get_record(self, record_id: str) -> dict | None:
        with self._require_engine().begin() as connection:
            row = connection.execute(
                records.select().where(records.c.id == record_id)
            ).mappings().first()
        return dict(row) if row else None

    def list_records(
        self,
        *,
        patient_id: str | None = None,
        owner_user_id: str | None = None,
        source_app: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Newest first.

        patient_id and owner_user_id are OR-ed, not AND-ed: "records about me"
        and "records I created" diverge for seeded data, for rows written
        before ownership existed, and any time subject resolution fails. A
        patient should see both.
        """
        statement = records.select()
        subject_filters = []
        if patient_id:
            subject_filters.append(records.c.patient_id == patient_id)
        if owner_user_id:
            subject_filters.append(records.c.owner_user_id == owner_user_id)
        if subject_filters:
            statement = statement.where(or_(*subject_filters))
        if source_app:
            statement = statement.where(records.c.source_app == source_app)
        if status:
            statement = statement.where(records.c.status == status)
        statement = statement.order_by(records.c.created_at.desc()).limit(limit)

        with self._require_engine().begin() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def latest_record_per_app(
        self,
        *,
        patient_id: str | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, dict | None]:
        """The most recent record from each module — one dashboard tile each."""
        latest: dict[str, dict | None] = {app: None for app in SOURCE_APPS}
        if not patient_id and not owner_user_id:
            return latest
        # The roster is small enough that one ordered pass beats a window
        # function here, and it keeps the SQL readable.
        for row in self.list_records(
            patient_id=patient_id, owner_user_id=owner_user_id, limit=500
        ):
            app = row.get("source_app")
            if app in latest and latest[app] is None:
                latest[app] = row
        return latest

    def list_patient_summaries(
        self,
        *,
        search: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """One row per patient for the clinician dashboard.

        Driven from `patients` rather than from records so a patient who has
        registered but not yet run anything still appears — an empty chart is
        a legitimate state, not an absent one.
        """
        engine = self._require_engine()

        patient_query = patients.select()
        if search and search.strip():
            pattern = f"%{search.strip()}%"
            patient_query = patient_query.where(
                or_(
                    patients.c.display_name.ilike(pattern),
                    patients.c.external_id.ilike(pattern),
                )
            )
        patient_query = patient_query.order_by(patients.c.display_name).limit(limit)

        with engine.begin() as connection:
            patient_rows = [dict(row) for row in connection.execute(patient_query).mappings()]
            if not patient_rows:
                return []

            ids = [row["id"] for row in patient_rows]
            record_rows = [
                dict(row)
                for row in connection.execute(
                    records.select()
                    .where(records.c.patient_id.in_(ids))
                    .order_by(records.c.created_at.desc())
                ).mappings()
            ]

        summaries = {
            row["id"]: {
                "patient_id": row["id"],
                "external_id": row["external_id"],
                "display_name": row["display_name"],
                "modules": {app: None for app in SOURCE_APPS},
                "completed_modules": 0,
                "record_count": 0,
                "last_activity": None,
            }
            for row in patient_rows
        }

        for record in record_rows:
            summary = summaries.get(record["patient_id"])
            if summary is None:
                continue
            summary["record_count"] += 1
            app = record.get("source_app")
            if app in summary["modules"] and summary["modules"][app] is None:
                summary["modules"][app] = record
                summary["completed_modules"] += 1
            created = record.get("created_at")
            if created and (summary["last_activity"] is None or created > summary["last_activity"]):
                summary["last_activity"] = created

        return [summaries[row["id"]] for row in patient_rows]

    def list_unassigned_records(self, *, limit: int = 100) -> list[dict]:
        """Records with no patient attached.

        Triage and stroke have no patient field on their forms, so anything
        run without a resolved subject lands here rather than vanishing.
        """
        statement = (
            records.select()
            .where(records.c.patient_id.is_(None))
            .order_by(records.c.created_at.desc())
            .limit(limit)
        )
        with self._require_engine().begin() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def list_pending_review(
        self,
        *,
        source_apps: tuple[str, ...] = ("stroke", "emc"),
        status: str = "PENDING_REVIEW",
        limit: int = 100,
    ) -> list[dict]:
        """The open, shared clinician review queue: every patient-submitted
        item awaiting action, across modules, newest first.

        Triage is excluded by the default source_apps — a clinician's own
        triage run stays instant, and a patient's self-check (status
        PATIENT_SELF_CHECK, not PENDING_REVIEW — see ctrse_core.patient_view)
        is deliberately never queued here either, since nothing is expected
        to action it. Both exclusions are belt-and-braces: the status filter
        alone already rules a self-check out. No per-item claiming: any
        clinician sees the same list, and a row simply drops off once its
        status moves past PENDING_REVIEW.
        """
        statement = (
            records.select()
            .where(records.c.source_app.in_(source_apps))
            .where(records.c.status == status)
            .order_by(records.c.created_at.desc())
        )
        with self._require_engine().begin() as connection:
            rows = [dict(row) for row in connection.execute(statement).mappings()]

        def is_demo_fixture(row: dict) -> bool:
            payload = row.get("output_payload")
            return (
                isinstance(payload, dict)
                and payload.get(DEMO_SEED_MARKER_KEY) == DEMO_SEED_MARKER_VALUE
            )

        # Seeded EMC fixtures are intentionally PENDING_REVIEW so the patient
        # history is realistic. They are not real submissions, however, and
        # must not clutter the shared clinician work queue.
        rows = [row for row in rows if not is_demo_fixture(row)][:limit]

        patient_ids = {row["patient_id"] for row in rows if row.get("patient_id")}
        if patient_ids:
            with self._require_engine().begin() as connection:
                patient_rows = {
                    p["id"]: p
                    for p in connection.execute(
                        patients.select().where(patients.c.id.in_(patient_ids))
                    ).mappings()
                }
            for row in rows:
                patient = patient_rows.get(row.get("patient_id"))
                row["patient_display_name"] = (patient or {}).get("display_name")
                row["patient_external_id"] = (patient or {}).get("external_id")
        else:
            for row in rows:
                row["patient_display_name"] = None
                row["patient_external_id"] = None

        return rows

    def list_audit_events(self, record_id: str, *, limit: int = 50) -> list[dict]:
        statement = (
            audit_events.select()
            .where(audit_events.c.record_id == record_id)
            .order_by(audit_events.c.occurred_at.desc())
            .limit(limit)
        )
        with self._require_engine().begin() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

    def safe_create_record(self, **kwargs: Any) -> str | None:
        """Persistence must not turn an otherwise valid clinical workflow into a 500."""
        if not self.enabled:
            return None
        try:
            return self.create_record(**kwargs)
        except (RuntimeError, SQLAlchemyError) as exc:
            LOGGER.warning("Shared database record write failed: %s", type(exc).__name__)
            return None

    def safe_update_record(self, record_id: str | None, **kwargs: Any) -> bool:
        if not self.enabled or not record_id:
            return False
        try:
            self.update_record(record_id, **kwargs)
            return True
        except (KeyError, RuntimeError, SQLAlchemyError) as exc:
            LOGGER.warning("Shared database record update failed: %s", type(exc).__name__)
            return False

    def safe_append_audit_event(self, **kwargs: Any) -> str | None:
        if not self.enabled:
            return None
        try:
            return self.append_audit_event(**kwargs)
        except (RuntimeError, SQLAlchemyError) as exc:
            LOGGER.warning("Shared database audit write failed: %s", type(exc).__name__)
            return None


@lru_cache(maxsize=1)
def get_store() -> ClinicalStore:
    return ClinicalStore()


def validate_json_payload(value: Any) -> None:
    """Useful for tests and migration smoke checks."""
    json.dumps(_json_value(value))
