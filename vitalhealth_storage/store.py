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

from sqlalchemy import JSON, Column, DateTime, ForeignKey, MetaData, String, Table, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError


metadata = MetaData()
LOGGER = logging.getLogger(__name__)

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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


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
        self.database_url = (database_url or os.getenv("DATABASE_URL", "")).strip()
        options: dict[str, Any] = {"pool_pre_ping": True}
        if self.database_url.startswith("postgresql"):
            options.update(
                pool_size=_positive_int_env("DATABASE_POOL_SIZE", 5),
                max_overflow=_positive_int_env("DATABASE_MAX_OVERFLOW", 5),
                connect_args={"connect_timeout": _positive_int_env("DATABASE_CONNECT_TIMEOUT", 5)},
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
    ) -> None:
        if not self.engine:
            raise RuntimeError("Shared database is not configured.")
        values: dict[str, Any] = {"status": status, "updated_at": _utcnow()}
        if input_payload is not None:
            values["input_payload"] = _json_value(input_payload)
        if output_payload is not None:
            values["output_payload"] = _json_value(output_payload)
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
