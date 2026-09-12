"""Persistent study-scoped holdout consumption, reserved before execution.

SQLite transactions prevent concurrent reuse. Reservations survive failures and
restarts; they are never automatically released. This is an audit guard for a
trusted local workspace, not tamper-proof storage or global data lineage.
"""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import re
import sqlite3
import uuid

from .graph_confirmation import FrozenGraphModel, HeldoutCases
from .model_hypotheses import HypothesisValidationError, _require
from .solver_runtime import SolverLimits, SolverProcessRunner, SolverRuntimeError, failure_details


def validate_study_id(study: str) -> None:
    _require(type(study) is str and bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", study)), "invalid_study_id")


class ConfirmationRegistry:
    def __init__(self, database: Path):
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            _require(not tables or "metadata" in tables, "unrecognized_confirmation_registry")
            if tables:
                version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
                _require(version is not None and version[0] == "1", "confirmation_registry_version_mismatch")
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            _require(version is None or version[0] == "1", "confirmation_registry_version_mismatch")
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('schema_version', '1')")
            connection.execute("""CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, study TEXT NOT NULL, model_hash TEXT NOT NULL, data_hash TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT, outcome TEXT)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS consumed_points (
                study TEXT NOT NULL, point_hash TEXT NOT NULL, attempt_id TEXT NOT NULL,
                PRIMARY KEY (study, point_hash), FOREIGN KEY (attempt_id) REFERENCES attempts(id))""")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.database), timeout=5.0)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def reserve(self, study: str, model: FrozenGraphModel, data: HeldoutCases) -> str:
        validate_study_id(study)
        # Revalidation prevents data constructed for another model from bypassing overlap.
        data = HeldoutCases.from_payload(data.public(), model)
        return self._reserve(study, model.digest, data.digest, data.point_hashes)

    def reserve_panel(self, study, panel, payload):
        from .graph_panel import FrozenModelPanel
        panel = FrozenModelPanel.from_payload(panel.public())
        data = panel.validate_holdout(payload)
        return self._reserve(study, panel.digest, data.digest, data.point_hashes)

    def _reserve(self, study, model_hash, data_hash, point_hashes):
        validate_study_id(study)
        attempt = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("INSERT INTO attempts(id, study, model_hash, data_hash, status) VALUES (?,?,?,?,?)",
                                   (attempt, study, model_hash, data_hash, "reserved"))
                connection.executemany("INSERT INTO consumed_points(study, point_hash, attempt_id) VALUES (?,?,?)",
                                       [(study, point, attempt) for point in point_hashes])
            except sqlite3.IntegrityError as exc:
                # Context manager rolls back the whole attempt and all new rows.
                raise HypothesisValidationError("holdout_already_consumed_in_study") from exc
        return attempt

    def complete(self, attempt: str, outcome: str) -> None:
        _require(outcome in ("passed_finite_heldout_checks", "failed_heldout_checks", "execution_incomplete", "panel_completed"),
                 "invalid_confirmation_outcome")
        with self._connect() as connection:
            cursor = connection.execute("UPDATE attempts SET status='completed', completed_at=CURRENT_TIMESTAMP, outcome=? "
                                        "WHERE id=? AND status='reserved'", (outcome, attempt))
            _require(cursor.rowcount == 1, "confirmation_attempt_not_reserved")

    def inspect(self, attempt: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT study, model_hash, data_hash, status, outcome FROM attempts WHERE id=?", (attempt,)).fetchone()
        _require(row is not None, "unknown_confirmation_attempt")
        return {"attempt_id": attempt, **dict(zip(("study", "model_hash", "data_hash", "status", "outcome"), row))}


def confirm_frozen_model(model: FrozenGraphModel, payload: dict, *, registry: ConfirmationRegistry,
                         study: str, cancel=None, limits: SolverLimits | None = None) -> dict:
    """One-way handoff. Failed/uncertain execution still consumes this holdout."""
    model = FrozenGraphModel.from_payload(model.public())
    data = HeldoutCases.from_payload(payload, model)
    attempt = registry.reserve(study, model, data)
    result = _execute_frozen(model, data, cancel=cancel, limits=limits)
    # Unexpected exceptions leave the reservation consumed, never fresh for retry.
    registry.complete(attempt, result["status"])
    result["consumption"] = {**registry.inspect(attempt), "guard_scope": "same_study_same_registry",
                             "case_id_order_or_target_change_does_not_reset": True,
                             "outside_study_freshness_verified": False, "local_store_tamper_proof": False}
    result["may_feed_search"] = False
    return result


def _execute_frozen(model, data, *, cancel=None, limits=None):
    """Internal evaluator; caller must reserve single-model or whole-panel use first."""
    try:
        result = SolverProcessRunner().execute("scalar_graph_confirm/v1", {
            "frozen_model": model.public(), "holdout": data.public(),
        }, limits=limits or SolverLimits(wall_seconds=30, max_evaluations=1000), cancel=cancel)
        _require(result.get("frozen_model_hash") == model.digest and result.get("holdout_hash") == data.digest,
                 "confirmation_result_scope_mismatch")
        _require(result.get("parameters") == model.public()["parameters"] and result.get("parameters_refitted") is False,
                 "confirmation_parameters_changed")
        _require(result.get("status") in ("passed_finite_heldout_checks", "failed_heldout_checks"),
                 "invalid_confirmation_outcome")
    except (SolverRuntimeError, HypothesisValidationError) as exc:
        code = exc.code if isinstance(exc, SolverRuntimeError) else "invalid_response"
        result = {"schema_version": "mathmodel.scalar-confirmation/v1", "status": "execution_incomplete",
                  "frozen_model_hash": model.digest, "holdout_hash": data.digest, "failure_code": code,
                  "may_feed_search": False, "parameters_refitted": False, **failure_details(code)}
    # An interruption or unexpected Python failure leaves a durable 'reserved'
    # record. It must not be retried as fresh independent evidence.
    result["may_feed_search"] = False
    return result
