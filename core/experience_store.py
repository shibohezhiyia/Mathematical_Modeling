"""Content-addressed, privacy-aware store for validated modeling recipes.

The store is deliberately a *seed* repository, not a second truth source.  A
recipe can be retrieved to initialise a new search, but its ``verified`` flag
is only scoped to the evidence recorded for that recipe and never authorises a
new task to skip type, unit, resource, or holdout checks.

Raw observations and credentials are not persisted by default.  Payloads are
redacted before hashing, so the hash also identifies exactly the safe object
that was stored rather than a caller-owned mutable object.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPERIENCE_SCHEMA_VERSION = "mathmodel.experience/v1"
STORE_SCHEMA_VERSION = "mathmodel.experience-store/v1"
_STATUSES = frozenset({"provisional", "verified", "rejected"})
_PURGEABLE_STATUSES = frozenset({"provisional", "rejected"})
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|bearer|"
    r"client[_-]?secret|password|passwd|secret|private[_-]?key|cookie)", re.I
)
_RAW_DATA_KEY_RE = re.compile(
    r"(?:raw[_-]?data|dataset[_-]?rows?|observations?|records?|samples?)$", re.I
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:bearer\s+|sk-|rk-|AIza[0-9A-Za-z_-]{8,}|ds-[0-9A-Za-z_-]{8,})"
    r"[A-Za-z0-9._~+/=-]{8,}"
)
_WINDOWS_PATH_RE = re.compile(r"(?i)^[A-Z]:[\\/].*")


class ExperienceStoreError(ValueError):
    """Raised for malformed recipes, unsafe lifecycle transitions, or storage errors."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperienceStoreError("经验配方必须是有限、可 JSON 序列化的数据") from exc


def _tokens(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = [part for part in re.split(r"[^A-Za-z0-9_:.\-]+", value) if part]
    elif isinstance(value, Mapping):
        raw = [f"{key}:{item}" for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
               for item in (item if isinstance(item, (list, tuple, set)) else [item])]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = [str(value)]
    return tuple(sorted({item.strip().lower() for item in raw if str(item).strip()}))


def _redact_string(value: str) -> str:
    text = str(value)
    text = _SECRET_VALUE_RE.sub("<redacted-secret>", text)
    # Absolute local paths are implementation details and may contain usernames.
    if _WINDOWS_PATH_RE.match(text) or text.startswith(("/", "\\\\")):
        return "<redacted-local-path>"
    return text


def redact_payload(value: Any, *, _key: str | None = None, _depth: int = 0,
                   _state: dict[str, Any] | None = None) -> Any:
    """Return a JSON-safe copy with credentials, raw rows and local paths removed.

    The function is conservative about keys but does not redact ordinary math
    text.  Callers should still pass a recipe rather than an entire workspace.
    """
    state = _state if _state is not None else {"nodes": 0, "active": set()}
    state["nodes"] += 1
    if _depth > 32 or state["nodes"] > 100_000:
        raise ExperienceStoreError("经验配方结构超过脱敏预算")
    container = isinstance(value, Mapping) or isinstance(value, (list, tuple, set))
    marker = id(value) if container else None
    if marker is not None:
        if marker in state["active"]:
            raise ExperienceStoreError("经验配方包含循环引用")
        state["active"].add(marker)
    try:
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            if len(value) > 10_000:
                raise ExperienceStoreError("经验配方字段数量超过脱敏预算")
            for raw_key, raw_value in value.items():
                if not isinstance(raw_key, str) or not raw_key.strip() or len(raw_key) > 256:
                    raise ExperienceStoreError("经验配方字段名无效")
                key = raw_key.strip()
                if _SENSITIVE_KEY_RE.search(key):
                    output[key] = "<redacted-secret>"
                elif _RAW_DATA_KEY_RE.search(key):
                    output[key] = {"redacted": True, "reason": "raw_observations_not_persisted"}
                else:
                    output[key] = redact_payload(raw_value, _key=key, _depth=_depth + 1, _state=state)
            return output
        if isinstance(value, (list, tuple, set)):
            if len(value) > 10_000:
                raise ExperienceStoreError("经验配方列表超过脱敏预算")
            return [redact_payload(item, _key=_key, _depth=_depth + 1, _state=state) for item in value]
    finally:
        if marker is not None:
            state["active"].remove(marker)
    if isinstance(value, Path):
        return "<redacted-local-path>"
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ExperienceStoreError("经验配方不能包含 NaN 或无穷值")
        return value
    raise ExperienceStoreError(f"经验配方包含不可序列化类型: {type(value).__name__}")


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ExperienceStore:
    """SQLite-indexed content-addressed recipe store with safe deletion."""

    def __init__(self, root: os.PathLike[str] | str):
        self.root = Path(root).resolve()
        self.objects = self.root / "objects"
        self.tmp = self.root / "tmp"
        self.database = self.root / "index.sqlite3"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    object_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('provisional','verified','rejected')),
                    created_at TEXT NOT NULL,
                    verified_at TEXT,
                    run_id TEXT,
                    topic_key TEXT,
                    graph_signature TEXT NOT NULL,
                    unit_signature TEXT NOT NULL,
                    objective_signature TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    verification TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_experiences_run ON experiences(run_id);
                CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at);
                CREATE INDEX IF NOT EXISTS idx_experiences_topic ON experiences(topic_key);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)",
                (STORE_SCHEMA_VERSION,),
            )
            # Older local stores may have been created by the first development
            # snapshot.  The additive migration keeps their recipes readable.
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(experiences)")}
            if "verification" not in columns:
                connection.execute("ALTER TABLE experiences ADD COLUMN verification TEXT")

    @staticmethod
    def _validate_identifier(value: str | None, label: str) -> str | None:
        if value is None:
            return None
        text = str(value)
        if not _ID_RE.fullmatch(text):
            raise ExperienceStoreError(f"{label} 不是安全标识符")
        return text

    @staticmethod
    def _safe_signature(value: Any, label: str) -> tuple[str, ...]:
        tokens = _tokens(value)
        if len(tokens) > 256 or any(len(token) > 160 for token in tokens):
            raise ExperienceStoreError(f"{label} 过大或包含超长签名项")
        return tokens

    def put(
        self,
        recipe: Mapping[str, Any],
        *,
        graph_signature: Any = (),
        unit_signature: Any = (),
        objective_signature: Any = (),
        run_id: str | None = None,
        topic_key: str | None = None,
        status: str = "provisional",
    ) -> dict[str, Any]:
        """Persist a redacted recipe and return its immutable content identity."""
        if not isinstance(recipe, Mapping):
            raise ExperienceStoreError("recipe 必须是对象")
        if status not in _STATUSES or status == "verified":
            raise ExperienceStoreError("新经验只能以 provisional 或 rejected 状态写入")
        safe_run = self._validate_identifier(run_id, "run_id")
        safe_topic = self._validate_identifier(topic_key, "topic_key")
        signatures = {
            "graph": self._safe_signature(graph_signature, "graph_signature"),
            "units": self._safe_signature(unit_signature, "unit_signature"),
            "objective": self._safe_signature(objective_signature, "objective_signature"),
        }
        payload = {
            "schema_version": EXPERIENCE_SCHEMA_VERSION,
            "kind": "validated_math_recipe",
            "recipe": redact_payload(recipe),
            "signatures": {key: list(value) for key, value in signatures.items()},
        }
        content = _canonical_bytes(payload)
        experience_id = hashlib.sha256(content).hexdigest()
        object_path = self.objects / experience_id[:2] / f"{experience_id}.json"
        object_path.parent.mkdir(parents=True, exist_ok=True)
        if not object_path.exists():
            descriptor, temporary_name = tempfile.mkstemp(prefix=".recipe-", suffix=".tmp", dir=self.tmp)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                temporary.write_bytes(content)
                os.replace(str(temporary), str(object_path))
            finally:
                temporary.unlink(missing_ok=True)
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO experiences(
                    experience_id,object_hash,status,created_at,run_id,topic_key,
                    graph_signature,unit_signature,objective_signature,schema_version
                ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (experience_id, experience_id, status, created_at, safe_run, safe_topic,
                 json.dumps(signatures["graph"]), json.dumps(signatures["units"]),
                 json.dumps(signatures["objective"]), EXPERIENCE_SCHEMA_VERSION),
            )
        return self.get(experience_id, include_recipe=False)

    def _row_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "experience_id": row["experience_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "verified_at": row["verified_at"],
            "run_id": row["run_id"],
            "topic_key": row["topic_key"],
            "graph_signature": json.loads(row["graph_signature"]),
            "unit_signature": json.loads(row["unit_signature"]),
            "objective_signature": json.loads(row["objective_signature"]),
            "schema_version": row["schema_version"],
            "verification": json.loads(row["verification"]) if row["verification"] else None,
            "seed_only": True,
            "requires_current_validation": True,
        }

    def get(self, experience_id: str, *, include_recipe: bool = True) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{64}", str(experience_id)):
            raise ExperienceStoreError("experience_id 不是 SHA-256 内容地址")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiences WHERE experience_id=?", (experience_id,)
            ).fetchone()
        if row is None:
            raise ExperienceStoreError("经验不存在")
        path = self.objects / experience_id[:2] / f"{experience_id}.json"
        if not path.is_file():
            raise ExperienceStoreError("经验对象缺失，索引与内容不一致")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != experience_id:
            raise ExperienceStoreError("经验对象校验和不匹配")
        result = self._row_summary(row)
        if include_recipe:
            try:
                result["payload"] = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExperienceStoreError("经验对象不是有效 JSON") from exc
        return result

    def promote_verified(self, experience_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        """Promote only with explicit replay, counterexample, and security evidence."""
        if not isinstance(evidence, Mapping):
            raise ExperienceStoreError("验证证据必须是对象")
        required = ("replay_passed", "counterexamples_checked", "security_checked")
        if any(evidence.get(key) is not True for key in required):
            raise ExperienceStoreError("verified 需要复现、反例和安全检查均为 true")
        scope = evidence.get("validation_scope")
        if not isinstance(scope, Mapping) or not scope:
            raise ExperienceStoreError("verified 必须记录非空 validation_scope")
        current = self.get(experience_id, include_recipe=False)
        if current["status"] == "rejected":
            raise ExperienceStoreError("已拒绝经验不能直接晋级")
        safe_evidence = redact_payload(evidence)
        with self._connect() as connection:
            connection.execute(
                "UPDATE experiences SET status='verified', verified_at=?, verification=? WHERE experience_id=?",
                (_utc_now(), json.dumps(safe_evidence, ensure_ascii=False, sort_keys=True), experience_id),
            )
        result = self.get(experience_id, include_recipe=False)
        return result

    def search(self, *, graph_signature: Any = (), unit_signature: Any = (),
               objective_signature: Any = (), limit: int = 20,
               include_provisional: bool = True) -> list[dict[str, Any]]:
        """Find structurally similar seeds; never returns an authorisation to reuse a result."""
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise ExperienceStoreError("limit 必须在 1 到 100 之间")
        query = {
            "graph": self._safe_signature(graph_signature, "graph_signature"),
            "units": self._safe_signature(unit_signature, "unit_signature"),
            "objective": self._safe_signature(objective_signature, "objective_signature"),
        }
        statuses = ("provisional", "verified") if include_provisional else ("verified",)
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM experiences WHERE status IN ({placeholders})", statuses
            ).fetchall()
        ranked: list[dict[str, Any]] = []
        for row in rows:
            record = self._row_summary(row)
            scores = {
                "graph": _jaccard(query["graph"], record["graph_signature"]),
                "units": _jaccard(query["units"], record["unit_signature"]),
                "objective": _jaccard(query["objective"], record["objective_signature"]),
            }
            score = 0.55 * scores["graph"] + 0.30 * scores["units"] + 0.15 * scores["objective"]
            if score <= 0:
                continue
            record["similarity"] = {"score": round(score, 12), "axes": scores}
            ranked.append(record)
        ranked.sort(key=lambda item: (-item["similarity"]["score"], item["experience_id"]))
        return ranked[:limit]

    def assess_revalidation(self, experience_id: str, current_scope: Mapping[str, Any]) -> dict[str, Any]:
        """Check whether a verified recipe still matches the current validation scope."""
        if not isinstance(current_scope, Mapping) or not current_scope:
            raise ExperienceStoreError("current_scope_must_be_nonempty_mapping")
        record = self.get(experience_id, include_recipe=False)
        verification = record.get("verification")
        if not isinstance(verification, Mapping) or not isinstance(verification.get("validation_scope"), Mapping):
            return {"status": "revalidate_required", "reason": "verification_scope_missing", "experience_id": experience_id}
        stored = verification["validation_scope"]
        mismatches = sorted(str(key) for key in set(stored) | set(current_scope)
                            if stored.get(key) != current_scope.get(key))
        return {"status": "current" if not mismatches else "stale", "experience_id": experience_id,
                "mismatched_scope_keys": mismatches,
                "policy": "stale_verified_recipes_remain_seed_only_and_must_be_revalidated"}

    def purge(self, *, run_id: str | None = None, topic_key: str | None = None,
              before: str | None = None, statuses: Sequence[str] = tuple(_PURGEABLE_STATUSES),
              dry_run: bool = True) -> dict[str, Any]:
        """Remove index entries by run/topic/date; verified entries are protected by default."""
        safe_run = self._validate_identifier(run_id, "run_id")
        safe_topic = self._validate_identifier(topic_key, "topic_key")
        requested = tuple(dict.fromkeys(str(item) for item in statuses))
        if not requested or any(item not in _STATUSES for item in requested):
            raise ExperienceStoreError("statuses 包含未知状态")
        if any(item == "verified" for item in requested) and not (run_id or topic_key or before):
            raise ExperienceStoreError("删除 verified 必须提供明确范围")
        clauses, params = [f"status IN ({','.join('?' for _ in requested)})"], list(requested)
        if safe_run is not None:
            clauses.append("run_id=?"); params.append(safe_run)
        if safe_topic is not None:
            clauses.append("topic_key=?"); params.append(safe_topic)
        if before is not None:
            try:
                datetime.fromisoformat(str(before).replace("Z", "+00:00"))
            except ValueError as exc:
                raise ExperienceStoreError("before 必须是 ISO-8601 时间") from exc
            clauses.append("created_at<?"); params.append(str(before))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT experience_id,object_hash FROM experiences WHERE " + " AND ".join(clauses), params
            ).fetchall()
        ids = [row["experience_id"] for row in rows]
        if not dry_run and ids:
            with self._connect() as connection:
                connection.executemany("DELETE FROM experiences WHERE experience_id=?", [(item,) for item in ids])
        deleted_objects = 0
        reclaimed_bytes = 0
        if not dry_run:
            for row in rows:
                with self._connect() as connection:
                    still_used = connection.execute(
                        "SELECT 1 FROM experiences WHERE object_hash=? LIMIT 1", (row["object_hash"],)
                    ).fetchone()
                if still_used:
                    continue
                path = self.objects / row["object_hash"][:2] / f"{row['object_hash']}.json"
                if path.is_file():
                    reclaimed_bytes += path.stat().st_size
                    path.unlink()
                    deleted_objects += 1
        return {
            "dry_run": bool(dry_run), "matched": len(ids), "deleted_records": 0 if dry_run else len(ids),
            "deleted_objects": deleted_objects, "reclaimed_bytes": reclaimed_bytes,
            "protected_verified_by_default": "verified" not in requested,
            "experience_ids": ids,
        }


__all__ = [
    "EXPERIENCE_SCHEMA_VERSION", "STORE_SCHEMA_VERSION", "ExperienceStoreError",
    "ExperienceStore", "redact_payload",
]
