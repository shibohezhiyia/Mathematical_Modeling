"""封存的真实未见题基准协议。

这个模块解决的是 ``OpenModelBench`` 没有覆盖的另一半问题：题面和参考解
必须在系统选型、提示开发和代码搜索期间不可见。公开仓库只保存哈希、预算和
运行记录；只有在运行完成后，持有仓外解封令牌的人才能提交参考解并计算分数。

它不是“生成几道随机题”的工具，也不声称合成夹具等于真实未见题。真实基准
仍需由项目外的题面、附件和参考解提供，并在首次公开运行前冻结。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import secrets
from typing import Any, Iterable, Mapping

from .data_provenance import DataProvenanceError, content_shingle_digests


BLIND_SCHEMA = "mathmodel.blind-benchmark/v1"
BLIND_CASE_SCHEMA = "mathmodel.blind-case/v1"
RUN_SCHEMA = "mathmodel.blind-run/v1"
SCORE_SCHEMA = "mathmodel.blind-score/v1"
CASE_SPLITS = frozenset({"unseen", "adversarial", "structure_transform"})
CASE_PROVENANCE = frozenset({"external_real", "synthetic_fixture"})
RUN_STATUSES = frozenset({"completed", "failed", "blocked", "timed_out", "not_run"})
SCORE_FIELDS = frozenset({
    "contract_correct", "numerically_correct", "constraint_validity",
    "evidence_completeness", "stability", "human_judgement",
})
_TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".md",
    ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".tsv", ".xml", ".sql", ".ipynb",
})
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", "dist", "build", "coverage", ".mypy_cache",
})
_FORBIDDEN_PAYLOAD_KEYS = frozenset({
    "statement", "question", "problem_text", "answer", "reference_answer",
    "ground_truth", "raw_data", "attachment_content", "solution_text",
})


class BlindBenchmarkError(ValueError):
    """封存基准协议错误。``code`` 是稳定的机器可读原因。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise BlindBenchmarkError(code)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BlindBenchmarkError("non_json_payload") from exc


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> tuple[str, int]:
    if not isinstance(path, Path):
        path = Path(path)
    try:
        if not path.is_file():
            _fail("sealed_file_missing")
        digest = sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
        return digest.hexdigest(), size
    except OSError as exc:
        raise BlindBenchmarkError("sealed_file_unreadable") from exc


def _sha(value: Any, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        _fail("invalid_digest")
    return value


def _case_id(value: Any) -> str:
    if type(value) is not str or not value or len(value) > 128 or any(ch.isspace() for ch in value):
        _fail("invalid_case_id")
    return value


def _safe_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if type(value) is not list or len(value) > 32:
        _fail("invalid_tags")
    if any(type(item) is not str or not item or len(item) > 80 or any(ch.isspace() for ch in item) for item in value):
        _fail("invalid_tags")
    return tuple(dict.fromkeys(value))


def _safe_budget(value: Mapping[str, Any]) -> dict[str, int]:
    if type(value) is not dict:
        _fail("invalid_budget")
    allowed = {"max_seconds", "max_memory_mb", "max_api_calls", "max_candidates", "seed"}
    if set(value) != allowed:
        _fail("invalid_budget_fields")
    limits = {}
    ranges = {
        "max_seconds": (1, 86_400), "max_memory_mb": (32, 1_048_576),
        "max_api_calls": (0, 100_000), "max_candidates": (1, 100_000),
        "seed": (0, 2**32 - 1),
    }
    for key, (lower, upper) in ranges.items():
        item = value[key]
        if type(item) is not int or not lower <= item <= upper:
            _fail("invalid_budget")
        limits[key] = item
    return limits


@dataclass(frozen=True)
class BlindCase:
    """不含题面和答案的公开案例承诺。"""

    case_id: str
    split: str
    family: str
    provenance: str
    statement_sha256: str
    statement_bytes: int
    attachment_sha256: tuple[str, ...] = ()
    answer_sha256: str | None = None
    answer_bytes: int | None = None
    unlock_commitment: str = ""
    tags: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BlindCase":
        if type(payload) is not dict:
            _fail("case_must_be_object")
        allowed = {
            "schema_version", "id", "split", "family", "provenance",
            "statement_sha256", "statement_bytes", "attachment_sha256",
            "answer_sha256", "answer_bytes", "unlock_commitment", "tags",
        }
        if set(payload) - allowed or payload.get("schema_version") != BLIND_CASE_SCHEMA:
            _fail("invalid_case_fields")
        case_id = _case_id(payload.get("id"))
        split, provenance = payload.get("split"), payload.get("provenance")
        if split not in CASE_SPLITS:
            _fail("invalid_blind_split")
        if provenance not in CASE_PROVENANCE:
            _fail("invalid_case_provenance")
        family = payload.get("family")
        if type(family) is not str or not family or len(family) > 80 or any(ch.isspace() for ch in family):
            _fail("invalid_case_family")
        statement_sha = _sha(payload.get("statement_sha256"))
        statement_bytes = payload.get("statement_bytes")
        if type(statement_bytes) is not int or not 1 <= statement_bytes <= 100_000_000:
            _fail("invalid_statement_size")
        attachments = payload.get("attachment_sha256", [])
        if type(attachments) is not list or len(attachments) > 128:
            _fail("invalid_attachment_digests")
        attachment_sha = tuple(_sha(item) for item in attachments)
        answer_sha = _sha(payload.get("answer_sha256"), optional=True)
        answer_bytes = payload.get("answer_bytes")
        if answer_sha is None:
            if answer_bytes is not None:
                _fail("answer_size_without_answer")
        elif type(answer_bytes) is not int or not 1 <= answer_bytes <= 500_000_000:
            _fail("invalid_answer_size")
        commitment = payload.get("unlock_commitment")
        if type(commitment) is not str or len(commitment) != 64 or any(ch not in "0123456789abcdef" for ch in commitment):
            _fail("invalid_unlock_commitment")
        return cls(case_id, split, family, provenance, statement_sha, statement_bytes,
                   attachment_sha, answer_sha, answer_bytes, commitment, _safe_tags(payload.get("tags")))

    def public(self) -> dict[str, Any]:
        return {
            "schema_version": BLIND_CASE_SCHEMA,
            "id": self.case_id,
            "split": self.split,
            "family": self.family,
            "provenance": self.provenance,
            "statement_sha256": self.statement_sha256,
            "statement_bytes": self.statement_bytes,
            "attachment_sha256": list(self.attachment_sha256),
            "answer_sha256": self.answer_sha256,
            "answer_bytes": self.answer_bytes,
            "unlock_commitment": self.unlock_commitment,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class BlindManifest:
    """冻结后的公开封存清单。"""

    _json: str

    @classmethod
    def create(cls, cases: Iterable[BlindCase | Mapping[str, Any]], *, budget: Mapping[str, Any],
               name: str = "real-unseen-benchmark", revision: int = 1) -> "BlindManifest":
        if type(name) is not str or not name.strip() or len(name) > 120:
            _fail("invalid_benchmark_name")
        if type(revision) is not int or revision < 1:
            _fail("invalid_benchmark_revision")
        parsed = [case if isinstance(case, BlindCase) else BlindCase.from_payload(case) for case in cases]
        if not parsed or len(parsed) > 10_000:
            _fail("invalid_case_count")
        ids = [case.case_id for case in parsed]
        if len(ids) != len(set(ids)):
            _fail("duplicate_case_id")
        payload = {
            "schema_version": BLIND_SCHEMA,
            "name": name.strip(),
            "revision": revision,
            "budget": _safe_budget(dict(budget)),
            "cases": [case.public() for case in sorted(parsed, key=lambda item: item.case_id)],
            "policy": {
                "statement_and_answer_not_in_public_manifest": True,
                "score_requires_unlock_token": True,
                "selection_before_unseen_unlock": True,
            },
        }
        return cls(_canonical(payload))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BlindManifest":
        if type(payload) is not dict or set(payload) != {"schema_version", "name", "revision", "budget", "cases", "policy"}:
            _fail("invalid_manifest_fields")
        if payload.get("schema_version") != BLIND_SCHEMA:
            _fail("manifest_schema_mismatch")
        manifest = cls.create(payload["cases"], budget=payload["budget"],
                              name=payload["name"], revision=payload["revision"])
        if payload["policy"] != json.loads(manifest._json)["policy"]:
            _fail("invalid_manifest_policy")
        return manifest

    @property
    def digest(self) -> str:
        return sha256(self._json.encode("utf-8")).hexdigest()

    def public(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def budget(self) -> dict[str, int]:
        return dict(self.public()["budget"])

    def cases(self, *, split: str | None = None) -> tuple[BlindCase, ...]:
        if split is not None and split not in CASE_SPLITS:
            _fail("invalid_blind_split")
        return tuple(BlindCase.from_payload(item) for item in self.public()["cases"]
                     if split is None or item["split"] == split)

    def case(self, case_id: str) -> BlindCase:
        case_id = _case_id(case_id)
        for item in self.cases():
            if item.case_id == case_id:
                return item
        _fail("unknown_case_id")


def build_sealed_case(*, case_id: str, family: str, statement_path: str | Path,
                      attachments: Iterable[str | Path] = (), answer_path: str | Path | None = None,
                      split: str = "unseen", provenance: str = "external_real",
                      tags: Iterable[str] = (), unlock_token: str | None = None) -> tuple[BlindCase, str]:
    """从仓外文件生成一个案例承诺；返回案例和一次性解封令牌。

    函数只返回哈希和令牌，不把题面、附件或答案内容复制到清单。
    """
    statement_sha, statement_bytes = _file_digest(Path(statement_path))
    attachment_results = [_file_digest(Path(path)) for path in attachments]
    attachment_sha = tuple(item[0] for item in attachment_results)
    answer_sha = answer_bytes = None
    if answer_path is not None:
        answer_sha, answer_bytes = _file_digest(Path(answer_path))
    token = unlock_token or secrets.token_urlsafe(32)
    if type(token) is not str or not 20 <= len(token) <= 256:
        _fail("invalid_unlock_token")
    commitment = sha256(token.encode("utf-8")).hexdigest()
    payload = {
        "schema_version": BLIND_CASE_SCHEMA, "id": case_id, "split": split,
        "family": family, "provenance": provenance,
        "statement_sha256": statement_sha, "statement_bytes": statement_bytes,
        "attachment_sha256": list(attachment_sha), "answer_sha256": answer_sha,
        "answer_bytes": answer_bytes, "unlock_commitment": commitment,
        "tags": list(tags),
    }
    return BlindCase.from_payload(payload), token


def scan_leakage(project_root: str | Path, manifest: BlindManifest | Mapping[str, Any], *,
                 exclude_paths: Iterable[str | Path] = (), protected_files: Iterable[str | Path] = ()) -> dict[str, Any]:
    """扫描公开项目是否包含封存题面的明显泄漏。

    扫描只返回相对路径、案例编号和稳定原因，不返回匹配文本。除哈希/案例编号
    外，若提供 ``protected_files``，还会以规范化后的整文件摘要检查题面或答案
    是否被原样复制进仓库。
    """
    if not isinstance(manifest, BlindManifest):
        manifest = BlindManifest.from_payload(manifest)
    root = Path(project_root).resolve()
    if not root.is_dir():
        _fail("project_root_missing")
    excludes = {Path(path).resolve() for path in exclude_paths}
    secret_digests: dict[str, set[str]] = {}
    protected_shingles: dict[str, set[str]] = {}
    for item in protected_files:
        path = Path(item)
        digest, _ = _file_digest(path)
        secret_digests.setdefault(digest, set()).add(path.name)
        try:
            if path.suffix.lower() in _TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8")
                protected_shingles[path.name] = set(content_shingle_digests(text, shingle_size=8))
        except (OSError, UnicodeError, DataProvenanceError):
            # Exact digests remain available even when a protected file cannot
            # be normalized for the partial-overlap screen.
            continue
    by_digest: dict[str, tuple[str, str]] = {}
    case_ids: dict[str, str] = {}
    for case in manifest.cases():
        by_digest[case.statement_sha256] = (case.case_id, "statement_digest")
        if case.answer_sha256:
            by_digest[case.answer_sha256] = (case.case_id, "answer_digest")
        for digest in case.attachment_sha256:
            by_digest[digest] = (case.case_id, "attachment_digest")
        case_ids[case.case_id] = case.case_id
    findings: list[dict[str, Any]] = []
    scanned = 0
    for path in root.rglob("*"):
        resolved = path.resolve()
        if (not path.is_file() or resolved in excludes
                or any(ex == resolved or ex in resolved.parents for ex in excludes)
                or any(part in _SKIP_DIRS for part in path.parts)):
            continue
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        scanned += 1
        digest = sha256(raw).hexdigest()
        rel = str(path.resolve().relative_to(root)).replace("\\", "/")
        if digest in by_digest:
            case_id, reason = by_digest[digest]
            findings.append({"case_id": case_id, "reason": reason, "path": rel})
        if digest in secret_digests:
            for label in sorted(secret_digests[digest]):
                findings.append({"case_id": "protected-file", "reason": "protected_file_digest", "path": rel, "source": label})
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if protected_shingles:
            try:
                candidate_shingles = set(content_shingle_digests(text, shingle_size=8))
            except DataProvenanceError:
                candidate_shingles = set()
            for source_name, source_shingles in protected_shingles.items():
                if candidate_shingles & source_shingles:
                    findings.append({"case_id": "protected-file", "reason": "protected_file_shingle_overlap",
                                     "path": rel, "source": source_name})
        for case_id in case_ids:
            if case_id in text:
                findings.append({"case_id": case_id, "reason": "case_id_literal", "path": rel})
    # 去重保证结果稳定，不暴露匹配内容。
    unique = sorted({(item["case_id"], item["reason"], item["path"], item.get("source")) for item in findings})
    normalized = [{"case_id": a, "reason": b, "path": c, **({"source": d} if d else {})} for a, b, c, d in unique]
    return {
        "schema_version": "mathmodel.blind-leakage/v1",
        "manifest_digest": manifest.digest,
        "status": "fail" if normalized else "pass",
        "scanned_files": scanned,
        "finding_count": len(normalized),
        "findings": normalized,
        "policy": {"content_is_not_returned": True, "case_ids_are_sensitive_markers": True,
                    "partial_overlap_screen": "normalized_word_shingles;_not_semantic_similarity"},
    }


def record_blind_run(manifest: BlindManifest | Mapping[str, Any], *, case_id: str, run_id: str,
                     status: str, budget: Mapping[str, Any], system_version: str,
                     artifact_digest: str | None = None, elapsed_seconds: float | None = None,
                     api_calls: int = 0, candidate_count: int = 0, failure_code: str | None = None,
                     output_digest: str | None = None) -> dict[str, Any]:
    """记录一次固定协议运行；不允许把题面、答案或任意文本结果写入运行账本。"""
    if not isinstance(manifest, BlindManifest):
        manifest = BlindManifest.from_payload(manifest)
    case = manifest.case(case_id)
    if type(run_id) is not str or not run_id or len(run_id) > 160 or any(ch.isspace() for ch in run_id):
        _fail("invalid_run_id")
    if status not in RUN_STATUSES:
        _fail("invalid_run_status")
    if type(system_version) is not str or not system_version.strip() or len(system_version) > 160:
        _fail("invalid_system_version")
    requested = _safe_budget(dict(budget))
    if requested != manifest.budget:
        _fail("fixed_budget_mismatch")
    for value, limit, code in ((api_calls, requested["max_api_calls"], "invalid_api_calls"),
                               (candidate_count, requested["max_candidates"], "invalid_candidate_count")):
        if type(value) is not int or not 0 <= value <= limit:
            _fail(code)
    if elapsed_seconds is not None and (type(elapsed_seconds) not in (int, float) or elapsed_seconds < 0 or elapsed_seconds > requested["max_seconds"]):
        _fail("invalid_elapsed_seconds")
    if status in {"failed", "blocked", "timed_out"} and (type(failure_code) is not str or not failure_code or len(failure_code) > 120):
        _fail("failed_run_requires_code")
    if status == "completed" and failure_code is not None:
        _fail("failure_code_on_completed_run")
    for digest in (artifact_digest, output_digest):
        if digest is not None:
            _sha(digest)
    row = {
        "schema_version": RUN_SCHEMA, "manifest_digest": manifest.digest,
        "case_id": case.case_id, "split": case.split, "run_id": run_id,
        "system_version": system_version.strip(), "status": status,
        "budget": requested, "api_calls": api_calls, "candidate_count": candidate_count,
        "elapsed_seconds": elapsed_seconds, "artifact_digest": artifact_digest,
        "output_digest": output_digest, "failure_code": failure_code,
    }
    # 防止未来新增字段误把题面带入协议。
    if set(row) & _FORBIDDEN_PAYLOAD_KEYS:
        _fail("sensitive_run_field")
    return row


def unlock_and_score(manifest: BlindManifest | Mapping[str, Any], run: Mapping[str, Any], *,
                     reference: Mapping[str, Any], unlock_token: str) -> dict[str, Any]:
    """用仓外令牌解封一个完成的运行并核验参考解承诺。"""
    if not isinstance(manifest, BlindManifest):
        manifest = BlindManifest.from_payload(manifest)
    if type(run) is not dict or run.get("schema_version") != RUN_SCHEMA or run.get("manifest_digest") != manifest.digest:
        _fail("run_scope_mismatch")
    case = manifest.case(run.get("case_id"))
    if run.get("status") != "completed":
        _fail("only_completed_run_can_be_scored")
    if type(unlock_token) is not str or not unlock_token:
        _fail("invalid_unlock_token")
    if sha256(unlock_token.encode("utf-8")).hexdigest() != case.unlock_commitment:
        _fail("unlock_token_mismatch")
    if case.answer_sha256 is None:
        _fail("reference_answer_unavailable")
    if type(reference) is not dict or set(reference) - {"schema_version", "manifest_digest", "case_id", "answer_sha256", "scores", "evaluator"}:
        _fail("invalid_reference_fields")
    if reference.get("schema_version") != SCORE_SCHEMA or reference.get("manifest_digest") != manifest.digest or reference.get("case_id") != case.case_id:
        _fail("reference_scope_mismatch")
    if reference.get("answer_sha256") != case.answer_sha256:
        _fail("reference_answer_commitment_mismatch")
    scores = reference.get("scores")
    if type(scores) is not dict or not scores or set(scores) - SCORE_FIELDS:
        _fail("invalid_score_fields")
    clean_scores: dict[str, float | bool] = {}
    for key, value in scores.items():
        if type(value) is bool and key == "contract_correct":
            clean_scores[key] = value
        elif type(value) in (int, float) and not isinstance(value, bool) and 0 <= float(value) <= 1:
            clean_scores[key] = float(value)
        else:
            _fail("invalid_score_value")
    evaluator = reference.get("evaluator")
    if type(evaluator) is not str or not evaluator.strip() or len(evaluator) > 160:
        _fail("invalid_evaluator")
    return {
        "schema_version": SCORE_SCHEMA, "manifest_digest": manifest.digest,
        "case_id": case.case_id, "run_id": run["run_id"], "scores": clean_scores,
        "evaluator": evaluator.strip(), "unlocked": True,
    }


def build_blind_report(manifest: BlindManifest | Mapping[str, Any], runs: Iterable[Mapping[str, Any]],
                       scores: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """生成解封前也安全的报告；没有参考解时绝不输出正确率。"""
    if not isinstance(manifest, BlindManifest):
        manifest = BlindManifest.from_payload(manifest)
    cases = {case.case_id: case for case in manifest.cases()}
    run_rows = list(runs)
    score_rows = list(scores)
    latest: dict[str, Mapping[str, Any]] = {}
    for row in run_rows:
        if type(row) is not dict or row.get("manifest_digest") != manifest.digest or row.get("schema_version") != RUN_SCHEMA:
            _fail("run_scope_mismatch")
        case_id = row.get("case_id")
        if case_id not in cases:
            _fail("unknown_case_id")
        if row.get("status") not in RUN_STATUSES:
            _fail("invalid_run_status")
        if case_id in latest:
            _fail("duplicate_run_case")
        latest[case_id] = row
    valid_scores: dict[str, Mapping[str, Any]] = {}
    for row in score_rows:
        if type(row) is not dict or row.get("schema_version") != SCORE_SCHEMA or row.get("manifest_digest") != manifest.digest or not row.get("unlocked"):
            _fail("score_scope_mismatch")
        if row.get("case_id") not in cases or row["case_id"] in valid_scores:
            _fail("duplicate_or_unknown_score")
        if row["case_id"] not in latest or latest[row["case_id"]].get("run_id") != row.get("run_id"):
            _fail("score_without_matching_run")
        valid_scores[row["case_id"]] = row
    counts = {status: 0 for status in RUN_STATUSES}
    for case_id in cases:
        counts[latest.get(case_id, {"status": "not_run"})["status"]] += 1
    all_real = bool(cases) and all(case.provenance == "external_real" for case in cases.values())
    score_means: dict[str, float] = {}
    for row in valid_scores.values():
        for key, value in row["scores"].items():
            if type(value) in (int, float):
                score_means.setdefault(key, 0.0)
                score_means[key] += float(value)
    for key in list(score_means):
        score_means[key] /= len(valid_scores)
    return {
        "schema_version": "mathmodel.blind-report/v1", "manifest_digest": manifest.digest,
        "case_count": len(cases), "run_count": len(run_rows), "scored_count": len(valid_scores),
        "status_counts": counts, "score_means": score_means,
        "evaluation_status": "scored" if valid_scores else "not_scored",
        "real_unseen_declared": all_real,
        "policy": {
            "accuracy_reported": bool(valid_scores),
            "unscored_cases_not_treated_as_failures": True,
            "synthetic_fixture_is_not_real_evidence": not all_real,
        },
    }


__all__ = [
    "BLIND_SCHEMA", "BLIND_CASE_SCHEMA", "RUN_SCHEMA", "SCORE_SCHEMA",
    "CASE_SPLITS", "CASE_PROVENANCE", "RUN_STATUSES", "BlindBenchmarkError",
    "BlindCase", "BlindManifest", "build_sealed_case", "scan_leakage",
    "record_blind_run", "unlock_and_score", "build_blind_report",
]
