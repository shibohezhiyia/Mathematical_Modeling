"""Create and verify a content addressed evaluation freeze.

The freeze records the exact source/configuration surface used by an external
evaluation.  It deliberately excludes ``data/`` and ``workspace/`` so private
holdouts cannot leak into a public manifest.  A matching digest is a
reproducibility precondition, not proof that the model is correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping


SCHEMA = "mathmodel.evaluation-freeze/v1"
_ROOTS = ("core", "scripts", "web", "examples")
_FILES = ("requirements.txt", "requirements-symbolic-baseline.txt", "pytest.ini", ".env.example")
_SKIP = {"__pycache__", ".pytest_cache", ".git", "node_modules", "data", "workspace"}
_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "sympy", "torch")


class EvaluationFreezeError(ValueError):
    pass


def _digest_bytes(path: Path) -> tuple[str, int]:
    digest, size = sha256(), 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise EvaluationFreezeError("freeze_file_unreadable") from exc
    return digest.hexdigest(), size


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_text(value: Any, code: str, maximum: int = 160) -> str:
    if type(value) is not str or not value.strip() or len(value.strip()) > maximum:
        raise EvaluationFreezeError(code)
    return value.strip()


def _collect_files(root: Path) -> list[dict[str, Any]]:
    files: list[Path] = []
    for name in _ROOTS:
        folder = root / name
        if folder.is_dir():
            files.extend(path for path in folder.rglob("*") if path.is_file()
                         and not any(part in _SKIP for part in path.parts))
    files.extend(path for path in (root / name for name in _FILES) if path.is_file())
    # The committed external-validation snapshot records this freeze digest;
    # exclude it to avoid a self-referential hash cycle while keeping public
    # configuration files in the freeze surface.
    unique = sorted({path.resolve() for path in files
                     if not (path.parent.name == "examples" and (
                         path.name.startswith("external_validation_") or
                         path.name.startswith("system_comparison_")))})
    if len(unique) > 10_000:
        raise EvaluationFreezeError("freeze_file_budget_exceeded")
    rows = []
    total = 0
    for path in unique:
        digest, size = _digest_bytes(path)
        total += size
        if total > 200 * 1024 * 1024:
            raise EvaluationFreezeError("freeze_byte_budget_exceeded")
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": size, "sha256": digest})
    if not rows:
        raise EvaluationFreezeError("freeze_source_surface_empty")
    return rows


def _dependencies() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def create_evaluation_freeze(
    root: str | Path, *, protocol_id: str, seed: int, budget: Mapping[str, Any],
    methods: Iterable[str], prompt_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    _validate_text(protocol_id, "protocol_id_invalid")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise EvaluationFreezeError("seed_invalid")
    if not isinstance(budget, Mapping) or not budget:
        raise EvaluationFreezeError("budget_invalid")
    clean_budget = dict(budget)
    if any(type(key) is not str or type(value) is not int or value < 0 for key, value in clean_budget.items()):
        raise EvaluationFreezeError("budget_invalid")
    clean_methods = tuple(_validate_text(item, "method_invalid", 120) for item in methods)
    if not clean_methods or len(set(clean_methods)) != len(clean_methods):
        raise EvaluationFreezeError("methods_invalid")
    prompts = dict(prompt_digests or {})
    for name, digest in prompts.items():
        _validate_text(name, "prompt_name_invalid", 200)
        if type(digest) is not str or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise EvaluationFreezeError("prompt_digest_invalid")
    files = _collect_files(project_root)
    payload = {
        "schema_version": SCHEMA, "protocol_id": protocol_id.strip(), "seed": seed,
        "budget": clean_budget, "methods": list(clean_methods), "prompt_digests": prompts,
        "dependencies": _dependencies(), "files": files,
        "policy": {"no_holdout_content": True, "no_data_or_workspace_files": True,
                   "freeze_before_unseen_read": True},
    }
    payload["freeze_digest"] = sha256(_canonical(payload).encode("utf-8")).hexdigest()
    payload["status"] = "frozen"
    return payload


def verify_evaluation_freeze(root: str | Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(freeze, Mapping) or freeze.get("schema_version") != SCHEMA:
        raise EvaluationFreezeError("freeze_schema_invalid")
    expected = freeze.get("files")
    if not isinstance(expected, list) or not expected:
        raise EvaluationFreezeError("freeze_files_invalid")
    project_root = Path(root).resolve()
    mismatches = []
    for row in expected:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise EvaluationFreezeError("freeze_file_record_invalid")
        if (type(row["path"]) is not str or not row["path"] or
                type(row["bytes"]) is not int or row["bytes"] < 0 or
                type(row["sha256"]) is not str or len(row["sha256"]) != 64 or
                any(ch not in "0123456789abcdef" for ch in row["sha256"].lower())):
            raise EvaluationFreezeError("freeze_file_record_invalid")
        relative = Path(row["path"])
        if relative.is_absolute() or any(part in _SKIP for part in relative.parts):
            raise EvaluationFreezeError("freeze_path_not_allowed")
        path = (project_root / relative).resolve()
        if project_root not in path.parents:
            raise EvaluationFreezeError("freeze_path_escape")
        if not path.is_file():
            mismatches.append({"path": relative.as_posix(), "reason": "missing"})
            continue
        digest, size = _digest_bytes(path)
        if digest != row["sha256"] or size != row["bytes"]:
            mismatches.append({"path": relative.as_posix(), "reason": "changed"})
    return {"schema_version": "mathmodel.evaluation-freeze-verify/v1",
            "status": "verified" if not mismatches else "mismatch",
            "mismatch_count": len(mismatches), "mismatches": mismatches,
            "freeze_digest": freeze.get("freeze_digest"),
            "policy": "file_identity_check_only;does_not_prove_model_correctness"}


__all__ = ["SCHEMA", "EvaluationFreezeError", "create_evaluation_freeze", "verify_evaluation_freeze"]
