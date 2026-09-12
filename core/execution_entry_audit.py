"""Static audit of numerical execution entry points.

This does not prove sandbox safety.  It catches accidental direct ``eval``/
``exec``/subprocess use in web or plugin paths and requires such calls to be
listed explicitly for review.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "mathmodel.execution-entry-audit/v1"
_EXCLUDED = {".git", "__pycache__", ".pytest_cache", "workspace", "data", "third_party", "venv", ".venv"}


def audit_execution_entries(root: str | Path, *, max_files: int = 10_000) -> dict[str, Any]:
    base = Path(root).resolve()
    if not base.is_dir() or base.anchor == base:
        raise ValueError("audit_root_invalid")
    files = [p for p in base.rglob("*.py") if not any(part in _EXCLUDED for part in p.parts)]
    if len(files) > max_files:
        raise ValueError("audit_file_budget_exceeded")
    findings = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    f"{node.func.value.id}.{node.func.attr}" if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) else None)
                if name in {"eval", "exec", "compile", "subprocess.run", "subprocess.Popen", "os.system"}:
                    findings.append({"file": str(path.relative_to(base)), "line": int(node.lineno),
                                     "operation": name, "review": "requires_guarded_entrypoint"})
    unguarded = [item for item in findings if item["operation"] in {"eval", "exec", "os.system"}]
    return {"schema_version": SCHEMA_VERSION, "status": "audited", "file_count": len(files),
            "finding_count": len(findings), "findings": findings,
            "unguarded_count": len(unguarded),
            "policy": "static_inventory_not_OS_sandbox_proof"}


__all__ = ["SCHEMA_VERSION", "audit_execution_entries"]
