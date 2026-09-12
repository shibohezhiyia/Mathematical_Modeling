"""Heuristic guard against leaking one competition/domain into generic IR."""
from __future__ import annotations

import re
from typing import Any, Mapping


class IRGenericityError(ValueError):
    pass


_FORBIDDEN = (
    ("competition_term", re.compile(r"赛题|高教社杯|数学建模大赛", re.I)),
    ("drone_identifier", re.compile(r"\bFY\s*\d+\b|无人机", re.I)),
    ("missile_identifier", re.compile(r"导弹|来袭导弹", re.I)),
    ("product_domain_term", re.compile(r"商品名称|单品编码", re.I)),
)


def audit_ir_genericity(ir: Mapping[str, Any], *, max_findings: int = 64) -> dict[str, Any]:
    if not isinstance(ir, Mapping) or type(max_findings) is not int or not 1 <= max_findings <= 512:
        raise IRGenericityError("invalid_ir_genericity_input")
    findings = []

    def visit(value: Any, path: str) -> None:
        if len(findings) >= max_findings:
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, str):
            matched = False
            for code, pattern in _FORBIDDEN:
                match = pattern.search(value)
                if match:
                    findings.append({"code": code, "path": path, "token": match.group(0)[:80]})
                    matched = True
                    break
            # M1/M2 are also ordinary matrix labels; only flag them when the
            # same field explicitly carries missile semantics.
            if not matched and ("导弹" in value or "来袭" in value):
                match = re.search(r"\bM\s*\d+\b", value, re.I)
                if match:
                    findings.append({"code": "missile_identifier", "path": path, "token": match.group(0)[:80]})

    visit(ir, "")
    return {"schema_version": "mathmodel.ir-genericity/v1", "status": "pass" if not findings else "blocked",
            "findings": findings, "finding_count": len(findings),
            "policy": "heuristic_guard_not_a_complete ontology or domain-leakage proof"}


__all__ = ["IRGenericityError", "audit_ir_genericity"]
