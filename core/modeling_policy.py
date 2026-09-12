"""Policy gate preventing arbitrary axioms and unjustified model pruning."""

from __future__ import annotations

from typing import Any, Mapping


class ModelingPolicyError(ValueError):
    pass


def audit_modeling_choice(choice: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(choice, Mapping):
        raise ModelingPolicyError("choice_must_be_mapping")
    checks = {
        "主控定律不是强制指定": choice.get("law_forced") is not True,
        "变量删减不是固定比例": choice.get("fixed_drop_fraction") is None,
        "因果图未无证据锁定": not (choice.get("causal_dag_locked") is True and choice.get("causal_evidence_status") != "verified"),
        "黑箱基线保留": choice.get("black_box_baseline_disabled") is not True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "mathmodel.modeling-policy/v1",
        "status": "pass" if not failed else "fail",
        "checks": [{"name": name, "status": "pass" if passed else "fail"} for name, passed in checks.items()],
        "failed_checks": failed,
        "policy": "complexity_and_interpretability_require_independent_evidence_comparison; policy_gate_does_not_choose_a_model",
    }


__all__ = ["ModelingPolicyError", "audit_modeling_choice"]
