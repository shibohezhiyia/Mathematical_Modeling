"""Route external research methods into bounded, auditable candidate plans.

The router deliberately plans methods instead of importing or executing third
party repositories.  It turns lightweight evidence about a task into a small
set of candidate arms that can later be compiled by the project's own
contracts.  This keeps external literature useful without making a paper,
package, or model checkpoint an implicit capability boundary.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class ExternalMethodRoutingError(ValueError):
    """Raised when a method-routing profile is ambiguous or unbounded."""


_FLAGS = (
    "numeric_data",
    "time_series",
    "spatial_grid",
    "known_dynamics",
    "noise_expected",
    "interaction_graph",
)
_METHODS = {
    "sindy": {
        "source_ids": ["pysindy-official-docs"],
        "prerequisites": ["numeric_data", "time_series"],
        "mode": "local_contract",
        "rationale": "稀疏识别时间序列动力学候选",
    },
    "weak_sindy": {
        "source_ids": ["pysindy-weak-pde-example"],
        "prerequisites": ["numeric_data", "time_series", "noise_expected"],
        "mode": "local_contract",
        "rationale": "对含噪时间序列优先尝试弱形式积分候选",
    },
    "pde_find": {
        "source_ids": ["pysindy-weak-pde-example"],
        "prerequisites": ["numeric_data", "spatial_grid"],
        "mode": "local_contract",
        "rationale": "在空间网格上构造受限偏微分方程特征库",
    },
    "ude": {
        "source_ids": ["sciml-ude-docs", "ude-original-paper"],
        "prerequisites": ["numeric_data", "time_series", "known_dynamics"],
        "mode": "optional_backend",
        "rationale": "在已知机理旁加入受验证的学习修正项",
    },
    "ude_neural": {
        "source_ids": ["sciml-ude-docs", "ude-original-paper"],
        "prerequisites": ["numeric_data", "time_series", "known_dynamics"],
        "mode": "optional_backend",
        "rationale": "在已知机理旁加入有界神经残差修正，并用时间留出复核",
    },
    "ude_joint": {
        "source_ids": ["sciml-ude-docs", "ude-original-paper"],
        "prerequisites": ["numeric_data", "time_series", "known_dynamics"],
        "mode": "optional_backend",
        "rationale": "联合拟合线性机理参数与轨迹级神经残差，保留独立时间留出",
    },
    "llm_sr": {
        "source_ids": ["llm-sr-official-code", "llm-srbench-paper"],
        "prerequisites": ["numeric_data"],
        "mode": "proposal_only",
        "execution_mode": "typed_tree_optional",
        "rationale": "外部模型只能提出类型化表达式树；本地拟合、留出和反例门负责执行与筛选",
    },
}


def _bool_flag(profile: Mapping[str, Any], name: str) -> bool:
    value = profile.get(name, False)
    if type(value) is not bool:
        raise ExternalMethodRoutingError(f"{name}_must_be_bool")
    return value


def plan_external_methods(
    profile: Mapping[str, Any],
    *,
    enabled_methods: Sequence[str] | None = None,
    installed_backends: Mapping[str, bool] | None = None,
    max_methods: int = 8,
) -> dict[str, Any]:
    """Build a deterministic method plan from task evidence.

    ``installed_backends`` is optional.  If omitted, dependency readiness is
    ``not_assessed`` rather than silently assuming that a third-party package
    exists.  The returned methods are proposals only; execution still goes
    through the project's candidate gate and resource sandbox.
    """
    if not isinstance(profile, Mapping):
        raise ExternalMethodRoutingError("profile_must_be_mapping")
    unknown = set(profile) - set(_FLAGS) - {"allow_external_repositories"}
    if unknown:
        raise ExternalMethodRoutingError("profile_contains_unknown_flags")
    flags = {name: _bool_flag(profile, name) for name in _FLAGS}
    allow_repositories = profile.get("allow_external_repositories", False)
    if type(allow_repositories) is not bool:
        raise ExternalMethodRoutingError("allow_external_repositories_must_be_bool")
    if type(max_methods) is not int or not 1 <= max_methods <= 32:
        raise ExternalMethodRoutingError("max_methods_invalid")
    if enabled_methods is None:
        requested = list(_METHODS)
    else:
        if isinstance(enabled_methods, (str, bytes)) or not isinstance(enabled_methods, Sequence):
            raise ExternalMethodRoutingError("enabled_methods_invalid")
        requested = list(enabled_methods)
        if any(not isinstance(item, str) for item in requested):
            raise ExternalMethodRoutingError("enabled_methods_unknown_or_duplicate")
        if len(set(requested)) != len(requested) or any(item not in _METHODS for item in requested):
            raise ExternalMethodRoutingError("enabled_methods_unknown_or_duplicate")
    if installed_backends is not None:
        if not isinstance(installed_backends, Mapping):
            raise ExternalMethodRoutingError("installed_backends_invalid")
        if any(type(value) is not bool for value in installed_backends.values()):
            raise ExternalMethodRoutingError("installed_backend_status_must_be_bool")
    candidates = []
    for method_name in requested:
        definition = _METHODS[method_name]
        missing = [flag for flag in definition["prerequisites"] if not flags[flag]]
        if missing:
            continue
        backend_key = method_name if method_name != "weak_sindy" else "pysindy"
        if installed_backends is None:
            dependency_status = "not_assessed"
        else:
            dependency_status = "ready" if installed_backends.get(backend_key, False) else "unavailable"
        if method_name == "llm_sr" and not allow_repositories:
            execution_status = "proposal_only"
        elif dependency_status == "unavailable":
            execution_status = "unavailable"
        else:
            execution_status = dependency_status
        candidates.append({
            "method": method_name,
            "status": execution_status,
            "dependency_status": dependency_status,
            "mode": definition["mode"],
            "execution_mode": definition.get("execution_mode", definition["mode"]),
            "source_ids": list(definition["source_ids"]),
            "prerequisites": list(definition["prerequisites"]),
            "rationale": definition["rationale"],
            "policy": "method_plan_is_not_execution_or_correctness_evidence",
        })
        if len(candidates) >= max_methods:
            break
    return {
        "schema_version": "mathmodel.external-method-plan/v1",
        "status": "planned" if candidates else "no_eligible_method",
        "profile": {**flags, "allow_external_repositories": allow_repositories},
        "candidates": candidates,
        "candidate_count": len(candidates),
        "policy": "external_sources_are_method_references; execution_requires_local_contracts_and_candidate_gate",
    }


__all__ = ["ExternalMethodRoutingError", "plan_external_methods"]
