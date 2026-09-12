"""Small hidden-equation benchmark for symbolic dynamics discovery.

The generated trajectories are synthetic and deliberately kept separate from
the discovery call. The report distinguishes support recovery from numerical
fit; passing this fixture is not evidence for real-world equation discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class SymbolicBenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class HiddenEquationCase:
    case_id: str
    times: tuple[float, ...]
    states: tuple[tuple[float, ...], ...]
    state_names: tuple[str, ...]
    expected_support: tuple[tuple[str, ...], ...]

    def public(self) -> dict[str, Any]:
        return {"id": self.case_id, "sample_count": len(self.times),
                "state_count": len(self.state_names), "state_names": list(self.state_names),
                "expected_support_hidden": True}


def build_unseen_equation_cases(*, sample_count: int = 96) -> tuple[HiddenEquationCase, ...]:
    if type(sample_count) is not int or not 32 <= sample_count <= 2_000:
        raise SymbolicBenchmarkError("invalid_sample_count")
    t = np.linspace(0.0, 4.0, sample_count)
    cases = (
        HiddenEquationCase("synthetic-linear-polynomial", tuple(t),
                           tuple(map(tuple, np.column_stack((t, t * t)))), ("x", "y"),
                           (("1",), ("x",))),
        HiddenEquationCase("synthetic-exponential-linear", tuple(t),
                           tuple(map(tuple, np.column_stack((t, np.exp(t))))), ("x", "y"),
                           (("1",), ("y",))),
        HiddenEquationCase("synthetic-quadratic-decay", tuple(t),
                           tuple(map(tuple, np.column_stack((1.0 / (1.0 + t), np.exp(t))))), ("x", "y"),
                           (("x^2",), ("y",))),
    )
    return cases


def evaluate_unseen_equation_benchmark(
    discover: Callable[..., Mapping[str, Any]], *, cases: Sequence[HiddenEquationCase] | None = None,
    max_cases: int = 16,
    fit_only_discover: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not callable(discover):
        raise SymbolicBenchmarkError("discover_callable_required")
    selected = tuple(cases if cases is not None else build_unseen_equation_cases())
    if not 1 <= len(selected) <= max_cases or type(max_cases) is not int or not 1 <= max_cases <= 64:
        raise SymbolicBenchmarkError("case_budget_exceeded")
    rows = []

    def score_case(case: HiddenEquationCase, runner: Callable[..., Mapping[str, Any]], *, mode: str) -> dict[str, Any]:
        """Score one runner while keeping hidden supports out of the report."""
        if not isinstance(case, HiddenEquationCase):
            raise SymbolicBenchmarkError("invalid_case")
        try:
            result = runner(case.times, case.states, state_names=case.state_names,
                              polynomial_degree=2, sparsity_threshold=5e-2,
                              residual_tolerance=1e-2,
                              **({"validation_fraction": 0.05} if mode == "fit_only" else {}))
        except Exception as exc:
            return {"id": case.case_id, "status": "failed", "error_code": type(exc).__name__}
        equations = result.get("equations") if isinstance(result, Mapping) else None
        if not isinstance(equations, list) or len(equations) != len(case.expected_support):
            return {"id": case.case_id, "status": "invalid_result", "support_correct": False}
        support_correct = True
        support_rows = []
        for equation, expected in zip(equations, case.expected_support):
            terms = equation.get("terms", []) if isinstance(equation, Mapping) else []
            actual = tuple(sorted(str(item.get("term")) for item in terms if isinstance(item, Mapping)
                                  and abs(float(item.get("coefficient", 0.0))) > 1e-3))
            expected_set = tuple(sorted(expected))
            support_rows.append({"expected_hidden": True, "term_count": len(actual)})
            support_correct &= actual == expected_set
        residuals = result.get("validation_derivative_rmse", [])
        finite_residual = isinstance(residuals, list) and all(
            type(item) in (int, float) and math.isfinite(float(item)) for item in residuals)
        return {"id": case.case_id, "status": "passed" if support_correct and finite_residual else "failed",
                     "support_correct": support_correct, "finite_validation_residual": finite_residual,
                     "support": support_rows, "discovery_status": result.get("status"), "mode": mode}

    for case in selected:
        rows.append(score_case(case, discover, mode="validated"))
    passed = sum(row.get("status") == "passed" for row in rows)
    report = {"schema_version": "mathmodel.symbolic-discovery-benchmark/v1", "case_count": len(rows),
              "passed_count": passed, "support_recovery_rate": passed / len(rows), "rows": rows,
              "policy": "synthetic_unseen_equations_only;support_recovery_is_not_real_world_accuracy"}
    if fit_only_discover is not None:
        if not callable(fit_only_discover):
            raise SymbolicBenchmarkError("fit_only_discover_must_be_callable")
        baseline_rows = [score_case(case, fit_only_discover, mode="fit_only") for case in selected]
        baseline_passed = sum(row.get("status") == "passed" for row in baseline_rows)
        report["fit_only_baseline"] = {
            "case_count": len(baseline_rows),
            "passed_count": baseline_passed,
            "support_recovery_rate": baseline_passed / len(baseline_rows),
            "rows": baseline_rows,
            "policy": "fit_only_is_a_paired_validation_split_ablation_not_an_independent_algorithm",
        }
    return report


__all__ = ["SymbolicBenchmarkError", "HiddenEquationCase", "build_unseen_equation_cases",
           "evaluate_unseen_equation_benchmark"]
