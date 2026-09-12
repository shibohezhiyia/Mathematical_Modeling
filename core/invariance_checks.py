"""Small, deterministic metamorphic checks for model pipelines."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping
import math


class InvarianceCheckError(ValueError):
    pass


def run_invariance_checks(
    baseline_input: Any,
    baseline_output: Any,
    evaluate: Callable[[Any], Any],
    transformations: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = 1e-8,
    max_cases: int = 64,
) -> dict[str, Any]:
    """Evaluate explicit metamorphic transformations without declaring proof.

    Each transformation contains ``id``, ``apply`` and ``compare`` callbacks.
    ``compare`` must return a boolean (for example exact entity permutation
    invariance or tolerance-based unit round-trip equivalence). The function
    records failures and execution errors separately.
    """
    if not callable(evaluate) or not isinstance(transformations, Iterable):
        raise InvarianceCheckError("invalid_invariance_contract")
    if type(max_cases) is not int or not 1 <= max_cases <= 256:
        raise InvarianceCheckError("invalid_invariance_budget")
    if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance < 0:
        raise InvarianceCheckError("invalid_invariance_tolerance")
    results = []
    seen_ids: set[str] = set()
    for index, item in enumerate(transformations):
        if index >= max_cases:
            raise InvarianceCheckError("invariance_case_budget_exceeded")
        if not isinstance(item, Mapping) or type(item.get("id")) is not str or not callable(item.get("apply")) or not callable(item.get("compare")):
            raise InvarianceCheckError("invalid_invariance_case")
        identifier = item["id"].strip()
        if not identifier:
            raise InvarianceCheckError("invariance_case_id_required")
        if identifier in seen_ids:
            raise InvarianceCheckError("invariance_case_id_must_be_unique")
        seen_ids.add(identifier)
        try:
            transformed_input = item["apply"](baseline_input)
            transformed_output = evaluate(transformed_input)
            passed = item["compare"](baseline_output, transformed_output, float(tolerance))
            if type(passed) is not bool:
                raise InvarianceCheckError("comparison_must_return_bool")
            results.append({"id": identifier, "status": "pass" if passed else "counterexample", "index": index})
        except InvarianceCheckError:
            raise
        except Exception as exc:  # callback failures are evidence, not success
            results.append({"id": identifier, "status": "execution_failed", "error_code": type(exc).__name__, "index": index})
    passed = sum(item["status"] == "pass" for item in results)
    return {"schema_version": "mathmodel.invariance/v1", "results": results,
            "passed": passed, "total": len(results),
            "status": "tested_not_falsified" if results and passed == len(results) else ("counterexample" if results else "not_assessed"),
            "policy": "finite_metamorphic_checks_do_not_prove_global_invariance"}


__all__ = ["InvarianceCheckError", "run_invariance_checks"]
