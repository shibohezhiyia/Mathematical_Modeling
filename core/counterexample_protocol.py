"""Typed counterexample records and bounded replay protocol.

The protocol deliberately separates a witness from an explanation.  A failed
solver call is not silently promoted to a mathematical counterexample, and a
finite search without a witness is never reported as a proof.  Callers must
provide the category and the evidence needed for that category; this makes
CEGIS feedback useful without allowing a language model to invent a reason.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


class CounterexampleProtocolError(ValueError):
    """Raised when a witness does not satisfy the evidence contract."""


SCHEMA_VERSION = "mathmodel.counterexample-protocol/v1"
_CATEGORIES = {
    "normative_violation",  # in-domain violation of a stated property
    "empirical_mismatch",   # in-domain disagreement with an observed value
    "numerical_failure",    # non-finite value, solver failure, or instability
    "out_of_domain",        # behavior outside the declared test domain
}
_REPLAY_STATUSES = {"reproduced", "not_reproduced", "not_assessed"}
_CASE_ORDER = {"endpoint": 0, "degenerate": 1, "known_feasible": 2,
               "known_infeasible": 3, "historical": 4, "random": 5, "optimized": 6}


def _bounded_text(value: Any, name: str, limit: int = 500) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise CounterexampleProtocolError(f"invalid_{name}")
    return value


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _safe_scalar(value: Any) -> bool:
    """Accept JSON-like evidence only, while rejecting non-finite numbers."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return len(value) <= 128 and all(isinstance(key, str) and len(key) <= 160 and _safe_scalar(item)
                                         for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return len(value) <= 256 and all(_safe_scalar(item) for item in value)
    return False


def normalize_counterexample(witness: Mapping[str, Any], *, contract_version: str) -> dict[str, Any]:
    """Validate and canonicalize one auditable counterexample.

    ``category`` is intentionally required instead of inferred from free text.
    This prevents a numerical exception from being mislabeled as a structural
    refutation.  The returned record is JSON-safe metadata; it contains no
    executable callback or arbitrary code.
    """
    if not isinstance(witness, Mapping):
        raise CounterexampleProtocolError("witness_must_be_mapping")
    category = witness.get("category")
    if category not in _CATEGORIES:
        raise CounterexampleProtocolError("counterexample_category_required")
    version = _bounded_text(contract_version, "contract_version", 160)
    witness_id = witness.get("witness_id", witness.get("id"))
    if not isinstance(witness_id, str) or not witness_id or len(witness_id) > 160:
        raise CounterexampleProtocolError("witness_id_required")
    if category in {"normative_violation", "empirical_mismatch"} and witness.get("in_domain") is not True:
        raise CounterexampleProtocolError("in_domain_evidence_required")
    if category == "out_of_domain" and witness.get("in_domain") is not False:
        raise CounterexampleProtocolError("out_of_domain_evidence_required")
    if category == "normative_violation":
        _bounded_text(witness.get("property_id"), "property_id", 160)
    if category == "empirical_mismatch":
        if "observed" not in witness or "predicted" not in witness:
            raise CounterexampleProtocolError("observed_and_predicted_required")
    if category == "numerical_failure":
        _bounded_text(str(witness.get("error", "")), "numerical_error", 1000)
    if category == "out_of_domain":
        _bounded_text(str(witness.get("domain" , "")), "domain", 500)
    values = witness.get("values", {})
    if values is not None and not isinstance(values, Mapping):
        raise CounterexampleProtocolError("witness_values_must_be_mapping")
    if isinstance(values, Mapping) and len(values) > 64:
        raise CounterexampleProtocolError("witness_values_too_large")
    for value in (values or {}).values():
        if not _safe_scalar(value):
            raise CounterexampleProtocolError("nonfinite_witness_value")
    replay = witness.get("replay", {})
    if replay is not None and not isinstance(replay, Mapping):
        raise CounterexampleProtocolError("replay_must_be_mapping")
    status = (replay or {}).get("status", "not_assessed")
    if status not in _REPLAY_STATUSES:
        raise CounterexampleProtocolError("invalid_replay_status")
    attempts = (replay or {}).get("attempts", 0)
    if type(attempts) is not int or not 0 <= attempts <= 10000:
        raise CounterexampleProtocolError("invalid_replay_attempts")
    result = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": version,
        "witness_id": witness_id,
        "category": category,
        "in_domain": witness.get("in_domain"),
        "property_id": witness.get("property_id"),
        "observed": witness.get("observed"),
        "predicted": witness.get("predicted"),
        "error": witness.get("error"),
        "domain": witness.get("domain"),
        "values": dict(values or {}),
        "replay": {"status": status, "attempts": attempts},
        "evidence": str(witness.get("evidence", ""))[:1000],
    }
    return result


def replay_counterexamples(
    witnesses: Sequence[Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any]], bool],
    *,
    contract_version: str,
    max_records: int = 128,
) -> dict[str, Any]:
    """Replay bounded historical witnesses after a model repair.

    ``evaluator`` receives the normalized witness and returns whether the
    original violation is reproduced.  Exceptions are recorded as
    ``not_assessed``; they never become a successful replay or a new proof.
    """
    if not isinstance(witnesses, Sequence) or isinstance(witnesses, (str, bytes)):
        raise CounterexampleProtocolError("witnesses_must_be_sequence")
    if type(max_records) is not int or not 1 <= max_records <= 4096:
        raise CounterexampleProtocolError("invalid_replay_budget")
    if not callable(evaluator):
        raise CounterexampleProtocolError("evaluator_must_be_callable")
    records: list[dict[str, Any]] = []
    reproduced = not_reproduced = not_assessed = 0
    for raw in list(witnesses)[:max_records]:
        try:
            item = normalize_counterexample(raw, contract_version=contract_version)
        except CounterexampleProtocolError as exc:
            records.append({"status": "not_assessed", "error": str(exc)})
            not_assessed += 1
            continue
        try:
            result = evaluator(item)
            if type(result) is not bool:
                raise CounterexampleProtocolError("evaluator_must_return_bool")
            status = "reproduced" if result else "not_reproduced"
        except Exception as exc:  # callbacks are untrusted; preserve audit trail
            status = "not_assessed"
            item["replay"]["error"] = type(exc).__name__
        item["replay"] = {"status": status, "attempts": 1, **{
            key: value for key, value in item["replay"].items() if key == "error"}}
        records.append(item)
        if status == "reproduced":
            reproduced += 1
        elif status == "not_reproduced":
            not_reproduced += 1
        else:
            not_assessed += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_version": contract_version,
        "records": records,
        "reproduced": reproduced,
        "not_reproduced": not_reproduced,
        "not_assessed": not_assessed,
        "status": "all_replayed" if records and not not_assessed else ("partial" if records else "not_assessed"),
        "policy": "finite_replay_is_evidence_not_mathematical_proof;_not_assessed_blocks_promotion",
    }


def run_counterexample_suite(
    cases: Sequence[Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    contract_version: str,
    max_cases: int = 256,
) -> dict[str, Any]:
    """Run deterministic boundary cases before stochastic/adversarial search.

    Cases are ordered as endpoint, degenerate, known feasible/infeasible,
    historical, random, and optimized.  The evaluator returns a small mapping
    with ``violation`` (bool) and, when true, the explicit witness fields
    accepted by :func:`normalize_counterexample`.  A clean finite run is
    reported as ``tested_not_falsified`` rather than ``proved``.
    """
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise CounterexampleProtocolError("cases_must_be_sequence")
    if type(max_cases) is not int or not 1 <= max_cases <= 4096:
        raise CounterexampleProtocolError("invalid_suite_budget")
    if not callable(evaluator):
        raise CounterexampleProtocolError("evaluator_must_be_callable")
    selected = list(cases)[:max_cases]
    normalized_cases: list[dict[str, Any]] = []
    for raw in selected:
        if not isinstance(raw, Mapping):
            raise CounterexampleProtocolError("case_must_be_mapping")
        case_id = raw.get("case_id")
        kind = raw.get("kind")
        if not isinstance(case_id, str) or not case_id or len(case_id) > 160:
            raise CounterexampleProtocolError("case_id_required")
        if kind not in _CASE_ORDER:
            raise CounterexampleProtocolError("invalid_case_kind")
        normalized_cases.append({"case_id": case_id, "kind": kind,
                                 "values": dict(raw.get("values", {})) if isinstance(raw.get("values", {}), Mapping) else {}})
    normalized_cases.sort(key=lambda item: (_CASE_ORDER[item["kind"]], item["case_id"]))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    not_assessed = 0
    for case in normalized_cases:
        try:
            result = evaluator(case)
            if not isinstance(result, Mapping) or type(result.get("violation")) is not bool:
                raise CounterexampleProtocolError("evaluator_result_must_include_bool_violation")
            if not result["violation"]:
                records.append({"case_id": case["case_id"], "kind": case["kind"], "status": "pass"})
                continue
            raw_witness = dict(result)
            raw_witness.setdefault("witness_id", case["case_id"])
            raw_witness.pop("violation", None)
            witness = normalize_counterexample(raw_witness, contract_version=contract_version)
            witness["case_id"] = case["case_id"]
            witness["case_kind"] = case["kind"]
            records.append(witness)
            failures.append(witness)
        except Exception as exc:  # untrusted evaluator; preserve but do not classify as proof
            not_assessed += 1
            records.append({"case_id": case["case_id"], "kind": case["kind"],
                            "status": "not_assessed", "error": type(exc).__name__})
    if failures:
        status = "counterexample_found"
    elif not_assessed:
        status = "not_assessed"
    elif records:
        status = "tested_not_falsified"
    else:
        status = "not_assessed"
    return {"schema_version": SCHEMA_VERSION, "contract_version": contract_version,
            "status": status, "tested_cases": len(records), "records": records,
            "counterexamples": failures, "not_assessed": not_assessed,
            "policy": "ordered_boundary_first;_finite_no_witness_is_tested_not_falsified"}


__all__ = ["SCHEMA_VERSION", "CounterexampleProtocolError", "normalize_counterexample",
           "replay_counterexamples", "run_counterexample_suite"]
