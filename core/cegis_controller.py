"""Generic bounded CEGIS orchestration for independently validated candidates.

The controller is deliberately solver-agnostic: evaluators decide whether a
candidate satisfies their mathematical contract, while mutators only propose
new candidates.  Neither callback can promote a result to a proof.  Resource
and process isolation remain the responsibility of the evaluator/backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import time
from typing import Any, Callable, Iterable, Mapping

from .ir_schema import canonical_json


SCHEMA_VERSION = "mathmodel.cegis-controller/v1"
_STATUSES = frozenset({"pass", "fail", "not_assessed"})


class CEGISControllerError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CEGISConfig:
    max_rounds: int = 16
    max_candidates: int = 64
    max_repairs: int = 32
    max_counterexamples: int = 128
    max_wall_seconds: float = 0.0
    max_cost_units: int = 0

    def __post_init__(self) -> None:
        if type(self.max_rounds) is not int or not 1 <= self.max_rounds <= 256:
            raise CEGISControllerError("invalid_round_budget")
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 1024:
            raise CEGISControllerError("invalid_candidate_budget")
        if type(self.max_repairs) is not int or not 0 <= self.max_repairs <= 1024:
            raise CEGISControllerError("invalid_repair_budget")
        if type(self.max_counterexamples) is not int or not 1 <= self.max_counterexamples <= 4096:
            raise CEGISControllerError("invalid_counterexample_budget")
        if type(self.max_wall_seconds) not in (int, float) or not math.isfinite(float(self.max_wall_seconds)) or not 0 <= float(self.max_wall_seconds) <= 86_400:
            raise CEGISControllerError("invalid_wall_time_budget")
        if type(self.max_cost_units) is not int or not 0 <= self.max_cost_units <= 10**12:
            raise CEGISControllerError("invalid_cost_budget")


def _candidate_hash(candidate: Mapping[str, Any]) -> str:
    try:
        encoded = canonical_json(dict(candidate))
    except Exception as exc:
        raise CEGISControllerError("candidate_not_bounded_json") from exc
    return sha256(encoded.encode("utf-8")).hexdigest()


def _feedback(result: Mapping[str, Any]) -> dict[str, Any]:
    violations = result.get("violations", [])
    if not isinstance(violations, list):
        violations = []
    bounded = []
    for item in violations[:16]:
        if isinstance(item, Mapping):
            bounded.append({
                "reason": str(item.get("reason", "unknown"))[:120],
                "witness_id": str(item.get("witness_id", ""))[:80],
            })
        else:
            bounded.append({"reason": "invalid_violation", "witness_id": ""})
    return {
        "status": str(result.get("status", "not_assessed"))[:32],
        "failure_code": str(result.get("failure_code", ""))[:120],
        "violations": bounded,
    }


def run_cegis(
    initial_candidates: Iterable[Mapping[str, Any]],
    evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    mutate: Callable[[Mapping[str, Any], Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    *,
    config: CEGISConfig | None = None,
) -> dict[str, Any]:
    """Run a bounded candidate/反例/变异 loop and return an auditable summary."""
    if not callable(evaluate) or not callable(mutate):
        raise CEGISControllerError("callbacks_must_be_callable")
    config = config or CEGISConfig()
    queue: list[tuple[dict[str, Any], str | None]] = []
    seen: set[str] = set()
    for candidate in initial_candidates:
        if not isinstance(candidate, Mapping):
            raise CEGISControllerError("candidate_must_be_object")
        copied = dict(candidate)
        digest = _candidate_hash(copied)
        if digest not in seen:
            seen.add(digest)
            queue.append((copied, None))
        if len(seen) >= config.max_candidates:
            break

    records: list[dict[str, Any]] = []
    accepted: list[str] = []
    counterexample_archive: list[dict[str, Any]] = []
    dropped_counterexamples = 0
    lineage: dict[str, str | None] = {}
    repairs = 0
    rounds = 0
    started_at = time.monotonic()
    cost_used = 0
    budget_reason = None
    while queue and rounds < config.max_rounds:
        if config.max_wall_seconds and time.monotonic() - started_at >= config.max_wall_seconds:
            budget_reason = "wall_time_budget_exhausted"
            break
        if config.max_cost_units and cost_used >= config.max_cost_units:
            budget_reason = "cost_budget_exhausted"
            break
        candidate, parent_hash = queue.pop(0)
        digest = _candidate_hash(candidate)
        lineage[digest] = parent_hash
        rounds += 1
        try:
            result = evaluate(candidate)
            if not isinstance(result, Mapping):
                raise CEGISControllerError("evaluator_result_must_be_object")
            status = result.get("status")
            if status not in _STATUSES:
                raise CEGISControllerError("invalid_evaluator_status")
            violations = result.get("violations", [])
            if not isinstance(violations, list) or len(violations) > 128:
                raise CEGISControllerError("invalid_evaluator_violations")
            raw_cost = result.get("cost_units", 0)
            if type(raw_cost) is not int or raw_cost < 0:
                raise CEGISControllerError("invalid_evaluator_cost_units")
            cost_used += raw_cost
            if config.max_cost_units and cost_used > config.max_cost_units:
                records.append({"round": rounds, "candidate_hash": digest, "parent_hash": parent_hash,
                                "status": "not_assessed", "score": None, "violation_count": 0,
                                "feedback": {"failure_code": "cost_budget_exhausted"},
                                "cost_units": cost_used})
                budget_reason = "cost_budget_exhausted"
                break
            feedback = _feedback(result)
            current_counterexamples = [
                {
                    "candidate_hash": digest,
                    "round": rounds,
                    "failure_code": feedback.get("failure_code", ""),
                    "reason": item.get("reason", "unknown"),
                    "witness_id": item.get("witness_id", ""),
                }
                for item in feedback["violations"]
            ]
            if current_counterexamples:
                counterexample_archive.extend(current_counterexamples)
                if len(counterexample_archive) > config.max_counterexamples:
                    dropped_counterexamples += len(counterexample_archive) - config.max_counterexamples
                    counterexample_archive = counterexample_archive[-config.max_counterexamples:]
            # The mutator receives a bounded, redacted archive.  Candidate
            # contents and raw evaluator output never enter this feedback.
            feedback["counterexample_archive"] = list(counterexample_archive)
            feedback["counterexample_archive_count"] = len(counterexample_archive)
            score = result.get("score")
            if type(score) not in (int, float) or not math.isfinite(float(score)):
                score = None
            record = {"round": rounds, "candidate_hash": digest, "parent_hash": parent_hash,
                      "status": status,
                      "score": float(score) if score is not None else None,
                      "violation_count": len(violations), "feedback": feedback}
            records.append(record)
            if status == "pass" and not violations:
                accepted.append(digest)
                continue
            if repairs >= config.max_repairs or len(seen) >= config.max_candidates:
                continue
            repairs += 1
            try:
                proposals = mutate(candidate, feedback)
                if not isinstance(proposals, Iterable) or isinstance(proposals, (str, bytes, Mapping)):
                    raise CEGISControllerError("mutator_result_must_be_iterable")
                for proposal in proposals:
                    if not isinstance(proposal, Mapping):
                        continue
                    copied = dict(proposal)
                    proposal_hash = _candidate_hash(copied)
                    if proposal_hash in seen:
                        continue
                    seen.add(proposal_hash)
                    queue.append((copied, digest))
                    if len(seen) >= config.max_candidates:
                        break
            except Exception:
                records[-1]["mutator_error"] = "mutator_failed_safe"
        except CEGISControllerError as exc:
            records.append({"round": rounds, "candidate_hash": digest, "parent_hash": parent_hash,
                            "status": "not_assessed", "score": None,
                            "violation_count": 0, "feedback": {"failure_code": exc.code},
                            "evaluator_error": exc.code})
        except Exception:
            records.append({"round": rounds, "candidate_hash": digest, "parent_hash": parent_hash,
                            "status": "not_assessed", "score": None,
                            "violation_count": 0,
                            "feedback": {"failure_code": "evaluator_failed_safe"},
                            "evaluator_error": "evaluator_failed_safe"})

    if accepted:
        status = "accepted_candidates"
    elif queue or rounds >= config.max_rounds or repairs >= config.max_repairs:
        status = "budget_exhausted"
    else:
        status = "candidate_set_inadequate"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "rounds": rounds,
        "candidate_count": len(seen),
        "repair_count": repairs,
        "accepted_candidate_hashes": list(dict.fromkeys(accepted)),
        "records": records,
        "lineage": dict(lineage),
        "counterexample_archive": list(counterexample_archive),
        "counterexample_archive_count": len(counterexample_archive),
        "dropped_counterexamples": dropped_counterexamples,
        "cost_units_used": cost_used,
        "budget_reason": budget_reason,
        "policy": {
            "accepted_is_tested_not_falsified": True,
            "no_mathematical_proof": True,
            "callbacks_are_not_sandboxed": True,
            "mutator_receives_bounded_counterexample_archive": True,
            "last_unchecked_candidate_never_promoted": True,
            "wall_time_and_cost_budgets_are_hard": True,
        },
    }


def minimize_counterexample(
    counterexample: Mapping[str, Any],
    is_violation: Callable[[Mapping[str, Any]], bool],
    shrink: Callable[[Mapping[str, Any]], Iterable[Mapping[str, Any]]],
    *,
    max_steps: int = 64,
) -> dict[str, Any]:
    """Minimize one witness by bounded, deterministic counterexample search.

    ``shrink`` proposes smaller inputs; only candidates that still reproduce
    the same violation are accepted.  The result is a local minimum under the
    supplied shrinker, never a proof of global minimality.  Exceptions and
    malformed proposals stop that branch as ``not_assessed`` rather than
    converting a resource failure into a mathematical counterexample.
    """
    if not isinstance(counterexample, Mapping):
        raise CEGISControllerError("counterexample_must_be_object")
    if not callable(is_violation) or not callable(shrink):
        raise CEGISControllerError("minimizer_callbacks_must_be_callable")
    if type(max_steps) is not int or not 1 <= max_steps <= 1024:
        raise CEGISControllerError("invalid_minimization_budget")
    current = dict(counterexample)
    try:
        initial_violation = is_violation(current)
        if type(initial_violation) is not bool:
            raise CEGISControllerError("violation_check_must_return_boolean")
        if not initial_violation:
            return {"status": "not_assessed", "reason": "input_is_not_a_violation",
                    "original": dict(counterexample), "final": dict(current), "tested": 1}
    except CEGISControllerError:
        raise
    except Exception:
        return {"status": "not_assessed", "reason": "violation_check_failed",
                "original": dict(counterexample), "final": dict(current), "tested": 1}
    tested = 1
    accepted_steps = 0
    seen = {_candidate_hash(current)}

    def complexity(value: Any) -> tuple[float, int]:
        """A deterministic size proxy; not a domain-specific minimality claim."""
        if isinstance(value, Mapping):
            parts = [complexity(key) for key in value]
            parts.extend(complexity(child) for child in value.values())
            return (sum(item[0] for item in parts), len(canonical_json(dict(value))))
        if isinstance(value, (list, tuple)):
            parts = [complexity(item) for item in value]
            return (sum(item[0] for item in parts), len(canonical_json(value)))
        if type(value) in (int, float) and not isinstance(value, bool):
            number = float(value)
            return (abs(number) if math.isfinite(number) else float("inf"), len(str(value)))
        if isinstance(value, str):
            return (0.0, len(value))
        return (0.0, len(str(value)))

    for _ in range(max_steps):
        try:
            proposals = shrink(current)
            if not isinstance(proposals, Iterable) or isinstance(proposals, (str, bytes, Mapping)):
                return {"status": "not_assessed", "reason": "shrink_result_must_be_iterable",
                        "original": dict(counterexample), "final": dict(current), "tested": tested}
            viable: list[tuple[int, str, dict[str, Any]]] = []
            for proposal in proposals:
                if tested >= max_steps + 1:
                    break
                if not isinstance(proposal, Mapping):
                    continue
                copied = dict(proposal)
                digest = _candidate_hash(copied)
                if digest in seen:
                    continue
                seen.add(digest)
                tested += 1
                try:
                    still_fails = is_violation(copied)
                except Exception:
                    continue
                if type(still_fails) is not bool or not still_fails:
                    continue
                candidate_complexity = complexity(copied)
                viable.append((candidate_complexity[0] * 1_000_000 + candidate_complexity[1], digest, copied))
            if not viable:
                break
            current_complexity = complexity(current)
            current_size = current_complexity[0] * 1_000_000 + current_complexity[1]
            smaller = [item for item in viable if item[0] < current_size]
            if not smaller:
                break
            _, _, current = min(smaller, key=lambda item: (item[0], item[1]))
            accepted_steps += 1
        except Exception:
            return {"status": "not_assessed", "reason": "minimization_failed",
                    "original": dict(counterexample), "final": dict(current),
                    "tested": tested, "accepted_steps": accepted_steps}
    return {
        "status": "minimized" if accepted_steps else "already_minimal_or_no_smaller_witness",
        "original": dict(counterexample), "final": dict(current), "tested": tested,
        "accepted_steps": accepted_steps,
        "proof_status": "locally_minimized_under_declared_shrinker",
        "policy": "finite_search_not_global_minimality_proof",
    }


__all__ = ["SCHEMA_VERSION", "CEGISControllerError", "CEGISConfig", "run_cegis", "minimize_counterexample"]
