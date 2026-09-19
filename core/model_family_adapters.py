"""Uniform compile/evaluate/diagnose/patch/replay adapter contract.

The adapter is intentionally a protocol, not a registry of solved题目.  A
family implementation owns its mathematics; this module only makes sure every
candidate enters the same bounded CEGIS loop and that compile/resource errors
are not mislabeled as mathematical counterexamples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .cegis_controller import CEGISConfig, CEGISControllerError, _candidate_hash, run_cegis


class ModelFamilyAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class ModelFamilyAdapter:
    family: str
    compile: Callable[[Mapping[str, Any]], Any]
    evaluate: Callable[[Any, Sequence[Mapping[str, Any]]], Mapping[str, Any]]
    diagnose: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    patch: Callable[[Mapping[str, Any], Mapping[str, Any]], Iterable[Mapping[str, Any]]]
    replay: Callable[[Any, Sequence[Mapping[str, Any]]], Mapping[str, Any]] | None = None

    def validate(self) -> "ModelFamilyAdapter":
        if not isinstance(self.family, str) or not self.family.strip() or len(self.family) > 80:
            raise ModelFamilyAdapterError("adapter_family_invalid")
        for name in ("compile", "evaluate", "diagnose", "patch"):
            if not callable(getattr(self, name)):
                raise ModelFamilyAdapterError(f"adapter_{name}_missing")
        if self.replay is not None and not callable(self.replay):
            raise ModelFamilyAdapterError("adapter_replay_invalid")
        return self


def run_model_family_cegis(
    adapter: ModelFamilyAdapter,
    initial_candidates: Iterable[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    *,
    config: CEGISConfig | None = None,
) -> dict[str, Any]:
    """Run one family through the shared compile/evaluate/patch loop.

    ``cases`` are immutable evaluator inputs.  The adapter can return a
    numerical failure as ``not_assessed``; only an explicit ``fail`` with
    witnesses is treated as a counterexample.  This distinction is critical
    for ODE/optimization backends where a timeout is not a refutation.
    """
    adapter = adapter.validate()
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or len(cases) > 4096:
        raise ModelFamilyAdapterError("cases_invalid")
    frozen_cases = tuple(dict(case) for case in cases if isinstance(case, Mapping))
    if len(frozen_cases) != len(cases):
        raise ModelFamilyAdapterError("case_must_be_mapping")

    candidate_snapshots: dict[str, dict[str, Any]] = {}

    def evaluate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            digest = _candidate_hash(candidate)
            # Keep only candidates actually sent to the adapter.  This lets a
            # downstream competition inspect accepted repairs without
            # serializing the entire search queue or unbounded evaluator data.
            candidate_snapshots[digest] = dict(candidate)
            compiled = adapter.compile(candidate)
            result = adapter.evaluate(compiled, frozen_cases)
            if not isinstance(result, Mapping):
                return {"status": "not_assessed", "failure_code": "evaluator_result_invalid"}
            normalized = dict(result)
            normalized.setdefault("cost_units", 0)
            return normalized
        except Exception as exc:
            return {"status": "not_assessed", "failure_code": "compile_or_evaluate_failed",
                    "diagnostic": type(exc).__name__}

    def mutate(candidate: Mapping[str, Any], feedback: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
        try:
            diagnostic = adapter.diagnose(feedback)
            if not isinstance(diagnostic, Mapping):
                return ()
            proposals = adapter.patch(candidate, diagnostic)
            if isinstance(proposals, (str, bytes, Mapping)):
                return ()
            return proposals
        except Exception:
            return ()

    try:
        result = run_cegis(initial_candidates, evaluate, mutate, config=config)
    except CEGISControllerError:
        raise
    except Exception as exc:
        raise ModelFamilyAdapterError("adapter_loop_failed") from exc
    result["adapter_family"] = adapter.family
    result["accepted_candidate_snapshots"] = [
        candidate_snapshots[digest]
        for digest in result.get("accepted_candidate_hashes", [])
        if digest in candidate_snapshots
    ][:32]
    result["policy"] = {**result.get("policy", {}),
                         "compile_before_evaluate": True,
                         "resource_failure_is_not_counterexample": True,
                         "adapter_replay_required_before_publish": adapter.replay is not None}
    return result


def replay_adapter_witnesses(
    adapter: ModelFamilyAdapter, compiled: Any, witnesses: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Replay historical witnesses through a family adapter when supported."""
    adapter = adapter.validate()
    if adapter.replay is None:
        return {"status": "not_assessed", "reason": "adapter_replay_not_implemented"}
    if not isinstance(witnesses, Sequence) or isinstance(witnesses, (str, bytes)):
        raise ModelFamilyAdapterError("witnesses_invalid")
    try:
        result = adapter.replay(compiled, tuple(dict(item) for item in witnesses))
    except Exception as exc:
        return {"status": "not_assessed", "reason": "replay_failed", "error": type(exc).__name__}
    return dict(result) if isinstance(result, Mapping) else {"status": "not_assessed", "reason": "replay_result_invalid"}


__all__ = ["ModelFamilyAdapter", "ModelFamilyAdapterError", "run_model_family_cegis", "replay_adapter_witnesses"]
