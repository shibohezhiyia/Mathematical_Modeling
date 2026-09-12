"""Frozen scalar-model snapshots and one-way held-out evaluation.

The evaluator has no fitting or mutation path. These finite checks do not prove
semantic correctness, unbiased sampling, causal validity or continuous-domain
properties. Freshness outside the recorded study remains a provenance limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json

import numpy as np

from .graph_evaluator import GraphDomainError, ScalarGraphProgram, _probe_points, _rms
from .graph_experiments import (SearchExperiment, ensure_scalar_executable, finite_number,
                                fingerprint, interface_matches, point_fingerprint, restore_problem, validate_point)
from .model_hypotheses import HypothesisIR, _canonical, _id, _keys, _require
from .solver_runtime import EvaluationCounter

FROZEN_VERSION = "mathmodel.frozen-scalar-model/v1"
HOLDOUT_VERSION = "mathmodel.scalar-holdout/v1"
CONFIRMATION_VERSION = "mathmodel.scalar-confirmation/v1"


@dataclass(frozen=True)
class FrozenGraphModel:
    _json: str = field(repr=False)

    @classmethod
    def from_payload(cls, payload: dict) -> "FrozenGraphModel":
        """Validate a numerical snapshot, not authenticate a caller's history."""
        payload = json.loads(_canonical(payload))
        _keys(payload, {"schema_version", "problem", "experiment", "hypothesis", "parameters", "development_evidence"})
        _require(payload["schema_version"] == FROZEN_VERSION, "frozen_model_version_mismatch")
        contract = restore_problem(payload["problem"])
        experiment = SearchExperiment.from_payload(payload["experiment"], contract)
        graph = HypothesisIR.from_payload(payload["hypothesis"], contract)
        template = HypothesisIR.from_payload(experiment.public()["template"], contract)
        interface_matches(graph, template)
        ensure_scalar_executable(graph)
        bounds = experiment.public()["parameter_bounds"]
        _keys(payload["parameters"], set(bounds))
        _require(all(finite_number(payload["parameters"][k]) and b[0] <= payload["parameters"][k] <= b[1]
                     for k, b in bounds.items()), "invalid_frozen_parameters")
        evidence = payload["development_evidence"]
        _keys(evidence, {"experiment_hash", "hypothesis_hash", "evaluation_hash", "evaluation", "selection", "final_data_seen"})
        _require(evidence["experiment_hash"] == experiment.digest and evidence["hypothesis_hash"] == graph.digest,
                 "frozen_evidence_scope_mismatch")
        import re
        _require(type(evidence["evaluation_hash"]) is str and bool(re.fullmatch(r"[0-9a-f]{64}", evidence["evaluation_hash"])),
                 "invalid_development_evidence_hash")
        evaluation = evidence["evaluation"]
        _keys(evaluation, {"hypothesis_hash", "experiment_hash", "parameters", "fit", "check_count",
                          "checked_witness_ids", "search_rmse", "status"})
        _require(fingerprint(evaluation) == evidence["evaluation_hash"] and
                 evaluation["hypothesis_hash"] == graph.digest and evaluation["experiment_hash"] == experiment.digest
                 and evaluation["parameters"] == payload["parameters"] and evaluation["status"] == "eligible_for_confirmation",
                 "frozen_evaluation_mismatch")
        _require(evidence["selection"] == "development_pareto" and evidence["final_data_seen"] is False,
                 "frozen_selection_policy_mismatch")
        return cls(_canonical(payload))

    @property
    def digest(self):
        return sha256(self._json.encode("utf-8")).hexdigest()

    def public(self):
        return json.loads(self._json)


def freeze_search_candidate(contract, experiment, result, candidate_hash) -> FrozenGraphModel:
    """Internal session path: result must be the session's private snapshot."""
    _require(type(candidate_hash) is str and candidate_hash in result["pareto_candidates"], "candidate_not_selectable")
    _require(result["experiment_hash"] == experiment.digest and result["contract_hash"] == contract.digest,
             "frozen_evidence_scope_mismatch")
    view = next(h for h in result["hypotheses"] if h["hypothesis_hash"] == candidate_hash)
    graph = {k: view[k] for k in ("id", "contract_hash", "nodes", "outputs", "assumptions")}
    latest = next(r for r in reversed(result["reports"]) if r["hypothesis_hash"] == candidate_hash)
    _require(latest["status"] == "eligible_for_confirmation" and latest["replay_passed"] is True,
             "candidate_not_selectable")
    # Operational elapsed times and process identifiers do not affect numerical identity.
    evaluation = {k: latest[k] for k in ("hypothesis_hash", "experiment_hash", "parameters", "fit", "check_count",
                                        "checked_witness_ids", "search_rmse", "status")}
    return FrozenGraphModel.from_payload({"schema_version": FROZEN_VERSION, "problem": contract.public(),
        "experiment": experiment.public(), "hypothesis": graph, "parameters": latest["parameters"],
        "development_evidence": {"experiment_hash": experiment.digest, "hypothesis_hash": candidate_hash,
            "evaluation_hash": fingerprint(evaluation), "evaluation": evaluation,
            "selection": "development_pareto", "final_data_seen": False}})


@dataclass(frozen=True)
class HeldoutCases:
    _json: str = field(repr=False)

    @classmethod
    def from_payload(cls, payload: dict, model: FrozenGraphModel) -> "HeldoutCases":
        payload = json.loads(_canonical(payload))
        _keys(payload, {"schema_version", "cases"})
        _require(payload["schema_version"] == HOLDOUT_VERSION, "holdout_version_mismatch")
        cases = payload["cases"]
        _require(type(cases) is list and 1 <= len(cases) <= 256, "holdout_case_budget")
        spec = model.public()["experiment"]
        outputs = set(model.public()["hypothesis"]["outputs"])
        used = {point_fingerprint(c["bindings"]) for partition in ("training_cases", "search_cases") for c in spec[partition]}
        used.update(point_fingerprint(p) for p in _probe_points(spec))
        ids, points = set(), set()
        for case in cases:
            _keys(case, {"id", "bindings", "expected"})
            _id(case["id"])
            _require(case["id"] not in ids, "duplicate_holdout_id")
            ids.add(case["id"])
            validate_point(case["bindings"], spec["domain"])
            identity = point_fingerprint(case["bindings"])
            _require(identity not in used, "holdout_development_overlap")
            _require(identity not in points, "duplicate_holdout_point")
            points.add(identity)
            _keys(case["expected"], outputs)
            _require(all(finite_number(v) for v in case["expected"].values()), "invalid_holdout_expected")
        return cls(_canonical(payload))

    def public(self):
        return json.loads(self._json)

    @property
    def digest(self):
        # IDs, order and numeric spelling cannot manufacture new data identity.
        rows = [{"point": point_fingerprint(c["bindings"]),
                 "expected": {k: float(v) if v != 0 else 0.0 for k, v in c["expected"].items()}}
                for c in self.public()["cases"]]
        return fingerprint(sorted(rows, key=lambda c: c["point"]))

    @property
    def point_hashes(self):
        return sorted(point_fingerprint(c["bindings"]) for c in self.public()["cases"])


def evaluate_frozen_request(request: dict, *, max_evaluations: int) -> dict:
    """Trusted worker entry: no optimizer, no trainable parameters, no search."""
    _keys(request, {"frozen_model", "holdout"})
    frozen = FrozenGraphModel.from_payload(request["frozen_model"])
    data = HeldoutCases.from_payload(request["holdout"], frozen)
    snapshot, cases = frozen.public(), data.public()["cases"]
    contract = restore_problem(snapshot["problem"])
    graph = HypothesisIR.from_payload(snapshot["hypothesis"], contract)
    spec, parameters = snapshot["experiment"], snapshot["parameters"]
    counter = EvaluationCounter(max_evaluations)
    program = ScalarGraphProgram(graph, counter, reuse_intermediates=False)
    points = [c["bindings"] for c in cases]
    try:
        predictions = list(program.evaluate(points, parameters))
    except GraphDomainError:
        predictions = []
        for point in points:
            try:
                predictions.append(program.evaluate([point], parameters)[0])
            except GraphDomainError:
                predictions.append(None)
    failures, checked = [], 0
    for case, predicted in zip(cases, predictions):
        checked += 1
        if predicted is None:
            failures.append({"case_id": case["id"], "check": "finite_output", "reason": "numeric_domain"})
            continue
        checked += len(program.outputs) + len(spec["properties"])
        for output, value in zip(program.outputs, predicted):
            expected = case["expected"][output]
            difference = abs(float(value) - expected)
            tolerance = spec["absolute_tolerance"] + spec["relative_tolerance"] * abs(expected)
            if difference > tolerance:
                failures.append({"case_id": case["id"], "output": output, "check": "reference",
                                 "reason": "locked_tolerance_exceeded"})
        for prop in spec["properties"]:
            value = float(predicted[program.outputs.index(prop["output"])])
            violated = False
            for key in ("lower", "upper"):
                bound = prop[key]
                if bound is not None:
                    tolerance = spec["absolute_tolerance"] + spec["relative_tolerance"] * abs(bound)
                    violated |= value < bound - tolerance if key == "lower" else value > bound + tolerance
            if violated:
                failures.append({"case_id": case["id"], "check": "conditional_property", "property_id": prop["id"],
                                 "reason": "locked_bound_violated"})
    metrics = []
    if all(p is not None for p in predictions):
        actual = np.asarray(predictions)
        target = np.asarray([[c["expected"][k] for k in program.outputs] for c in cases])
        with np.errstate(over="ignore", invalid="ignore"):
            residual = actual - target
        if np.isfinite(residual).all():
            for index, output in enumerate(program.outputs):
                metric = {"output": output, "rmse": _rms(residual[:, index]),
                          "max_absolute_error": float(np.max(np.abs(residual[:, index]))),
                          "training_mean_baseline_rmse": None}
                if spec["training_cases"]:
                    baseline = float(np.mean([c["expected"][output] for c in spec["training_cases"]]))
                    metric["training_mean_baseline_rmse"] = _rms(target[:, index] - baseline)
                metrics.append(metric)
    return {"schema_version": CONFIRMATION_VERSION, "frozen_model_hash": frozen.digest,
            "hypothesis_hash": graph.digest, "experiment_hash": fingerprint(spec), "holdout_hash": data.digest,
            "status": "failed_heldout_checks" if failures else "passed_finite_heldout_checks",
            "parameters": parameters, "parameters_refitted": False, "case_count": len(cases),
            "check_count": checked, "planned_check_count": len(cases) * (1 + len(program.outputs) + len(spec["properties"])),
            "failed_check_count": len(failures), "failures": failures[:64],
            "failure_details_truncated": len(failures) > 64, "metrics": metrics,
            "evaluations_used": counter.used, "may_feed_search": False,
            "semantic_verdict": "not_assessed", "continuous_domain_proof": False,
            "sampling_representativeness": "not_assessed", "uncertainty_calibration": "not_assessed",
            "split_guarantee": "disjoint_exact_input_points_only", "entity_time_dependence_assessed": False,
            "authority": "finite_heldout_evidence_under_declared_assumptions"}
