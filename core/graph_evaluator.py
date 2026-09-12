"""Trusted scalar-graph interpreter and training-only parameter estimation.

Production entry is the resource-supervised worker, not in-process model code.
No strings are evaluated as programs. Every result is conditional on the bound
experiment; sampled properties never become universal/semantic proof.
"""
from __future__ import annotations

import random
import re
from collections import OrderedDict

import numpy as np

from .graph_experiments import (
    SearchExperiment, ensure_scalar_executable, fingerprint, interface_matches, point_fingerprint,
    restore_problem, validate_point,
)
from .model_hypotheses import HypothesisIR, _keys, _require
from .solver_runtime import EvaluationCounter

EVALUATOR_VERSION = "mathmodel.scalar-graph-evaluator/v4"


class GraphDomainError(ArithmeticError):
    def __init__(self, node_id):
        self.node_id = node_id
        super().__init__("graph_numeric_domain")


class ScalarGraphProgram:
    """Compile dependency order once; evaluate only allow-listed array arithmetic."""

    def __init__(self, graph: HypothesisIR, counter: EvaluationCounter, *, reuse_intermediates: bool = True):
        ensure_scalar_executable(graph)
        _require(type(reuse_intermediates) is bool, "invalid_intermediate_cache_policy")
        payload = graph.payload()
        by_id = {n["id"]: n for n in payload["nodes"]}
        self.order, seen = [], set()

        def visit(key):
            if key in seen:
                return
            for child in by_id[key]["inputs"]:
                visit(child)
            seen.add(key)
            self.order.append(by_id[key])

        for key in payload["outputs"]:
            visit(key)
        self.outputs, self.counter = payload["outputs"], counter
        self.symbols = [n["id"] for n in self.order if n["op"] == "variable"]
        self.parameter_dependent = {}
        for n in self.order:
            self.parameter_dependent[n["id"]] = n["op"] == "parameter" or any(
                self.parameter_dependent[i] for i in n["inputs"])
        self.reuse_intermediates = reuse_intermediates
        self._cache = OrderedDict()
        self.nodes_computed = self.nodes_reused = 0

    def cache_statistics(self):
        return {"enabled": self.reuse_intermediates, "scope": "one_graph_exact_ordered_input_points",
                "entries": len(self._cache), "max_entries": 4, "max_array_bytes": 2_097_152,
                "array_bytes": sum(v.nbytes for entry in self._cache.values() for v in entry.values()),
                "nodes_computed": self.nodes_computed, "nodes_reused": self.nodes_reused,
                "verdicts_cached": False, "checks_skipped": False}

    def evaluate(self, points: list[dict], parameters: dict) -> np.ndarray:
        self.counter.consume()  # Includes finite-difference and replay calls.
        _require(len(points) <= 512, "graph_batch_limit")
        signature = tuple(tuple(float(p[key]) for key in self.symbols) for p in points)
        cached = self._cache.get(signature, {}) if self.reuse_intermediates else {}
        independent = {}
        values = {}
        for node in self.order:
            key, op = node["id"], node["op"]
            if key in cached:
                values[key] = cached[key]
                independent[key] = cached[key]
                self.nodes_reused += 1
                continue
            self.nodes_computed += 1
            args = [values[i] for i in node["inputs"]]
            try:
                with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                    if op == "variable":
                        value = np.asarray([p[key] for p in points], dtype=float)
                    elif op == "parameter":
                        value = np.full(len(points), parameters[key], dtype=float)
                    elif op == "constant":
                        value = np.full(len(points), node["attributes"]["value"], dtype=float)
                    elif op == "add":
                        value = args[0] + args[1]
                    elif op == "subtract":
                        value = args[0] - args[1]
                    elif op == "multiply":
                        value = args[0] * args[1]
                    elif op == "divide":
                        value = args[0] / args[1]
                    elif op == "minimum":
                        value = np.minimum(args[0], args[1])
                    elif op == "maximum":
                        value = np.maximum(args[0], args[1])
                    elif op == "negate":
                        value = -args[0]
                    elif op == "abs":
                        value = np.abs(args[0])
                    elif op == "sqrt":
                        value = np.sqrt(args[0])
                    elif op == "exp":
                        value = np.exp(args[0])
                    elif op == "log":
                        value = np.log(args[0])
                    elif op == "sin":
                        value = np.sin(args[0])
                    elif op == "cos":
                        value = np.cos(args[0])
                    elif op == "observation":
                        value = args[0]
                    else:
                        raise ValueError("unsupported scalar operator")
                if not np.isfinite(value).all():
                    raise FloatingPointError()
                values[key] = value
                if not self.parameter_dependent[key]:
                    independent[key] = value
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise GraphDomainError(key) from exc
        # Publish only completed evaluations. A numeric failure never becomes a
        # cached verdict or a permanently rejected input.
        if self.reuse_intermediates:
            self._cache[signature] = independent
            self._cache.move_to_end(signature)
            while len(self._cache) > 4 or sum(v.nbytes for entry in self._cache.values()
                                             for v in entry.values()) > 2_097_152:
                self._cache.popitem(last=False)
        return np.column_stack([values[key] for key in self.outputs])


def _probe_points(spec: dict) -> list[dict]:
    """Fixed boundaries/zero slices plus seeded interior; never a full-grid claim."""
    domain = spec["domain"]
    if not spec["probe_count"]:
        return []
    center = {key: (bounds[0] + bounds[1]) / 2 for key, bounds in domain.items()}
    points = [center, {k: b[0] for k, b in domain.items()}, {k: b[1] for k, b in domain.items()}]
    for key, bounds in domain.items():
        for value in (*bounds, *([0.0] if bounds[0] <= 0 <= bounds[1] else [])):
            points.append({**center, key: value})
    generator = random.Random(spec["seed"])
    points.extend({key: generator.uniform(*bounds) for key, bounds in domain.items()}
                  for _ in range(spec["probe_count"]))
    return list({point_fingerprint(p): p for p in points}.values())


def _validate_replay(replay: list, experiment: SearchExperiment):
    _require(type(replay) is list and len(replay) <= 128, "replay_budget_exceeded")
    spec = experiment.public()
    cases = {c["id"]: c for c in spec["search_cases"]}
    properties = {p["id"]: p for p in spec["properties"]}
    seen = set()
    for witness in replay:
        _keys(witness, {"id", "experiment_hash", "origin_hash", "bindings", "check"})
        _require(witness["experiment_hash"] == experiment.digest, "stale_counterexample_scope")
        _require(type(witness["origin_hash"]) is str and bool(re.fullmatch(r"[0-9a-f]{64}", witness["origin_hash"])),
                 "invalid_counterexample_origin")
        validate_point(witness["bindings"], spec["domain"])
        check = witness["check"]
        _require(type(check) is dict, "invalid_counterexample_check")
        kind = check.get("kind")
        if kind == "reference":
            _keys(check, {"kind", "case_id"})
            _require(type(check["case_id"]) is str and check["case_id"] in cases and
                     witness["bindings"] == cases[check["case_id"]]["bindings"],
                     "counterexample_reference_mismatch")
        elif kind == "property":
            _keys(check, {"kind", "property_id"})
            _require(type(check["property_id"]) is str and check["property_id"] in properties, "unknown_counterexample_property")
        else:
            _require(kind == "finite", "invalid_counterexample_check")
            _keys(check, {"kind"})
        identity = {k: witness[k] for k in ("experiment_hash", "bindings", "check")}
        _require(witness["id"] == fingerprint(identity), "counterexample_hash_mismatch")
        _require(witness["id"] not in seen, "duplicate_counterexample")
        seen.add(witness["id"])


def _rms(values) -> float:
    scale = float(np.max(np.abs(values), initial=0.0))
    return float(scale * np.sqrt(np.mean((values / scale)**2))) if scale else 0.0


def evaluate_graph_request(request: dict, *, max_evaluations: int) -> dict:
    _keys(request, {"problem", "hypothesis", "experiment", "replay"}, {"reuse_intermediates"})
    contract = restore_problem(request["problem"])
    experiment = SearchExperiment.from_payload(request["experiment"], contract)
    graph = HypothesisIR.from_payload(request["hypothesis"], contract)
    template = HypothesisIR.from_payload(experiment.public()["template"], contract)
    interface_matches(graph, template)
    _validate_replay(request["replay"], experiment)
    spec = experiment.public()
    counter = EvaluationCounter(max_evaluations)
    program = ScalarGraphProgram(graph, counter, reuse_intermediates=request.get("reuse_intermediates", True))
    parameters = {key: bound[2] for key, bound in spec["parameter_bounds"].items()}
    training = spec["training_cases"]
    output_keys = program.outputs
    fit = {"status": "not_required", "training_cases": len(training), "jacobian_rank": None}
    result = {"schema_version": EVALUATOR_VERSION, "experiment_hash": experiment.digest,
              "hypothesis_hash": graph.digest, "status": "not_assessed", "parameters": {}, "fit": fit,
              "witnesses": [], "violations": [], "check_count": 0, "replay_count": len(request["replay"]),
              "replay_passed": False, "search_rmse": None, "final_confirmation": "not_run",
              "authority": "development_evidence_only", "semantic_verdict": "not_assessed"}
    points = [c["bindings"] for c in training]
    target = np.asarray([[c["expected"][key] for key in output_keys] for c in training], dtype=float)
    try:
        if parameters:
            from .graph_parameter_fit import fit_training_parameters

            # Reserve training recheck and the worst-case batch + point fallback
            # for every verification group. Nothing is skipped to fund restarts.
            group_sizes = (len(request["replay"]), len(spec["search_cases"]), len(_probe_points(spec)))
            reserve = 1 + sum(size + 1 for size in group_sizes if size)
            parameters, fit = fit_training_parameters(
                program, points, target, spec["parameter_bounds"], reserved_evaluations=reserve)
            result["fit"] = fit
            if parameters is None:
                result["evaluations_used"] = counter.used
                result["intermediate_cache"] = program.cache_statistics()
                return result
        if training:
            prediction = program.evaluate(points, parameters)
            with np.errstate(over="raise", invalid="raise"):
                fit["train_rmse"] = _rms(prediction - target)
    except (GraphDomainError, FloatingPointError) as exc:
        fit.update(status="numeric_failure", failure_node=getattr(exc, "node_id", None))
        result["evaluations_used"] = counter.used
        result["intermediate_cache"] = program.cache_statistics()
        return result  # A failed fitting attempt is not a structural counterexample.
    result["parameters"] = parameters
    references = {case["id"]: case for case in spec["search_cases"]}
    properties = {prop["id"]: prop for prop in spec["properties"]}
    seen_witnesses = set()

    def record(point, check, reason):
        identity = {"experiment_hash": experiment.digest, "bindings": point, "check": check}
        key = fingerprint(identity)
        if key not in seen_witnesses:
            seen_witnesses.add(key)
            if len(result["witnesses"]) < 64:
                result["witnesses"].append({"id": key, **identity, "origin_hash": graph.digest})
                result["violations"].append({"witness_id": key, "reason": reason})

    def check_points(items):
        if not items:
            return []
        inputs = [item[0] for item in items]
        try:
            predictions = list(program.evaluate(inputs, parameters))
        except GraphDomainError:
            predictions = []
            for point in inputs:
                try:
                    predictions.append(program.evaluate([point], parameters)[0])
                except GraphDomainError:
                    predictions.append(None)
        for (point, checks), predicted in zip(items, predictions):
            result["check_count"] += len(checks)
            if predicted is None:
                record(point, {"kind": "finite"}, "numeric_domain_at_bound_input")
                continue
            for check in checks:
                kind = check["kind"]
                if kind == "reference":
                    expected = references[check["case_id"]]["expected"]
                    if any(abs(float(value) - expected[key]) > spec["absolute_tolerance"] +
                           spec["relative_tolerance"] * abs(expected[key]) for key, value in zip(output_keys, predicted)):
                        record(point, check, "reference_tolerance_exceeded")
                elif kind == "property":
                    prop = properties[check["property_id"]]
                    value = float(predicted[output_keys.index(prop["output"])])
                    violated = False
                    for key in ("lower", "upper"):
                        bound = prop[key]
                        if bound is not None:
                            tolerance = spec["absolute_tolerance"] + spec["relative_tolerance"] * abs(bound)
                            violated |= value < bound - tolerance if key == "lower" else value > bound + tolerance
                    if violated:
                        record(point, check, "conditional_property_violated")
        return predictions

    # Historical points are never silently dropped or replaced by new random probes.
    check_points([(w["bindings"], [w["check"]]) for w in request["replay"]])
    result["replay_passed"] = not seen_witnesses
    if not result["replay_passed"]:
        result.update(status="rejected_by_replay", evaluations_used=counter.used,
                      violation_count=len(seen_witnesses), new_checks_skipped=True,
                      intermediate_cache=program.cache_statistics())
        return result
    items = [(c["bindings"], [{"kind": "reference", "case_id": c["id"]}]) for c in spec["search_cases"]]
    predictions = check_points(items)
    if predictions and all(p is not None for p in predictions):
        expected = np.asarray([[c["expected"][k] for k in output_keys] for c in spec["search_cases"]])
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = np.asarray(predictions) - expected
        if np.isfinite(residuals).all():
            result["search_rmse"] = _rms(residuals)
            # Keep the disjoint search partition available to downstream
            # candidate comparison.  This is development evidence only: the
            # final confirmation set is still read from its separate registry.
            result["search_predictions"] = [
                [float(value) for value in np.asarray(prediction, dtype=float).reshape(-1)]
                for prediction in predictions
            ]
            result["search_expected"] = [
                [float(value) for value in row] for row in expected
            ]
    checks = [{"kind": "finite"}, *[{"kind": "property", "property_id": p} for p in properties]]
    probes = _probe_points(spec)
    check_points([(p, checks) for p in probes])
    result.update(status="rejected_by_checks" if seen_witnesses else "eligible_for_confirmation",
                  violation_count=len(seen_witnesses), witness_records_truncated=len(seen_witnesses) > 64,
                  probe_count=len(probes), evaluations_used=counter.used, new_checks_skipped=False)
    result["intermediate_cache"] = program.cache_statistics()
    return result
