"""Bounded primitive-graph search, with immutable experiments and witness replay.

This development search cannot certify a model or consume a final test set.
The built-in mutator is a small, deterministic grammar baseline, not an LLM or
domain-specific solver. It can be compared against later proposal backends.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any, Mapping

from .graph_evaluator import EVALUATOR_VERSION, _validate_replay
from .cegis_repair import build_repair_directives, prioritized_operators
from .graph_experiments import (
    SearchExperiment, algebraic_equivalence_key, apply_graph_patch, diversity_audit,
    ensure_scalar_executable, fingerprint,
    interface_matches,
)
from .model_hypotheses import EvidenceLedger, HypothesisIR, HypothesisValidationError, ProblemContract, _require
from .solver_runtime import SolverLimits, SolverProcessRunner, SolverRuntimeError, failure_details


@dataclass(frozen=True)
class GraphSearchBudget:
    max_candidates: int = 8
    max_patch_attempts: int = 32
    max_evaluations: int = 2000
    per_candidate_evaluations: int = 500
    wall_seconds: float = 60.0
    reuse_intermediates: bool = True

    def __post_init__(self):
        for value, maximum in ((self.max_candidates, 16), (self.max_patch_attempts, 128),
                               (self.max_evaluations, 10000), (self.per_candidate_evaluations, 1000)):
            _require(type(value) is int and 1 <= value <= maximum, "invalid_graph_search_budget")
        _require(type(self.wall_seconds) in (int, float) and math.isfinite(self.wall_seconds)
                 and 0.05 <= self.wall_seconds <= 120, "invalid_graph_search_deadline")
        _require(type(self.reuse_intermediates) is bool, "invalid_intermediate_cache_policy")


_DIAGNOSTIC_PRIMITIVE_OPS = {
    "monotone_constraint": ("minimum", "maximum"),
    "saturating_response": ("minimum", "maximum"),
    "sinusoidal_basis": ("sin", "cos"),
    "oscillatory_state": ("sin", "cos"),
    "seasonal_lag": (),  # scalar graph baseline has no lag node yet
    "piecewise_regime": ("minimum", "maximum"),
    "event_switch": ("minimum", "maximum"),
    "threshold_constraint": ("minimum", "maximum"),
    "variance_link": ("exp", "log"),
    "multiplicative_noise": ("multiply",),
    "intercept": ("add",),
    "group_effect": ("add",),
    "exogenous_input": (),
    "delay_embedding": (),
    "latent_state": (),
    "robust_loss": (),
}


def _hinted_operators(search_hints: list[dict] | None) -> tuple[str, ...]:
    """Map server-owned hint IDs to a closed operator set; never execute text."""
    preferred: list[str] = []
    for hint in search_hints or []:
        if not isinstance(hint, Mapping):
            continue
        primitives = hint.get("candidate_primitives", ())
        if not isinstance(primitives, (list, tuple)):
            continue
        for primitive in primitives:
            preferred.extend(_DIAGNOSTIC_PRIMITIVE_OPS.get(str(primitive), ()))
    defaults = ("add", "subtract", "multiply", "divide", "minimum", "maximum")
    binary = {"add", "subtract", "multiply", "divide", "minimum", "maximum"}
    return tuple(dict.fromkeys([item for item in [*preferred, *defaults] if item in binary]))


def _hinted_unary_operators(search_hints: list[dict] | None) -> tuple[str, ...]:
    defaults = ("negate", "abs", "sqrt", "exp", "log", "sin", "cos", "observation")
    if not search_hints:
        return defaults
    preferred: list[str] = []
    for hint in search_hints or []:
        if not isinstance(hint, Mapping):
            continue
        primitives = hint.get("candidate_primitives", ())
        if not isinstance(primitives, (list, tuple)):
            continue
        for primitive in primitives:
            preferred.extend(_DIAGNOSTIC_PRIMITIVE_OPS.get(str(primitive), ()))
    unary = {"negate", "abs", "sqrt", "exp", "log", "sin", "cos", "observation"}
    return tuple(dict.fromkeys([item for item in [*preferred, *defaults] if item in unary]))


def _hinted_wrap_operators(search_hints: list[dict] | None) -> tuple[str, ...]:
    defaults = ("abs", "exp", "log", "sin", "cos", "negate")
    if not search_hints:
        return defaults
    preferred = [item for item in _hinted_unary_operators(search_hints)
                 if item in {"abs", "exp", "log", "sin", "cos", "negate"}]
    return tuple(dict.fromkeys([*preferred, *defaults]))


def primitive_patches(parent: HypothesisIR, feedback: dict, *, search_hints: list[dict] | None = None) -> list[dict]:
    """Small graph mutations; diagnostic hints affect priority, never acceptance."""
    payload = parent.payload()
    numeric_failure = any(v.get("reason") == "numeric_domain_at_bound_input" for v in feedback.get("violations", []))
    hinted = _hinted_operators(search_hints)
    repair_binary = prioritized_operators(feedback)
    defaults = ("multiply", "add", "subtract", "divide", "minimum", "maximum") if numeric_failure else (
        "add", "subtract", "multiply", "divide", "minimum", "maximum")
    operators = tuple(dict.fromkeys([*repair_binary, *hinted, *defaults]))
    repair_unary = prioritized_operators(feedback, unary=True)
    unary_operators = tuple(dict.fromkeys([*repair_unary, *_hinted_unary_operators(search_hints)]))
    wrap_operators = tuple(dict.fromkeys([*repair_unary, *_hinted_wrap_operators(search_hints)]))
    patches = []
    by_id = {n["id"]: n for n in payload["nodes"]}

    def append(replacement, added=None):
        patches.append({"parent_hash": parent.digest, "id": f"mutation_{len(patches)}",
                        "replace_nodes": [replacement], "add_nodes": added or [], "remove_node_ids": []})

    for node in reversed(payload["nodes"]):
        if node["op"] in ("variable", "parameter", "constant"):
            continue
        if node["op"] == "unknown_mechanism":
            allowed = node["attributes"]["allowed_operators"]
        else:
            allowed = (*operators, "negate", "abs", "sqrt", "exp", "log", "sin", "cos", "observation")
        inputs = node["inputs"]
        if not inputs:
            continue
        if node["op"] == "unknown_mechanism" and len(inputs) > 2:
            # Multi-input table bindings are lowered to a left-associated
            # binary fold.  Only dimensionless inputs are admitted here: a
            # unit-aware fold needs to derive every intermediate dimension and
            # must not reuse the final output type by guesswork.
            node_dimensions = [by_id[item]["type"].get("dimensions") for item in inputs]
            target_dimensions = node["type"].get("dimensions")
            if not isinstance(target_dimensions, dict) or any(not isinstance(item, dict) for item in node_dimensions):
                continue

            def dim_vector(value):
                # The public type omits zero exponents.  Keep the calculation
                # closed over the seven base dimensions used by MathType.
                return tuple(float(value.get(key, 0)) for key in ("M", "L", "T", "I", "Theta", "N", "J"))

            def dim_payload(vector):
                keys = ("M", "L", "T", "I", "Theta", "N", "J")
                return {key: int(item) if float(item).is_integer() else float(item)
                        for key, item in zip(keys, vector) if item != 0}

            target_vector = dim_vector(target_dimensions)
            input_vectors = [dim_vector(item) for item in node_dimensions]
            for op in operators:
                if op not in allowed:
                    continue
                current_vector = input_vectors[0]
                intermediate_types = []
                valid = True
                for child_vector in input_vectors[1:]:
                    if op in {"add", "subtract", "minimum", "maximum"}:
                        if current_vector != child_vector:
                            valid = False
                            break
                        next_vector = current_vector
                    elif op == "multiply":
                        next_vector = tuple(a + b for a, b in zip(current_vector, child_vector))
                    elif op == "divide":
                        next_vector = tuple(a - b for a, b in zip(current_vector, child_vector))
                    else:
                        # Other primitives are unary and cannot lower an
                        # n-ary mechanism without inventing semantics.
                        valid = False
                        break
                    intermediate_types.append(dim_payload(next_vector))
                    current_vector = next_vector
                if not valid or current_vector != target_vector:
                    continue
                added = []
                previous = inputs[0]
                for index, child_id in enumerate(inputs[1:-1], start=1):
                    key = f"fold_{node['id'][:24]}_{index}_{op}"
                    if key in by_id:
                        previous = key
                        continue
                    fold_node = {**deepcopy(node), "id": key, "op": op,
                                  "type": {**deepcopy(node["type"]),
                                          "dimensions": intermediate_types[index - 1]},
                                  "inputs": [previous, child_id], "attributes": {}}
                    added.append(fold_node)
                    previous = key
                replacement = {**deepcopy(node), "op": op,
                               "inputs": [previous, inputs[-1]], "attributes": {}}
                append(replacement, added)
                if len(patches) >= 16:
                    return patches
            continue
        for op in operators:
            if op not in allowed or (op == node["op"] and len(inputs) == 2):
                continue
            replacement = {**deepcopy(node), "op": op, "inputs": (inputs * 2)[:2], "attributes": {}}
            append(replacement)
            if len(patches) >= 16:
                return patches
        if len(inputs) == 1:
            for op in unary_operators:
                if op in allowed and op != node["op"]:
                    append({**deepcopy(node), "op": op, "attributes": {}})
                    if len(patches) >= 16:
                        return patches
        elif len(inputs) == 2 and node["op"] in operators:
            # Insert a primitive along an existing edge; this expands structure
            # while preserving all external symbols (e.g. a*x -> a*exp(x)).
            for index in sorted(range(2), key=lambda i: by_id[inputs[i]]["op"] == "parameter"):
                child = by_id[inputs[index]]
                for op in wrap_operators:
                    if op in ("exp", "log", "sin", "cos") and child["type"]["dimensions"] != {}:
                        continue
                    key = f"wrap_{node['id'][:36]}_{index}_{op}"
                    if key in by_id:
                        continue
                    added = {**deepcopy(child), "id": key, "op": op, "inputs": [child["id"]], "attributes": {}}
                    replacement = deepcopy(node)
                    replacement["inputs"][index] = key
                    append(replacement, [added])
                    if len(patches) >= 16:
                        return patches
    return patches


class GraphSearchSession:
    """A session owns budgets and all historical counterexamples; it is single-use.

    Accepts explicit bound data, not a natural-language permission to execute.
    External proposal integrations should submit JSON patches in a separate,
    bounded adapter; there is no arbitrary callback/code argument here.
    """

    def __init__(self, contract: ProblemContract, experiment: SearchExperiment, *,
                 budget: GraphSearchBudget | None = None):
        self.contract = contract
        self.experiment = SearchExperiment.from_payload(experiment.public(), contract)
        self.template = HypothesisIR.from_payload(experiment.public()["template"], contract)
        self.budget = budget or GraphSearchBudget()
        self._lock = threading.Lock()
        self._started = False
        self._outcome = None
        self._frozen = None

    def freeze(self, candidate_hash: str):
        """Select exactly once, using the private development-only outcome."""
        from .graph_confirmation import freeze_search_candidate
        with self._lock:
            _require(self._outcome is not None, "search_not_finished")
            if self._frozen is not None:
                _require(self._frozen.public()["development_evidence"]["hypothesis_hash"] == candidate_hash,
                         "candidate_selection_already_frozen")
                return self._frozen
            self._frozen = freeze_search_candidate(self.contract, self.experiment, self._outcome, candidate_hash)
            return self._frozen

    def run(self, candidates: list[dict], *, cancel: threading.Event | None = None,
            grammar_search: bool = True, supplied_patches: list[dict] | None = None,
            diagnostic_hints: list[dict] | None = None,
            model_patch_generator=None) -> dict:
        with self._lock:
            _require(not self._started, "search_session_already_used")
            self._started = True
        _require(type(candidates) is list and len(candidates) <= self.budget.max_candidates, "initial_candidate_budget")
        _require(type(grammar_search) is bool, "invalid_search_mode")
        supplied_patches = [] if supplied_patches is None else supplied_patches
        _require(type(supplied_patches) is list and len(supplied_patches) <= self.budget.max_patch_attempts,
                 "patch_budget_exceeded")
        diagnostic_hints = [] if diagnostic_hints is None else diagnostic_hints
        _require(type(diagnostic_hints) is list and len(diagnostic_hints) <= 32,
                 "diagnostic_hint_budget_exceeded")
        for hint in diagnostic_hints:
            _require(isinstance(hint, Mapping), "invalid_diagnostic_hint")
            _require(hint.get("status") == "proposal_not_executed" and
                     hint.get("hard_constraint") is False and
                     hint.get("may_feed_search") is True and
                     hint.get("requires_current_validation") is True,
                     "unsafe_diagnostic_hint_policy")
            primitives = hint.get("candidate_primitives")
            _require(type(primitives) is list and len(primitives) <= 16 and
                     all(type(item) is str and len(item) <= 80 for item in primitives),
                     "invalid_diagnostic_hint_primitives")
        if model_patch_generator is not None:
            # Only the concrete, validated adapter may enter this loop.  The
            # HTTP backend has an enforced request timeout; accepting arbitrary
            # callbacks here would defeat the search wall-clock budget.
            from .graph_patch_generator import GraphPatchGenerator
            from .semantic_model_compiler import HttpSemanticBackend
            _require(isinstance(model_patch_generator, GraphPatchGenerator),
                     "invalid_model_patch_generator")
            _require(isinstance(model_patch_generator.backend, HttpSemanticBackend),
                     "model_patch_backend_lacks_enforced_timeout")
        started = time.monotonic()
        deadline = started + self.budget.wall_seconds
        queue, graphs, visited, archive = deque(), {}, set(), {}
        reports, rejections, lineage = [], [], []
        external_patch_events = []
        external_generation_open = model_patch_generator is not None
        initial_external_calls = (
            model_patch_generator.controller.public()["calls_used"]
            if model_patch_generator is not None else 0
        )
        ledger = EvidenceLedger()
        attempts, charged, executions = 0, 0, 0
        termination = "candidate_pool_exhausted"

        def reject(exc, parent_hash=None):
            rejections.append({"code": exc.code, "node_id": exc.node_id, "parent_hash": parent_hash})

        def append_report(report):
            report["repair_directives"] = build_repair_directives(report)
            reports.append(report)

        def admit(graph):
            interface_matches(graph, self.template)
            equivalence_key = algebraic_equivalence_key(graph)
            if equivalence_key in visited:
                return False
            visited.add(equivalence_key)
            graphs[graph.digest] = graph
            queue.append(graph)
            return True

        for payload in candidates:
            try:
                admit(HypothesisIR.from_payload(payload, self.contract))
            except HypothesisValidationError as exc:
                reject(exc)

        while queue:
            if cancel is not None and cancel.is_set():
                termination = "cancelled"
                break
            remaining = deadline - time.monotonic()
            if remaining < 0.05:
                termination = "deadline_exhausted"
                break
            if executions >= self.budget.max_candidates:
                termination = "candidate_budget_exhausted"
                break
            if charged >= self.budget.max_evaluations:
                termination = "evaluation_budget_exhausted"
                break
            graph = queue.popleft()
            result = None
            try:
                ensure_scalar_executable(graph)
            except HypothesisValidationError as exc:
                reject(exc, graph.digest)
                # Typed holes can still be filled by a valid local patch.
                result = {"status": "not_executable", "hypothesis_hash": graph.digest, "violations": []}
            if result is None:
                limit = min(self.budget.per_candidate_evaluations, self.budget.max_evaluations - charged)
                previous_charge = charged
                executions += 1
                try:
                    result = SolverProcessRunner().execute("scalar_graph/v1", {
                        "problem": self.contract.public(), "hypothesis": graph.payload(),
                        "experiment": self.experiment.public(), "replay": list(archive.values()),
                        "reuse_intermediates": self.budget.reuse_intermediates,
                    }, limits=SolverLimits(wall_seconds=min(remaining, 30.0), max_evaluations=limit), cancel=cancel)
                    _require(result["hypothesis_hash"] == graph.digest and result["experiment_hash"] == self.experiment.digest,
                             "execution_scope_mismatch")
                    used = result["evaluations_used"]
                    _require(type(used) is int and 0 <= used <= limit, "invalid_evaluation_accounting")
                    charged += used
                    # Archive is append-only, even when the original model disappears.
                    _validate_replay(result["witnesses"], self.experiment)
                    _require(all(w["origin_hash"] == graph.digest for w in result["witnesses"]),
                             "counterexample_origin_mismatch")
                    additions = {w["id"]: w for w in result["witnesses"] if w["id"] not in archive}
                    if len(archive) + len(additions) > 128:
                        termination = "counterexample_budget_exhausted"
                        append_report(result)
                        break  # Never evict old witnesses to promote a new model.
                    archive.update(additions)
                    for witness in additions.values():
                        ledger = ledger.append(graph, method="counterexample", outcome="fail", scope={
                            "input_hash": fingerprint({"experiment_hash": self.experiment.digest,
                                                       "bindings": witness["bindings"], "check": witness["check"]}),
                            "evaluator_version": EVALUATOR_VERSION,
                            "domain": {"inputs": self.experiment.public()["domain"]},
                            "seed": self.experiment.public()["seed"], "experiment_hash": self.experiment.digest,
                            "witness_id": witness["id"], "final_confirmation": "not_run",
                            "semantic_verdict": "not_assessed"})
                    result["checked_witness_ids"] = list(archive) if not additions else [
                        key for key in archive if key not in additions]
                    if additions:
                        # A new witness can invalidate an earlier provisional winner.
                        # Schedule replay; if budget ends first, it is not selectable.
                        queued_hashes = {g.digest for g in queue}
                        for previous in reports:
                            if previous["status"] == "eligible_for_confirmation":
                                previous["selection_status"] = "requires_new_replay"
                                key = previous["hypothesis_hash"]
                                if key not in queued_hashes:
                                    queue.append(graphs[key])
                                    queued_hashes.add(key)
                    scope = {"input_hash": self.experiment.digest, "evaluator_version": EVALUATOR_VERSION,
                             "domain": {"inputs": self.experiment.public()["domain"]}, "seed": self.experiment.public()["seed"],
                             "checked_cases": result["check_count"], "replayed_witnesses": result["replay_count"],
                             "witness_ids": [w["id"] for w in result["witnesses"]],
                             "replay_witness_ids": result["checked_witness_ids"],
                             "final_confirmation": "not_run", "semantic_verdict": "not_assessed"}
                    ledger = ledger.append(graph, method="numerical_test",
                        outcome="pass" if result["status"] == "eligible_for_confirmation" else
                                "fail" if result["status"].startswith("rejected_") else "not_assessed", scope=scope)
                    if result.get("witness_records_truncated"):
                        termination = "counterexample_budget_exhausted"
                        append_report(result)
                        break
                except HypothesisValidationError as exc:
                    charged = previous_charge + limit
                    result = {"hypothesis_hash": graph.digest, "status": "execution_incomplete",
                              "failure_code": "invalid_response", "validation_code": exc.code,
                              **failure_details("invalid_response")}
                except SolverRuntimeError as exc:
                    # Missing child counters are charged conservatively, never a free retry.
                    charged = previous_charge + limit
                    result = {"hypothesis_hash": graph.digest, "status": "execution_incomplete", "failure_code": exc.code,
                              "execution_supervision": exc.metadata, **failure_details(exc.code)}
                    if exc.code == "cancelled":
                        termination = "cancelled"
                        append_report(result)
                        break
            # Preserve the bounded CEGIS interpretation next to the raw worker
            # result.  Directives are preferences only; they never constitute
            # an accepted repair or a mathematical verdict.
            append_report(result)
            if result["status"] in ("eligible_for_confirmation", "execution_incomplete"):
                continue
            patches = [p for p in supplied_patches if type(p) is dict and p.get("parent_hash") == graph.digest]
            if external_generation_open:
                remaining_for_model = deadline - time.monotonic()
                configured_timeout = model_patch_generator.config.timeout_seconds
                if remaining_for_model >= configured_timeout:
                    proposal = model_patch_generator.propose(
                        graph, self.contract, self.experiment, result,
                    )
                    external_patch_events.append({
                        "parent_hash": graph.digest,
                        "status": proposal["status"],
                        "error_code": proposal["error_code"],
                        "response_sha256": proposal["response_sha256"],
                        "accepted_count": len(proposal["accepted"]),
                        "rejected": proposal["rejected"],
                        "authority": "mutation_proposal_only",
                    })
                    patches += [item["patch"] for item in proposal["accepted"]]
                    if proposal["status"] == "failed_safe":
                        external_generation_open = False
                else:
                    external_patch_events.append({
                        "parent_hash": graph.digest,
                        "status": "skipped_deadline",
                        "error_code": "insufficient_time_for_bounded_model_call",
                        "response_sha256": None,
                        "accepted_count": 0,
                        "rejected": [],
                        "authority": "mutation_proposal_only",
                    })
                    external_generation_open = False
            if grammar_search:
                patches += primitive_patches(graph, result, search_hints=diagnostic_hints)
            for patch in patches:
                if attempts >= self.budget.max_patch_attempts:
                    break
                attempts += 1  # Includes invalid, duplicate and no-effect patches.
                try:
                    revised = apply_graph_patch(graph, patch, self.contract)
                    if admit(revised):
                        lineage.append({"parent_hash": graph.digest, "hypothesis_hash": revised.digest,
                                        "patch": deepcopy(patch), "trigger_status": result["status"]})
                except HypothesisValidationError as exc:
                    reject(exc, graph.digest)
        if termination == "candidate_pool_exhausted" and attempts >= self.budget.max_patch_attempts:
            termination = "patch_budget_exhausted"
        latest = {r["hypothesis_hash"]: r for r in reports}
        eligible = [r for r in latest.values() if r["status"] == "eligible_for_confirmation"
                    and set(archive) <= set(r.get("checked_witness_ids", []))]
        if termination == "counterexample_budget_exhausted":
            eligible = []  # Full witness closure was not retained; no provisional winner.
        # Development fit and expression size are separate Pareto axes. No arbitrary
        # penalty can compensate for a failed property or historical counterexample.
        def dominates(left, right):
            a = (left["search_rmse"] or 0.0, len(graphs[left["hypothesis_hash"]].payload()["nodes"]))
            b = (right["search_rmse"] or 0.0, len(graphs[right["hypothesis_hash"]].payload()["nodes"]))
            return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))
        pareto = [r["hypothesis_hash"] for r in eligible if not any(dominates(other, r) for other in eligible)]
        published_graphs = []
        for graph in graphs.values():
            view = graph.public()
            indices = [i for i, r in enumerate(reports) if r["hypothesis_hash"] == graph.digest
                       and r["status"] != "not_executable"]
            view["execution_status"] = "see_scoped_reports" if indices else "not_executed"
            view["execution_report_indices"] = indices
            published_graphs.append(view)
        result = {"schema_version": "mathmodel.graph-search/v1", "contract_hash": self.contract.digest,
                "experiment_hash": self.experiment.digest, "termination": termination,
                "status": "candidates_need_confirmation" if eligible else "no_candidate_passed",
                "final_confirmation": "not_run", "semantic_verdict": "not_assessed",
                "reports": reports, "rejections": rejections, "lineage": lineage,
                "hypotheses": published_graphs,
                "counterexamples": list(archive.values()), "pareto_candidates": pareto,
                "external_patch_events": external_patch_events,
                "diversity_audit": diversity_audit(list(graphs.values())),
                "evidence_ledger": ledger.public(), "budget": {**asdict(self.budget), "executions": executions,
                    "evaluations_charged": charged, "patch_attempts": attempts,
                    "elapsed_seconds": round(time.monotonic() - started, 6)},
                "policy": {"facts_mutable": False, "test_data_available": False, "thresholds_mutable": False,
                           "raw_code_allowed": False,
                           "external_api_calls": (
                               model_patch_generator.controller.public()["calls_used"] - initial_external_calls
                               if model_patch_generator is not None else 0
                           ),
                           "external_model_is_judge": False,
                           "finite_checks_are_proof": False, "real_world_correctness_verified": False,
                           "diagnostic_hint_count": len(diagnostic_hints)}}
        with self._lock:
            self._outcome = deepcopy(result)
        return result
