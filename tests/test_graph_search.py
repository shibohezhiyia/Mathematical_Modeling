"""Executable graph combinations must compete without changing their judge."""
from copy import deepcopy
import json
import threading
from pathlib import Path
import subprocess
import sys

import pytest

from core.graph_evaluator import ScalarGraphProgram, evaluate_graph_request
from core.graph_experiments import (
    SearchExperiment, algebraic_equivalence_key, apply_graph_patch, commutative_rewrites,
    diversity_audit, fingerprint, mechanism_diversity_key,
)
from core.graph_search import GraphSearchBudget, GraphSearchSession, primitive_patches
from core.graph_search_artifacts import run_search_bundle, search_report
from core.model_hypotheses import HypothesisIR, HypothesisValidationError, ProblemContract
from core.solver_runtime import EvaluationCounter, SolverLimits, SolverProcessRunner, SolverRuntimeError


def node(key, op, inputs=(), attrs=None, dims=None):
    return {"id": key, "op": op, "inputs": list(inputs),
            "type": {"dtype": "real", "shape": [], "dimensions": {} if dims is None else dims},
            "attributes": attrs or {}, "assumption_ids": ["mechanism"], "context_fact_ids": ["statement"]}


def fixture_graph(op="observation", parameter=False):
    contract = ProblemContract.create("研究给定区间上输入 x 与输出 y 的函数关系。")
    nodes = [node("x", "variable", attrs={"name": "x", "role": "observed"})]
    if parameter:
        nodes += [node("a", "parameter", attrs={"name": "a", "role": "parameter"})]
        inputs = ["a", "x"]
    else:
        inputs = ["x"] if op in ("observation", "negate", "abs", "sqrt", "log", "exp", "sin", "cos") else ["x", "x"]
    nodes.append(node("y", op, inputs))
    payload = {"id": "initial", "contract_hash": contract.digest, "nodes": nodes, "outputs": ["y"],
               "assumptions": [{"id": "mechanism", "text": "在给定区间检验待定函数，不预先认定结构正确。"}]}
    return contract, HypothesisIR.from_payload(payload, contract)


def cases(xs, fn, prefix):
    return [{"id": f"{prefix}_{i}", "bindings": {"x": float(x)}, "expected": {"y": float(fn(x))}}
            for i, x in enumerate(xs)]


def experiment(contract, graph, fn=lambda x: x*x, **kwargs):
    defaults = dict(domain={"x": [-3, 3]}, training_cases=cases([-2, -1, 1], fn, "train"),
                    search_cases=cases([-1.5, 0, 0.5, 2], fn, "search"), probe_count=4)
    defaults.update(kwargs)
    return SearchExperiment.create(contract, graph, **defaults)


def request(contract, graph, exp, replay=None):
    return {"problem": contract.public(), "hypothesis": graph.payload(), "experiment": exp.public(), "replay": replay or []}


def direct(contract, graph, exp, replay=None, maximum=1000):
    return evaluate_graph_request(request(contract, graph, exp, replay), max_evaluations=maximum)


def patch(parent, op, inputs=("x", "x")):
    old = deepcopy(parent.payload()["nodes"][-1])
    old.update(op=op, inputs=list(inputs), attributes={})
    return {"id": "changed", "parent_hash": parent.digest, "replace_nodes": [old], "add_nodes": [], "remove_node_ids": []}


def in_process_runner(monkeypatch):
    # Numerical unit tests isolate search logic; separate tests use real workers.
    def run(self, key, payload, *, limits, cancel=None):
        return evaluate_graph_request(payload, max_evaluations=limits.max_evaluations)
    monkeypatch.setattr(SolverProcessRunner, "execute", run)


def test_parameter_fit_never_sees_search_targets():
    contract, graph = fixture_graph("multiply", parameter=True)
    exp = experiment(contract, graph, lambda x: 2*x, parameter_bounds={"a": [0, 5, 1]})
    result = direct(contract, graph, exp)
    assert result["parameters"]["a"] == pytest.approx(2)
    assert result["status"] == "eligible_for_confirmation"
    changed = exp.public()
    changed["search_cases"][0]["expected"]["y"] = 1e6
    other = direct(contract, graph, SearchExperiment.from_payload(changed, contract))
    assert result["parameters"] == other["parameters"]
    assert other["status"] == "rejected_by_checks"
    assert result["fit"]["global_optimum_proven"] is False


def test_ambiguous_development_keeps_distinct_candidates_for_locked_holdout(monkeypatch):
    """Finite ambiguity is retained; development does not force a winner."""
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    absolute = apply_graph_patch(graph, patch(graph, "abs", inputs=("x",)), contract)
    training = cases([-1, 0, 1], lambda x: abs(x), "amb_train")
    search = cases([2], lambda x: abs(x), "amb_search")
    exp = experiment(contract, graph, lambda x: abs(x), domain={"x": [-2, 2]},
                     training_cases=training, search_cases=search)
    result = GraphSearchSession(contract, exp, budget=GraphSearchBudget(
        max_candidates=8, max_patch_attempts=1, max_evaluations=100,
        per_candidate_evaluations=50, wall_seconds=5)).run(
            [graph.payload(), absolute.payload()], grammar_search=False)
    assert len(result["pareto_candidates"]) == 2
    winners = [next(item for item in result["hypotheses"] if item["hypothesis_hash"] == key)
               for key in result["pareto_candidates"]]
    assert {next(node["op"] for node in item["nodes"] if node["id"] == "y") for item in winners} == {"observation", "abs"}
    assert result["diversity_audit"]["distinct_mechanism_count"] >= 2


def test_primitive_search_discovers_square_without_question_solver(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    exp = experiment(contract, graph)
    result = GraphSearchSession(contract, exp).run([graph.payload()])
    assert result["status"] == "candidates_need_confirmation"
    assert result["counterexamples"]
    assert result["lineage"]
    assert all("repair_directives" in report for report in result["reports"])
    initial = next(h for h in result["hypotheses"] if h["hypothesis_hash"] == graph.digest)
    assert initial["execution_status"] == "see_scoped_reports"
    assert initial["execution_report_indices"] == [0]
    winners = [r for r in result["reports"] if r["hypothesis_hash"] in result["pareto_candidates"]]
    assert winners and all(r["search_rmse"] == 0 for r in winners)
    assert all(r["replay_count"] > 0 and r["replay_passed"] for r in winners)
    assert result["final_confirmation"] == "not_run"
    assert result["policy"]["external_api_calls"] == 0
    assert all(e["scope"]["semantic_verdict"] == "not_assessed" for e in result["evidence_ledger"]["records"])


def test_diagnostic_hint_prioritizes_closed_unary_operator_without_widening_grammar():
    contract, graph = fixture_graph("multiply")
    hints = [{"status": "proposal_not_executed", "hard_constraint": False,
              "may_feed_search": True, "requires_current_validation": True,
              "candidate_primitives": ["sinusoidal_basis"]}]
    patches = primitive_patches(graph, {"violations": []}, search_hints=hints)
    assert patches
    assert patches[0]["replace_nodes"][0]["op"] in {"sin", "cos", "add", "subtract", "multiply", "divide", "minimum", "maximum"}
    assert all(patch["replace_nodes"][0]["op"] not in {"sin", "cos"} or
               len(patch["replace_nodes"][0]["inputs"]) == 1 for patch in patches)


def test_search_rejects_hint_that_tries_to_change_hard_policy():
    contract, graph = fixture_graph()
    with pytest.raises(HypothesisValidationError, match="unsafe_diagnostic_hint_policy"):
        GraphSearchSession(contract, experiment(contract, graph)).run([graph.payload()], diagnostic_hints=[{
            "status": "verified", "hard_constraint": True, "may_feed_search": True,
            "requires_current_validation": False, "candidate_primitives": ["sinusoidal_basis"],
        }])


def test_structure_search_inserts_nonlinearity_without_losing_parameter_bindings(monkeypatch):
    import math
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph("multiply", parameter=True)
    exp = experiment(contract, graph, lambda x: 2*math.exp(x), parameter_bounds={"a": [0.1, 5, 1]})
    result = GraphSearchSession(contract, exp).run([graph.payload()])
    assert result["pareto_candidates"]
    winners = [h for h in result["hypotheses"] if h["hypothesis_hash"] in result["pareto_candidates"]]
    assert any(any(n["op"] == "exp" for n in h["nodes"]) for h in winners)
    winner_hashes = {h["hypothesis_hash"] for h in winners}
    assert any(r.get("parameters", {}).get("a") == pytest.approx(2)
               for r in result["reports"] if r["hypothesis_hash"] in winner_hashes)


def test_commutative_rewrite_is_a_proof_record_not_a_new_mechanism():
    contract, graph = fixture_graph("multiply", parameter=True)
    rewrites = commutative_rewrites(graph, contract)
    assert len(rewrites) == 1
    assert rewrites[0]["rule"] == "commutativity:multiply"
    assert rewrites[0]["exact_algebraic"] is True
    assert rewrites[0]["floating_point_equivalence"] == "within_tolerance_only"
    assert algebraic_equivalence_key(graph) == rewrites[0]["algebraic_equivalence_key"]


def test_search_does_not_spend_budget_on_commutative_duplicate(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph("multiply", parameter=True)
    swapped_payload = graph.payload()
    swapped_payload["nodes"][-1]["inputs"] = ["x", "a"]
    swapped_payload["id"] = "swapped"
    swapped = HypothesisIR.from_payload(swapped_payload, contract)
    assert algebraic_equivalence_key(graph) == algebraic_equivalence_key(swapped)
    exp = experiment(contract, graph, lambda x: 2 * x, parameter_bounds={"a": [0, 5, 1]})
    result = GraphSearchSession(contract, exp).run([graph.payload(), swapped.payload()])
    assert len([r for r in result["reports"] if r["status"] != "not_executable"]) == 1


def test_mechanism_diversity_audit_flags_symbol_rename_without_rejecting_candidate():
    contract, graph = fixture_graph("multiply", parameter=True)
    payload = graph.payload()
    payload["id"] = "renamed"
    for item in payload["nodes"]:
        if item["id"] == "x":
            item["id"] = "input_renamed"
            item["attributes"]["name"] = "input_renamed"
        item["inputs"] = ["input_renamed" if value == "x" else value for value in item["inputs"]]
    renamed = HypothesisIR.from_payload(payload, contract)
    assert mechanism_diversity_key(graph) == mechanism_diversity_key(renamed)
    audit = diversity_audit([graph, renamed])
    assert audit["distinct_mechanism_count"] == 1
    assert len(audit["collision_groups"]) == 1
    assert audit["policy"] == "diagnostic_only_no_automatic_rejection"


def test_bound_counterexample_is_replayed_and_cannot_be_forgotten():
    contract, graph = fixture_graph()
    exp = experiment(contract, graph)
    initial = direct(contract, graph, exp)
    revised = apply_graph_patch(graph, patch(graph, "add"), contract)
    result = direct(contract, revised, exp, initial["witnesses"])
    assert result["status"] == "rejected_by_replay"
    assert result["new_checks_skipped"] is True
    assert result["replay_count"] == len(initial["witnesses"])
    square = apply_graph_patch(graph, patch(graph, "multiply"), contract)
    assert direct(contract, square, exp, initial["witnesses"])["status"] == "eligible_for_confirmation"


def test_domain_failure_generates_exact_finite_check_witness():
    contract, graph = fixture_graph("divide")
    exp = experiment(contract, graph, lambda x: 1)
    result = direct(contract, graph, exp)
    # Fitting is not needed; training inputs avoid zero. Search contains zero.
    assert result["status"] == "rejected_by_checks"
    assert any(w["bindings"]["x"] == 0 and w["check"] == {"kind": "finite"} for w in result["witnesses"])


def test_fitting_failure_is_not_a_structural_counterexample():
    contract, graph = fixture_graph("log")
    result = direct(contract, graph, experiment(contract, graph))
    assert result["fit"]["status"] == "numeric_failure"
    assert result["status"] == "not_assessed"
    assert result["witnesses"] == []


@pytest.mark.parametrize("op, function", [
    ("abs", lambda x: abs(x)),
    ("sin", lambda x: __import__("math").sin(x)),
    ("cos", lambda x: __import__("math").cos(x)),
])
def test_new_dimension_safe_scalar_primitives_execute(op, function):
    contract, graph = fixture_graph(op)
    result = direct(contract, graph, experiment(contract, graph, function))
    assert result["status"] == "eligible_for_confirmation"
    assert result["search_rmse"] == pytest.approx(0.0)
    assert len(result["search_predictions"]) == len(result["search_expected"])
    assert result["search_predictions"] == result["search_expected"]


@pytest.mark.parametrize("op, function", [
    ("minimum", lambda x: min(x, 0.25)),
    ("maximum", lambda x: max(x, -0.25)),
])
def test_bounded_binary_primitives_execute(op, function):
    contract, graph = fixture_graph(op)
    payload = graph.payload()
    payload["nodes"].append(node("limit", "constant", attrs={"value": 0.25 if op == "minimum" else -0.25}))
    payload["nodes"][-2]["inputs"] = ["x", "limit"]
    graph = HypothesisIR.from_payload(payload, contract)
    result = direct(contract, graph, experiment(contract, graph, function))
    assert result["status"] == "eligible_for_confirmation"
    assert result["search_rmse"] == pytest.approx(0.0)


def test_no_data_conditional_property_search():
    contract, graph = fixture_graph("negate")
    exp = experiment(contract, graph, training_cases=[], search_cases=[], properties=[
        {"id": "nonnegative", "output": "y", "lower": 0, "upper": None,
         "source": {"kind": "assumption", "id": "mechanism"}},
    ])
    result = direct(contract, graph, exp)
    assert result["status"] == "rejected_by_checks"
    assert result["search_rmse"] is None
    assert any(w["check"]["kind"] == "property" for w in result["witnesses"])
    assert result["semantic_verdict"] == "not_assessed"


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(final_test=[]), "unexpected_fields"),
    (lambda p: p.update(contract_hash="0"*64), "contract_hash_mismatch"),
    (lambda p: p.update(absolute_tolerance=-1), "invalid_check_tolerance"),
    (lambda p: p.update(seed=True), "invalid_experiment_seed"),
    (lambda p: p.update(probe_count=10000), "invalid_probe_budget"),
    (lambda p: p.update(domain={"x": [3, -3]}), "invalid_domain"),
    (lambda p: p["search_cases"][0].update(bindings={"x": -2}), "training_search_overlap"),
    (lambda p: p["search_cases"][0].update(bindings={"x": 100}), "point_outside_domain"),
    (lambda p: p["search_cases"][0].update(expected={"y": float("nan")}), "nonfinite_or_large_number"),
])
def test_experiment_rejects_bad_or_leaking_inputs(mutation, error):
    contract, graph = fixture_graph()
    payload = experiment(contract, graph).public()
    mutation(payload)
    with pytest.raises(HypothesisValidationError, match=error):
        SearchExperiment.from_payload(payload, contract)


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(parent_hash="0"*64), "stale_patch_parent"),
    (lambda p: p.update(absolute_tolerance=1e9), "unexpected_fields"),
    (lambda p: p.update(assumptions=[]), "unexpected_fields"),
    (lambda p: p["replace_nodes"][0].update(inputs=["y", "x"]), "expression_dependency_cycle"),
    (lambda p: p["replace_nodes"][0].update(op="eval"), "unsupported_operator"),
    (lambda p: p.update(remove_node_ids=["x"]), "protected_symbol_bindings"),
    (lambda p: p["replace_nodes"][0]["type"].update(dimensions={"T": 1}), "protected_node_type"),
    (lambda p: p["replace_nodes"].append(deepcopy(p["replace_nodes"][0])), "duplicate_patch_target"),
])
def test_graph_patch_cannot_modify_the_judge(mutation, error):
    contract, graph = fixture_graph()
    value = patch(graph, "multiply")
    mutation(value)
    with pytest.raises(HypothesisValidationError, match=error):
        apply_graph_patch(graph, value, contract)


def test_replay_cannot_change_expected_values_or_scope():
    contract, graph = fixture_graph()
    exp = experiment(contract, graph)
    witnesses = direct(contract, graph, exp)["witnesses"]
    old = deepcopy(witnesses)
    witnesses[0]["experiment_hash"] = "0"*64
    with pytest.raises(HypothesisValidationError, match="stale_counterexample_scope"):
        direct(contract, graph, exp, witnesses)
    old[0]["bindings"]["x"] = 0.1
    with pytest.raises(HypothesisValidationError, match="counterexample_reference_mismatch"):
        direct(contract, graph, exp, old)


@pytest.mark.parametrize("field,error", [("output", "unknown_property_output"), ("source", "unknown_property_source")])
def test_malformed_property_references_fail_with_structured_codes(field, error):
    contract, graph = fixture_graph()
    prop = {"id": "bound", "output": "y", "lower": 0, "upper": None,
            "source": {"kind": "assumption", "id": "mechanism"}}
    if field == "output":
        prop["output"] = []
    else:
        prop["source"]["id"] = []
    with pytest.raises(HypothesisValidationError, match=error):
        experiment(contract, graph, properties=[prop])


def test_non_list_empty_inputs_are_not_silently_replaced():
    contract, graph = fixture_graph()
    with pytest.raises(HypothesisValidationError, match="case_budget_exceeded"):
        experiment(contract, graph, training_cases=0)


def test_malformed_replay_reference_is_not_an_unhandled_type_error():
    contract, graph = fixture_graph()
    exp = experiment(contract, graph)
    replay = direct(contract, graph, exp)["witnesses"]
    replay[0]["check"]["case_id"] = []
    with pytest.raises(HypothesisValidationError, match="counterexample_reference_mismatch"):
        direct(contract, graph, exp, replay)


def test_plain_json_contract_and_patch_are_immutable_snapshots():
    contract, graph = fixture_graph()
    exp = experiment(contract, graph)
    original = exp.digest
    exp.public()["absolute_tolerance"] = 10000
    graph.payload()["nodes"].clear()
    assert exp.digest == original
    assert len(graph.payload()["nodes"]) == 2


def test_overlap_normalizes_negative_zero():
    contract, graph = fixture_graph()
    with pytest.raises(HypothesisValidationError, match="training_search_overlap"):
        experiment(contract, graph, training_cases=cases([0.0], lambda x: x, "train"),
                   search_cases=cases([-0.0], lambda x: x, "search"))


def test_unknown_mechanism_is_filled_without_widening_its_operator_space(monkeypatch):
    in_process_runner(monkeypatch)
    contract, original = fixture_graph()
    payload = original.payload()
    payload["nodes"][-1].update(op="unknown_mechanism", attributes={"allowed_operators": ["multiply"], "properties": []})
    graph = HypothesisIR.from_payload(payload, contract)
    exp = experiment(contract, graph)
    with pytest.raises(HypothesisValidationError, match="mechanism_search_space_violation"):
        apply_graph_patch(graph, patch(graph, "add"), contract)
    result = GraphSearchSession(contract, exp).run([graph.payload()])
    assert result["status"] == "candidates_need_confirmation"
    assert result["budget"]["executions"] == 1  # Typed hole itself never runs.
    assert len(result["lineage"]) == 1


def test_invalid_type_mutation_consumes_patch_budget_not_numeric_budget(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    invalid = patch(graph, "multiply")
    invalid["replace_nodes"][0]["inputs"] = ["missing", "x"]
    result = GraphSearchSession(contract, experiment(contract, graph)).run(
        [graph.payload()], supplied_patches=[invalid], grammar_search=False)
    assert result["budget"]["executions"] == 1
    assert result["budget"]["patch_attempts"] == 1
    assert any(r["code"] == "unknown_input_node" for r in result["rejections"])


def test_earlier_winner_is_replayed_after_new_witnesses(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    exp = experiment(contract, graph, lambda x: x)
    other = apply_graph_patch(graph, patch(graph, "multiply"), contract)
    result = GraphSearchSession(contract, exp, budget=GraphSearchBudget(max_candidates=3)).run(
        [graph.payload(), other.payload()], grammar_search=False)
    assert result["pareto_candidates"] == [graph.digest]
    assert result["reports"][-1]["hypothesis_hash"] == graph.digest
    assert result["reports"][-1]["replay_count"] == len(result["counterexamples"])


def test_initial_candidate_cannot_change_assumptions_or_symbols(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    exp = experiment(contract, graph)
    modified = graph.payload()
    modified["assumptions"][0]["text"] = "可以随意放宽问题"
    result = GraphSearchSession(contract, exp).run([modified])
    assert result["budget"]["executions"] == 0
    assert result["rejections"][0]["code"] == "protected_assumptions"


@pytest.mark.parametrize("op,value,expected", [("exp", 1, 2.718281828459045), ("log", 2, 0.6931471805599453),
                                                ("negate", 2, -2), ("subtract", 2, 0), ("divide", 2, 1)])
def test_interpreter_primitive_values(op, value, expected):
    _, graph = fixture_graph(op)
    program = ScalarGraphProgram(graph, EvaluationCounter(10))
    assert program.evaluate([{"x": value}], {})[0, 0] == pytest.approx(expected)


def test_negative_quantity_unit_and_multi_node_patch():
    contract, original = fixture_graph()
    payload = original.payload()
    payload["nodes"][0]["type"]["dimensions"] = {"L": 1}
    payload["nodes"][1]["type"]["dimensions"] = {"L": 1}
    graph = HypothesisIR.from_payload(payload, contract)
    plan = patch(graph, "multiply")
    with pytest.raises(HypothesisValidationError, match="output_dimension_mismatch"):
        apply_graph_patch(graph, plan, contract)
    plan = patch(graph, "add", ["x", "offset"])
    plan["add_nodes"] = [node("offset", "constant", attrs={"value": -1}, dims={"L": 1})]
    revised = apply_graph_patch(graph, plan, contract)
    assert ScalarGraphProgram(revised, EvaluationCounter(10)).evaluate([{"x": 2}], {})[0, 0] == 1


def test_resource_failures_do_not_become_mathematical_counterexamples(monkeypatch):
    def fail(*args, **kwargs):
        raise SolverRuntimeError("memory_limit")
    monkeypatch.setattr(SolverProcessRunner, "execute", fail)
    contract, graph = fixture_graph()
    result = GraphSearchSession(contract, experiment(contract, graph), budget=GraphSearchBudget(max_evaluations=10)).run([graph.payload()])
    assert result["status"] == "no_candidate_passed"
    assert not result["counterexamples"] and not result["pareto_candidates"]
    assert result["budget"]["evaluations_charged"] == 10


def test_counterexample_archive_overflow_cannot_promote_candidates(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    exp = experiment(contract, graph, training_cases=[],
                     search_cases=cases([0.1 + i/100 for i in range(100)], lambda x: x*x, "search"))
    result = GraphSearchSession(contract, exp).run([graph.payload()])
    assert result["termination"] == "counterexample_budget_exhausted"
    assert result["pareto_candidates"] == []
    assert len(result["counterexamples"]) == 64
    assert result["reports"][0]["witness_records_truncated"]


def test_forged_worker_result_does_not_cross_scope(monkeypatch):
    def wrong_scope(*args, **kwargs):
        return {"hypothesis_hash": "0"*64, "experiment_hash": "0"*64, "status": "eligible_for_confirmation"}
    monkeypatch.setattr(SolverProcessRunner, "execute", wrong_scope)
    contract, graph = fixture_graph()
    result = GraphSearchSession(contract, experiment(contract, graph)).run([graph.payload()])
    assert result["reports"][0]["failure_code"] == "invalid_response"
    assert result["pareto_candidates"] == [] and not result["counterexamples"]


def test_no_unchecked_promotion_when_candidate_budget_exhausted(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    result = GraphSearchSession(contract, experiment(contract, graph), budget=GraphSearchBudget(max_candidates=1)).run([graph.payload()])
    assert result["termination"] == "candidate_budget_exhausted"
    assert result["pareto_candidates"] == []


def test_pre_cancel_and_single_use_session():
    contract, graph = fixture_graph()
    session = GraphSearchSession(contract, experiment(contract, graph))
    cancel = threading.Event()
    cancel.set()
    assert session.run([graph.payload()], cancel=cancel)["termination"] == "cancelled"
    with pytest.raises(HypothesisValidationError, match="search_session_already_used"):
        session.run([graph.payload()])


def test_new_witness_disqualifies_earlier_winner_if_replay_budget_ends(monkeypatch):
    in_process_runner(monkeypatch)
    contract, graph = fixture_graph()
    exp = experiment(contract, graph, lambda x: x)
    other = apply_graph_patch(graph, patch(graph, "multiply"), contract)
    result = GraphSearchSession(contract, exp, budget=GraphSearchBudget(max_candidates=2)).run(
        [graph.payload(), other.payload()], grammar_search=False)
    assert result["reports"][0]["status"] == "eligible_for_confirmation"
    assert result["reports"][0]["selection_status"] == "requires_new_replay"
    assert result["pareto_candidates"] == []
    assert result["termination"] == "candidate_budget_exhausted"


def test_real_worker_runs_graph_under_resource_supervision():
    contract, graph = fixture_graph("multiply", parameter=True)
    exp = experiment(contract, graph, lambda x: 2*x, parameter_bounds={"a": [0, 5, 1]})
    result = SolverProcessRunner().execute("scalar_graph/v1", request(contract, graph, exp))
    assert result["status"] == "eligible_for_confirmation"
    assert result["parameters"]["a"] == pytest.approx(2)
    assert result["execution_supervision"]["process_isolated"] is True
    assert result["execution_supervision"]["permission_isolated"] is True


def test_worker_rechecks_forged_contract_and_evaluation_budget():
    contract, graph = fixture_graph()
    exp = experiment(contract, graph)
    invalid = request(contract, graph, exp)
    invalid["hypothesis"]["nodes"][-1]["op"] = "exec"
    with pytest.raises(SolverRuntimeError, match="invalid_contract"):
        SolverProcessRunner().execute("scalar_graph/v1", invalid)
    with pytest.raises(SolverRuntimeError, match="evaluation_limit"):
        SolverProcessRunner().execute("scalar_graph/v1", request(contract, graph, exp), limits=SolverLimits(max_evaluations=1))


def test_intermediate_cache_never_reuses_parameter_dependent_nodes():
    _, graph = fixture_graph("multiply", parameter=True)
    program = ScalarGraphProgram(graph, EvaluationCounter(10))
    assert program.evaluate([{"x": 2}], {"a": 3})[0, 0] == 6
    assert program.evaluate([{"x": 2}], {"a": 4})[0, 0] == 8
    assert program.cache_statistics()["nodes_reused"] == 1  # x only, not a or a*x.
    assert program.evaluate([{"x": 3}], {"a": 4})[0, 0] == 12
    assert program.counter.used == 3  # Reuse never removes an evaluation from the budget.


def test_intermediate_cache_is_bounded_and_does_not_alias_outputs():
    _, graph = fixture_graph()
    program = ScalarGraphProgram(graph, EvaluationCounter(100))
    output = program.evaluate([{"x": 1}], {})
    output[:] = 99
    assert program.evaluate([{"x": 1}], {})[0, 0] == 1
    for index in range(10):
        program.evaluate([{"x": index}], {})
    stats = program.cache_statistics()
    assert stats["entries"] == 4
    assert stats["array_bytes"] <= stats["max_array_bytes"]


@pytest.mark.parametrize("parameter", [True, False])
def test_cache_on_off_preserves_all_results_and_check_counts(parameter):
    contract, graph = fixture_graph("multiply", parameter=parameter)
    exp = experiment(contract, graph, (lambda x: 2*x) if parameter else (lambda x: x*x),
                     parameter_bounds={"a": [0, 5, 1]} if parameter else {})
    payload = request(contract, graph, exp)
    cached = evaluate_graph_request(payload, max_evaluations=1000)
    uncached = evaluate_graph_request({**payload, "reuse_intermediates": False}, max_evaluations=1000)
    cached_stats, uncached_stats = cached.pop("intermediate_cache"), uncached.pop("intermediate_cache")
    assert cached == uncached
    assert not uncached_stats["enabled"]
    if parameter:
        assert cached_stats["nodes_reused"] > 0
        assert cached_stats["nodes_computed"] < uncached_stats["nodes_computed"]


def test_run_bundle_uses_unique_manifests_and_scoped_artifacts(monkeypatch, tmp_path):
    in_process_runner(monkeypatch)
    example = Path(__file__).resolve().parents[1] / "examples" / "graph_search_scalar.json"
    bundle = json.loads(example.read_text(encoding="utf-8"))
    result, first = run_search_bundle(bundle, output_root=tmp_path)
    _, second = run_search_bundle(bundle, output_root=tmp_path)
    assert first != second
    manifest = json.loads((first / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["artifacts"]) == 4
    assert (first / "evidence" / "counterexamples.json").is_file()
    assert (first / "reports" / "graph_search.md").is_file()
    assert result["semantic_verdict"] == "not_assessed"


def test_report_escapes_untrusted_status():
    value = {"contract_hash": "0"*64, "experiment_hash": "0"*64, "termination": "cancelled",
             "budget": {"executions": 1, "evaluations_charged": 2}, "counterexamples": [], "pareto_candidates": [],
             "reports": [{"hypothesis_hash": "0"*64, "status": "<script>bad</script>|injection\n"}]}
    report = search_report(value)
    assert "<script>" not in report
    assert "&lt;script&gt;" in report and "&#124;" in report


def test_cli_real_worker_output_and_exit_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / "run_graph_search.py"),
                             str(root / "examples" / "graph_search_scalar.json"), "--output-root", str(tmp_path)],
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "最终独立确认未执行" in result.stdout.decode("utf-8")
    assert len(list(tmp_path.glob("*/artifact_manifest.json"))) == 1


def test_cli_rejects_invalid_payload_without_echoing_contents(tmp_path):
    from scripts.run_graph_search import main
    # This is ordinary test-fixture data, never executed as code.
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"secret": "private_test_value"}', encoding="utf-8")
    assert main([str(invalid), "--output-root", str(tmp_path / "unused")]) == 2
    assert not (tmp_path / "unused").exists()
