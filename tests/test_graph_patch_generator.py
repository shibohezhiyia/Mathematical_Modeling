"""A model may propose graph mutations, but cannot change the experiment."""
from copy import deepcopy
import json

import pytest

from core.graph_experiments import SearchExperiment, apply_graph_patch
from core.graph_patch_generator import GraphPatchGenerator, minimal_patch_feedback
from core.graph_search import GraphSearchBudget, GraphSearchSession
from core.model_hypotheses import (
    HypothesisIR, HypothesisValidationError, ProblemContract, SearchController,
)
from core.semantic_model_compiler import (
    CallableSemanticBackend, HttpSemanticBackend, SemanticCompilerConfig,
)


def _node(key, op, inputs=(), attrs=None):
    return {
        "id": key, "op": op, "inputs": list(inputs),
        "type": {"dtype": "real", "shape": [], "dimensions": {}},
        "attributes": attrs or {}, "assumption_ids": ["mechanism"],
        "context_fact_ids": ["statement"],
    }


def _fixture():
    contract = ProblemContract.create("研究区间上 x 与 y 的关系。")
    payload = {
        "id": "initial", "contract_hash": contract.digest,
        "nodes": [
            _node("x", "variable", attrs={"name": "x", "role": "observed"}),
            _node("y", "observation", ["x"]),
        ],
        "outputs": ["y"],
        "assumptions": [{"id": "mechanism", "text": "函数结构尚待检验。"}],
    }
    graph = HypothesisIR.from_payload(payload, contract)
    experiment = SearchExperiment.create(
        contract, graph, domain={"x": [-2, 2]},
        training_cases=[],
        search_cases=[{"id": "check", "bindings": {"x": 2.0},
                       "expected": {"y": 4.0}}],
        probe_count=0,
    )
    return contract, graph, experiment


def _patch(parent):
    output = deepcopy(parent.payload()["nodes"][-1])
    output.update(op="multiply", inputs=["x", "x"], attributes={})
    return {
        "parent_hash": parent.digest, "id": "square_mutation",
        "replace_nodes": [output], "add_nodes": [], "remove_node_ids": [],
    }


def _generator(response, *, controller=None, api_key="test-secret"):
    backend = CallableSemanticBackend(lambda _messages: response)
    return GraphPatchGenerator(
        SemanticCompilerConfig(provider="callable", model_name="fixture", api_key=api_key),
        backend, controller=controller,
    )


def test_model_patch_is_validated_and_can_enter_existing_graph_search_contract():
    contract, graph, experiment = _fixture()
    patch = _patch(graph)
    generator = _generator(json.dumps({"patches": [patch]}))
    result = generator.propose(graph, contract, experiment, {
        "status": "rejected_by_checks",
        "violations": [{"reason": "reference_mismatch", "witness_id": "w1"}],
        "witnesses": [{"bindings": {"x": 2.0}, "check": {"kind": "reference"}}],
        "traceback": "must-never-be-forwarded",
    })
    assert result["status"] == "proposed"
    assert result["accepted"][0]["patch"] == patch
    assert result["policy"]["may_modify_judge"] is False
    candidate = apply_graph_patch(graph, result["accepted"][0]["patch"], contract)
    assert candidate.digest == result["accepted"][0]["candidate_hash"]
    prompt = json.loads(generator._prompt(graph, experiment, {"traceback": "private"})[1]["content"])
    assert "traceback" not in prompt["diagnostic"]
    assert "private" not in json.dumps(prompt)
    assert "test-secret" not in json.dumps(prompt)


def test_model_patch_cannot_modify_symbols_or_smuggle_judge_fields():
    contract, graph, experiment = _fixture()
    symbol = deepcopy(graph.payload()["nodes"][0])
    symbol["attributes"]["name"] = "other"
    protected_patch = {
        "parent_hash": graph.digest, "id": "rename_symbol",
        "replace_nodes": [symbol], "add_nodes": [], "remove_node_ids": [],
    }
    result = _generator(json.dumps({"patches": [protected_patch]})).propose(
        graph, contract, experiment, {"status": "rejected_by_checks"},
    )
    assert result["status"] == "no_valid_patches"
    assert result["rejected"][0]["code"] == "protected_symbol_bindings"

    injected = _patch(graph)
    injected["new_tolerance"] = 1e9
    other = _generator(json.dumps({"patches": [injected]})).propose(
        graph, contract, experiment, {"status": "rejected_by_checks"},
    )
    assert other["status"] == "no_valid_patches"
    assert other["rejected"][0]["code"] == "unexpected_fields"


def test_patch_call_budget_is_consumed_and_provider_errors_are_redacted():
    contract, graph, experiment = _fixture()
    controller = SearchController(max_calls=1, max_candidates=2)
    generator = _generator(json.dumps({"patches": []}), controller=controller)
    first = generator.propose(graph, contract, experiment, {"status": "not_assessed"})
    second = generator.propose(graph, contract, experiment, {"status": "not_assessed"})
    assert first["status"] == "no_valid_patches"
    assert second["status"] == "failed_safe"
    assert second["error_code"] == "model_call_budget_exhausted"

    failing = GraphPatchGenerator(
        SemanticCompilerConfig(provider="callable", model_name="fixture"),
        CallableSemanticBackend(lambda _messages: (_ for _ in ()).throw(
            RuntimeError("private provider trace")
        )),
    )
    failure = failing.propose(graph, contract, experiment, {"status": "not_assessed"})
    assert failure["error_code"] == "patch_backend_failed"
    assert "private provider trace" not in repr(failure)


def test_feedback_rejects_nonfinite_counterexample_values():
    with pytest.raises(HypothesisValidationError, match="invalid_patch_feedback_bindings"):
        minimal_patch_feedback({"witnesses": [{
            "bindings": {"x": float("nan")}, "check": {"kind": "finite"},
        }]})


class _FixtureHttpBackend(HttpSemanticBackend):
    """Test double preserving the production backend's timeout contract."""

    def __init__(self, config):
        super().__init__(config)
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        request = json.loads(messages[1]["content"])
        parent = request["typed_graph"]
        output = deepcopy(parent["nodes"][-1])
        output.update(op="multiply", inputs=["x", "x"], attributes={})
        return json.dumps({"patches": [{
            "parent_hash": request["parent_hash"], "id": "square_from_model",
            "replace_nodes": [output], "add_nodes": [], "remove_node_ids": [],
        }]})


def test_bounded_http_mutator_closes_the_cegis_loop_without_becoming_judge(monkeypatch):
    from core.graph_evaluator import evaluate_graph_request
    from core.solver_runtime import SolverProcessRunner

    def in_process(self, key, payload, *, limits, cancel=None):
        return evaluate_graph_request(payload, max_evaluations=limits.max_evaluations)

    monkeypatch.setattr(SolverProcessRunner, "execute", in_process)
    contract, graph, experiment = _fixture()
    config = SemanticCompilerConfig(
        provider="ollama", base_url="http://localhost:11434",
        model_name="fixture", timeout_seconds=5,
    )
    backend = _FixtureHttpBackend(config)
    generator = GraphPatchGenerator(config, backend)
    result = GraphSearchSession(
        contract, experiment,
        budget=GraphSearchBudget(max_candidates=3, max_patch_attempts=4,
                                 max_evaluations=100, per_candidate_evaluations=30,
                                 wall_seconds=10),
    ).run([graph.payload()], grammar_search=False, model_patch_generator=generator)
    assert result["status"] == "candidates_need_confirmation"
    assert result["pareto_candidates"]
    assert result["policy"]["external_api_calls"] == 1
    assert result["policy"]["external_model_is_judge"] is False
    assert result["external_patch_events"][0]["accepted_count"] == 1
    assert backend.calls == 1


def test_search_refuses_unbounded_callable_mutator():
    contract, graph, experiment = _fixture()
    generator = _generator(json.dumps({"patches": []}))
    with pytest.raises(HypothesisValidationError,
                       match="model_patch_backend_lacks_enforced_timeout"):
        GraphSearchSession(contract, experiment).run(
            [graph.payload()], model_patch_generator=generator,
        )
