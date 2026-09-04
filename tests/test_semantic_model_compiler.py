"""Safety and execution tests for the optional semantic model compiler."""

import json

import pytest

from core.mechanistic_modeling import MechanisticModelingEngine
from core.semantic_model_compiler import (
    CallableSemanticBackend,
    SemanticCompilerConfig,
    SemanticModelCompiler,
    _attach_image_parts,
)


PROBLEM = (
    "已显式给出 variables=[x,y]，coefficient_matrix=[[2,1],[1,-1]]，"
    "right_hand_side=[5,1]，units={x:1,y:1}。"
)


def _compiler(payload):
    config = SemanticCompilerConfig(
        provider="callable", model_name="embedded-test-model",
    )
    backend = CallableSemanticBackend(
        lambda messages: json.dumps(payload, ensure_ascii=False)
    )
    return SemanticModelCompiler(config, backend)


def _linear_contract(coefficient=-1):
    return {
        "id": "system",
        "kind": "linear_system",
        "variables": ["x", "y"],
        "coefficient_matrix": [[2, 1], [1, coefficient]],
        "right_hand_side": [5, 1],
        "units": {"x": "1", "y": "1"},
        "parse_status": "machine_verified",
        "source": "user_verified_ir_override",
    }


def _proposal(contract=None, quote=PROBLEM):
    return {
        "relations": [{
            "contract": contract or _linear_contract(),
            "evidence": [{
                "quote": quote,
                "supports": [
                    "variables", "coefficient_matrix", "right_hand_side", "units",
                ],
            }],
        }],
        "unresolved_questions": [],
    }


def test_semantic_multimodal_parts_use_openai_image_url_shape_without_mutation():
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "题面"}]
    result = _attach_image_parts(messages, [{"data_url": "data:image/png;base64,aGVsbG8="}])
    assert result[-1]["content"] == [
        {"type": "text", "text": "题面"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8=", "detail": "high"}},
    ]
    assert messages[-1]["content"] == "题面"


def test_grounded_model_relation_is_revalidated_and_executed():
    result = MechanisticModelingEngine(
        semantic_compiler=_compiler(_proposal()),
    ).analyze(PROBLEM)

    compilation = result["semantic_model_compilation"]
    assert compilation["status"] == "accepted"
    assert compilation["accepted_count"] == 1
    relation = compilation["accepted_relations"][0]
    assert relation["id"] == "model_system"
    assert relation["parse_status"] == "machine_verified"
    assert relation["source"] == "model_proposed_grounded_ir"
    assert relation["model_provenance"]["authority"] == "semantic_proposal_only"
    serialized = json.dumps(compilation, ensure_ascii=False)
    assert '"api_key":' not in serialized
    assert "top-secret" not in serialized
    numerical = result["numerical_results"][0]
    assert numerical["solution"] == pytest.approx({"x": 2.0, "y": 1.0})
    assert numerical["independent_audit"]["status"] == "pass"


def test_hallucinated_number_is_deferred_even_when_contract_shape_is_valid():
    result = MechanisticModelingEngine(
        semantic_compiler=_compiler(_proposal(_linear_contract(coefficient=99))),
    ).analyze(PROBLEM)

    compilation = result["semantic_model_compilation"]
    assert compilation["status"] == "no_accepted_relations"
    errors = compilation["deferred_proposals"][0]["errors"]
    assert any("ungrounded_numeric_value:coefficient_matrix.1.1:99" in item for item in errors)
    assert result["numerical_results"] == []


def test_hallucinated_numeric_literal_inside_an_expression_is_deferred():
    problem = (
        "显式 ODE：state_variables=[x]，rhs={x:-2*x}，initial_values={x:10}，"
        "parameters={}，time_variable=t，time_span=[0,1]，units={x:1,t:s}。"
    )
    contract = {
        "id": "ode", "kind": "ode_system", "state_variables": ["x"],
        "rhs": {"x": "-99*x"}, "initial_values": {"x": 10},
        "parameters": {}, "time_variable": "t", "time_span": [0, 1],
        "units": {"x": "1", "t": "s"},
    }
    payload = {
        "relations": [{
            "contract": contract,
            "evidence": [{
                "quote": problem,
                "supports": [
                    "state_variables", "rhs", "initial_values", "parameters",
                    "time_variable", "time_span", "units",
                ],
            }],
        }],
        "unresolved_questions": [],
    }
    result = MechanisticModelingEngine(
        semantic_compiler=_compiler(payload),
    ).analyze(problem)
    errors = result["semantic_model_compilation"]["deferred_proposals"][0]["errors"]
    assert any("ungrounded_numeric_value:rhs.x.__literal_0:-99" in item for item in errors)
    assert result["numerical_results"] == []


def test_non_exact_evidence_quote_cannot_ground_a_relation():
    result = MechanisticModelingEngine(
        semantic_compiler=_compiler(_proposal(quote="题面里并不存在的伪造引文")),
    ).analyze(PROBLEM)

    compilation = result["semantic_model_compilation"]
    assert compilation["accepted_count"] == 0
    assert any(
        "not_an_exact_statement_quote" in item
        for item in compilation["deferred_proposals"][0]["errors"]
    )


def test_model_failure_is_isolated_and_verified_override_still_executes():
    config = SemanticCompilerConfig(provider="callable", model_name="broken-model")
    compiler = SemanticModelCompiler(
        config, CallableSemanticBackend(lambda messages: "not-json")
    )
    result = MechanisticModelingEngine(semantic_compiler=compiler).analyze(
        "执行显式结构化关系。",
        ir_override={"relations": [_linear_contract()]},
    )

    assert result["semantic_model_compilation"]["status"] == "failed_safe"
    assert result["execution_status"] == "executed"
    assert result["numerical_results"][0]["solution"] == pytest.approx({"x": 2.0, "y": 1.0})


def test_duplicate_json_keys_are_rejected_without_execution():
    config = SemanticCompilerConfig(provider="callable", model_name="duplicate-key-model")
    compiler = SemanticModelCompiler(
        config,
        CallableSemanticBackend(lambda messages: '{"relations": [], "relations": []}'),
    )
    result = MechanisticModelingEngine(semantic_compiler=compiler).analyze(
        "分析一个尚未完整给定的线性系统。"
    )
    assert result["semantic_model_compilation"]["status"] == "failed_safe"
    assert result["semantic_model_compilation"]["error_type"] == "ValueError"
    assert result["numerical_results"] == []


def test_model_api_configuration_enforces_network_boundaries_and_hides_secret():
    with pytest.raises(ValueError, match="loopback"):
        SemanticCompilerConfig(
            provider="ollama", base_url="http://example.com:11434",
            model_name="small",
        ).validate()
    with pytest.raises(ValueError, match="HTTPS"):
        SemanticCompilerConfig(
            provider="openai_compatible", base_url="http://api.example.com/v1",
            model_name="small",
        ).validate()

    config = SemanticCompilerConfig(
        provider="callable", model_name="embedded", api_key="top-secret",
    ).validate()
    assert config.public()["api_key_configured"] is True
    assert "top-secret" not in repr(config)
    assert "top-secret" not in json.dumps(config.public())
