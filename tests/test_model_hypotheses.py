"""Proposals can invent structures, but cannot grant themselves authority."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json

import pytest

from core.hypothesis_generator import HypothesisGenerator, proposal_schema
from core.model_hypotheses import (
    EvidenceLedger, HypothesisIR, HypothesisValidationError, MathType,
    ProblemContract, SearchController, SemanticHypothesisSet, decode_proposal,
)
from core.semantic_model_compiler import CallableSemanticBackend, SemanticCompilerConfig


def math_type(dimensions=None, shape=None, dtype="real"):
    return {"dtype": dtype, "shape": shape or [], "dimensions": dimensions}


def node(identifier, op, inputs, dimensions, attributes=None, shape=None, dtype="real"):
    return {"id": identifier, "op": op, "inputs": inputs, "type": math_type(dimensions, shape, dtype),
            "attributes": attributes or {}, "assumption_ids": ["a1"], "context_fact_ids": ["statement"]}


def candidate(contract):
    return {"id": "h1", "contract_hash": contract.digest,
            "assumptions": [{"id": "a1", "text": "候选状态变化率可能正比于状态，尚未验证。"}],
            "nodes": [
                node("x", "variable", [], {"L": 1}, {"name": "x", "role": "state"}),
                node("k", "parameter", [], {"T": -1}, {"name": "k", "role": "parameter"}),
                node("rate", "multiply", ["x", "k"], {"L": 1, "T": -1}),
            ], "outputs": ["rate"]}


def generator(completion, api_key=""):
    return HypothesisGenerator(SemanticCompilerConfig(provider="callable", model_name="fake", api_key=api_key),
                               CallableSemanticBackend(completion))


def respond(messages):
    contract = ProblemContract.create("研究物体运动，但没有给出运动方程。")
    assert json.loads(messages[-1]["content"])["contract_hash"] == contract.digest
    return json.dumps({"hypotheses": [candidate(contract)], "questions": []}, ensure_ascii=False)


def test_exact_fact_spans_and_deep_immutability():
    facts = [{"id": "cap", "text": "容量100", "start": 0, "end": 5}]
    contract = ProblemContract.create("容量100，目标待定", facts=facts, hard_constraint_ids=("cap",))
    original_hash = contract.digest
    facts[0]["text"] = "伪造"
    contract.public()["facts"][0]["text"] = "伪造"
    assert contract.digest == original_hash
    assert contract.public()["facts"][0]["text"] == "容量100"
    with pytest.raises(FrozenInstanceError):
        contract._json = "{}"
    with pytest.raises(HypothesisValidationError, match="fact_quote_mismatch"):
        ProblemContract.create("容量100", facts=facts)


def test_new_structure_is_proposed_without_inventing_statement_facts():
    contract = ProblemContract.create("研究物体运动，但没有给出运动方程。")
    result = generator(respond).propose(contract)
    assert result["status"] == "proposed"
    assert result["policy"]["may_modify_facts"] is False
    assert result["policy"]["may_execute"] is False
    assert result["hypotheses"][0]["execution_status"] == "not_executed"
    assert result["hypotheses"][0]["authority"] == "hypothesis_only"
    assert result["hypotheses"][0]["validation_status"] == "type_checked"
    assert result["evidence_ledger"]["records"][0]["method"] == "type_check"
    assert contract.public()["facts"][0]["text"] == "研究物体运动，但没有给出运动方程。"


def test_revision_cannot_silently_drop_or_rebind_hard_constraints():
    contract = ProblemContract.create("容量100，目标待定", facts=[
        {"id": "cap", "text": "容量100", "start": 0, "end": 5},
    ], hard_constraint_ids=("cap",))
    revised = contract.revise("容量100，目标为最小成本")
    assert revised.public()["hard_constraint_ids"] == ["cap"]
    with pytest.raises(HypothesisValidationError, match="fact_quote_mismatch"):
        contract.revise("容量200，目标为最小成本")


def test_unknown_mechanism_is_a_typed_hole_not_a_failure_or_executable():
    contract = ProblemContract.create("研究变化规律")
    payload = candidate(contract)
    payload["nodes"] = [payload["nodes"][0], node("unknown", "unknown_mechanism", ["x"], {"L": 1, "T": -1},
                        {"allowed_operators": ["multiply", "divide"], "properties": ["可能趋于饱和，待检验"]})]
    payload["outputs"] = ["unknown"]
    result = HypothesisIR.from_payload(payload, contract).public()
    assert result["validation_status"] == "needs_bindings"
    assert result["unknown_mechanisms"][0]["status"] == "unfilled"
    assert result["unknown_mechanisms"][0]["physical_meaning_verified"] is False
    assert "mechanism_unfilled:unknown" in result["obligations"]


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(contract_hash="0" * 64), "contract_hash_mismatch"),
    (lambda p: p.update(verified=True), "unexpected_fields"),
    (lambda p: p.update(hard_constraints=[]), "unexpected_fields"),
    (lambda p: p["nodes"][0].update(source="statement_fact"), "unexpected_fields"),
    (lambda p: p["nodes"][0].update(context_fact_ids=["invented"]), "unknown_provenance_reference"),
    (lambda p: p["nodes"][0].update(assumption_ids=[]), "node_assumption_required"),
    (lambda p: p["nodes"][2].update(op="exec"), "unsupported_operator"),
    (lambda p: p["nodes"][2].update(inputs=["x", "missing"]), "unknown_input_node"),
    (lambda p: p["nodes"][2].update(inputs=["rate", "k"]), "expression_dependency_cycle"),
    (lambda p: p["nodes"][2].update(inputs=["x"]), "operator_arity_mismatch"),
    (lambda p: p["nodes"][2].update(type=math_type({"L": 1})), "output_dimension_mismatch"),
    (lambda p: p["nodes"][2].update(op="add"), "dimension_mismatch"),
    (lambda p: p["nodes"][2].update(type=math_type({"L": 1, "T": -1}, [3])), "output_shape_mismatch"),
    (lambda p: p.update(outputs=["x"]), "unreachable_nodes"),
])
def test_illegal_or_self_authorizing_graphs_are_rejected(mutation, error):
    contract = ProblemContract.create("研究变化规律")
    payload = candidate(contract)
    mutation(payload)
    with pytest.raises(HypothesisValidationError, match=error):
        HypothesisIR.from_payload(payload, contract)


def test_unknown_units_remain_unresolved():
    contract = ProblemContract.create("单位未知")
    payload = candidate(contract)
    payload["nodes"][0]["type"]["dimensions"] = None
    graph = HypothesisIR.from_payload(payload, contract)
    assert graph.public()["validation_status"] == "needs_bindings"
    assert "units_unresolved:x" in graph.public()["obligations"]


def test_variable_descriptor_binds_domain_time_observability_and_source():
    contract = ProblemContract.create("观测状态随时间变化")
    payload = candidate(contract)
    payload["nodes"][0]["attributes"]["descriptor"] = {
        "value_domain": {"kind": "interval", "lower": 0, "upper": 100,
                          "closed_lower": True, "closed_upper": True},
        "time_semantics": "ordered",
        "observability": "observed",
        "source": {"kind": "field", "id": "state_column"},
    }
    graph = HypothesisIR.from_payload(payload, contract)
    descriptor = graph.public()["nodes"][0]["attributes"]["descriptor"]
    assert descriptor["value_domain"]["kind"] == "interval"
    assert descriptor["time_semantics"] == "ordered"
    assert descriptor["source"]["kind"] == "field"


def test_variable_descriptor_rejects_observability_role_mismatch():
    contract = ProblemContract.create("存在隐状态")
    payload = candidate(contract)
    payload["nodes"][0]["attributes"]["role"] = "latent_state"
    payload["nodes"][0]["attributes"]["descriptor"] = {
        "value_domain": {"kind": "nonnegative"},
        "time_semantics": "instant",
        "observability": "observed",
        "source": {"kind": "field", "id": "hidden"},
    }
    with pytest.raises(HypothesisValidationError, match="latent_descriptor_mismatch"):
        HypothesisIR.from_payload(payload, contract)


def test_dimensioned_exponential_is_rejected():
    contract = ProblemContract.create("研究变化规律")
    payload = candidate(contract)
    payload["nodes"] = [payload["nodes"][0], node("rate", "exp", ["x"], {})]
    with pytest.raises(HypothesisValidationError, match="transcendental_requires_dimensionless"):
        HypothesisIR.from_payload(payload, contract)


def test_sqrt_propagates_fractional_dimensions():
    contract = ProblemContract.create("研究长度量的平方根")
    payload = candidate(contract)
    payload["nodes"] = [
        node("x", "variable", [], {"L": 2}, {"name": "x", "role": "state"}),
        node("root", "sqrt", ["x"], {"L": 1}),
    ]
    payload["outputs"] = ["root"]
    graph = HypothesisIR.from_payload(payload, contract)
    assert graph.public()["nodes"][1]["type"]["dimensions"] == {"L": 1}


def test_trigonometric_primitive_requires_dimensionless_input():
    contract = ProblemContract.create("研究量的周期变化")
    payload = candidate(contract)
    payload["nodes"] = [
        node("x", "variable", [], {"L": 1}, {"name": "x", "role": "state"}),
        node("wave", "sin", ["x"], {}),
    ]
    payload["outputs"] = ["wave"]
    with pytest.raises(HypothesisValidationError, match="transcendental_requires_dimensionless"):
        HypothesisIR.from_payload(payload, contract)


def test_bounded_primitives_require_matching_dimensions():
    contract = ProblemContract.create("研究容量截断")
    payload = candidate(contract)
    payload["nodes"] = [
        node("x", "variable", [], {"L": 1}, {"name": "x", "role": "state"}),
        node("limit", "constant", [], {"T": 1}, {"value": 1}),
        node("bounded", "maximum", ["x", "limit"], {"L": 1}),
    ]
    payload["outputs"] = ["bounded"]
    with pytest.raises(HypothesisValidationError, match="dimension_mismatch"):
        HypothesisIR.from_payload(payload, contract)


def test_calculus_units_and_boundary_obligations():
    contract = ProblemContract.create("研究位置随时间变化")
    payload = candidate(contract)
    payload["nodes"][1] = node("t", "variable", [], {"T": 1}, {"name": "t", "role": "coordinate"})
    payload["nodes"][2] = node("rate", "derivative", ["x", "t"], {"L": 1, "T": -1})
    graph = HypothesisIR.from_payload(payload, contract)
    assert "domain_or_boundary_check_required:rate" in graph.public()["obligations"]
    assert graph.public()["execution_status"] == "not_executed"


def test_fractional_unit_exponents_are_canonical():
    result = MathType.parse(math_type({"L": "2/4", "T": -1})).public()
    assert result["dimensions"] == {"L": "1/2", "T": -1}


def test_contract_or_candidate_changes_invalidate_evidence():
    contract = ProblemContract.create("研究变化规律")
    graph = HypothesisIR.from_payload(candidate(contract), contract)
    ledger = EvidenceLedger().append(graph, method="type_check", outcome="pass", scope={"checker": "v1"})
    assert len(ledger.for_hypothesis(graph)) == 1
    revision = contract.revise("研究变化规律，限制时间范围")
    revised = HypothesisIR.from_payload(candidate(revision), revision)
    assert ledger.for_hypothesis(revised) == []
    assert revision.public()["parent_hash"] == contract.digest
    modified = candidate(contract)
    modified["nodes"][2]["op"] = "divide"
    modified["nodes"][2]["type"]["dimensions"] = {"L": 1, "T": 1}
    assert ledger.for_hypothesis(HypothesisIR.from_payload(modified, contract)) == []
    exported = ledger.public()
    exported["records"][0]["outcome"] = "fail"
    assert ledger.for_hypothesis(graph)[0]["outcome"] == "pass"
    with pytest.raises(HypothesisValidationError, match="unsupported_evidence_method"):
        ledger.append(graph, method="formal_proof", outcome="pass", scope={"fake": True})


def test_user_confirmation_creates_immutable_contract_revision():
    contract = ProblemContract.create("系统在区间内运行。")
    revised = contract.record_confirmation(
        "边界是否包含终点？", "包含终点。", fact_id="boundary_confirmation", hard_constraint=True,
    )
    assert revised.digest != contract.digest
    assert revised.public()["revision"] == contract.public()["revision"] + 1
    assert revised.public()["parent_hash"] == contract.digest
    fact = next(item for item in revised.public()["facts"] if item["id"] == "boundary_confirmation")
    assert revised.public()["statement"][fact["start"]:fact["end"]] == fact["text"]
    assert "boundary_confirmation" in revised.public()["hard_constraint_ids"]
    assert "用户确认" not in contract.public()["statement"]


def test_user_confirmation_rejects_duplicate_fact_ids():
    contract = ProblemContract.create("原始题面。", facts=[
        {"id": "given", "start": 0, "end": 5, "text": "原始题面。"},
    ])
    with pytest.raises(HypothesisValidationError, match="duplicate_fact_id"):
        contract.record_confirmation("问题", "回答", fact_id="given")


def test_user_confirmation_default_ids_are_revision_scoped():
    contract = ProblemContract.create("题面")
    first = contract.record_confirmation("问题一", "回答一")
    second = first.record_confirmation("问题二", "回答二")
    ids = [item["id"] for item in second.public()["facts"]]
    assert ids[-2:] == ["user_confirmation_1", "user_confirmation_2"]


def test_problem_contract_round_trip_revalidates_public_payload():
    original = ProblemContract.create("容量为 100。", facts=[
        {"id": "capacity", "start": 0, "end": 8, "text": "容量为 100。"},
    ], hard_constraint_ids=("capacity",))
    restored = ProblemContract.from_payload(original.public())
    assert restored.digest == original.digest
    assert restored.public() == original.public()


def test_problem_contract_rehydration_rejects_tampered_quote_and_noncanonical_constraints():
    contract = ProblemContract.create("容量为 100。", facts=[
        {"id": "capacity", "start": 0, "end": 8, "text": "容量为 100。"},
    ], hard_constraint_ids=("capacity",))
    tampered = contract.public()
    tampered["facts"][0]["text"] = "容量为 999。"
    with pytest.raises(HypothesisValidationError, match="fact_quote_mismatch"):
        ProblemContract.from_payload(tampered)

    duplicated = contract.public()
    duplicated["hard_constraint_ids"].append("capacity")
    with pytest.raises(HypothesisValidationError, match="noncanonical_problem_contract"):
        ProblemContract.from_payload(duplicated)


def test_failed_model_call_uses_budget_and_redacts_errors():
    def fail(_messages):
        raise RuntimeError("secret-token and private-server-trace")

    contract, controller = ProblemContract.create("研究变化规律"), SearchController()
    proposer = generator(fail, api_key="secret-token")
    first = proposer.propose(contract, controller=controller)
    second = proposer.propose(contract, controller=controller)
    assert first["error_code"] == "hypothesis_backend_failed"
    assert second["error_code"] == "model_call_budget_exhausted"
    assert first["search_state"]["calls_used"] == 1
    assert "secret-token" not in json.dumps(first)
    assert "private-server-trace" not in json.dumps(first)


def test_raw_secret_in_response_is_not_persisted():
    result = generator(lambda _: '{"hypotheses": [], "questions": ["secret-token"]}', api_key="secret-token").propose(
        ProblemContract.create("研究变化规律"))
    assert result["error_code"] == "sensitive_response_rejected"
    assert "secret-token" not in json.dumps(result)


def test_good_and_bad_candidates_are_isolated_and_exact_duplicates_deduplicated():
    contract = ProblemContract.create("研究变化规律")
    first = candidate(contract)
    duplicate = {**deepcopy(first), "id": "h2"}
    bad = {**deepcopy(first), "id": "h3", "contract_hash": "0" * 64}
    result = generator(lambda _: json.dumps({"hypotheses": [first, duplicate, bad], "questions": []})).propose(contract)
    assert len(result["hypotheses"]) == 1
    assert len(result["rejected_proposals"]) == 2
    assert result["search_state"]["candidates_registered"] == 1


@pytest.mark.parametrize("raw", [
    '{"hypotheses": [], "hypotheses": [], "questions": []}',
    '{"hypotheses": [], "questions": [], "evidence": {"verified": true}}',
    '{"hypotheses": [], "questions": [], "score": NaN}',
    '```json\n{"hypotheses": [], "questions": []}\n```',
    '{"hypotheses": ' + '[' * 30 + '0' + ']' * 30 + ', "questions": []}',
])
def test_untrusted_json_is_bounded_and_cannot_write_evidence(raw):
    result = generator(lambda _: raw).propose(ProblemContract.create("研究变化规律"))
    assert result["status"] == "failed_safe"
    assert result["hypotheses"] == []
    assert result["evidence_ledger"]["records"] == []


def test_search_budgets_and_empty_candidates():
    with pytest.raises(HypothesisValidationError):
        SearchController(max_calls=True)
    contract = ProblemContract.create("信息不足")
    result = generator(lambda _: '{"hypotheses": [], "questions": ["是否存在外部输入？"]}').propose(contract)
    assert result["status"] == "no_valid_hypotheses"
    assert result["questions"] == ["是否存在外部输入？"]
    with pytest.raises(HypothesisValidationError, match="node_budget_exceeded"):
        HypothesisIR.from_payload(candidate(contract), contract, max_nodes=2)


def test_clarification_questions_can_offer_bounded_choices_with_impacts():
    contract = ProblemContract.create("信息不足")
    raw = json.dumps({"hypotheses": [], "questions": [{
        "question": "边界条件如何处理？",
        "options": [
            {"id": "closed", "label": "闭区间", "impact": "事件在边界点计入。"},
            {"id": "open", "label": "开区间", "impact": "事件恰在边界点不计入。"},
        ],
    }]}, ensure_ascii=False)
    result = generator(lambda _: raw).propose(contract)
    assert result["questions"] == ["边界条件如何处理？"]
    assert result["question_options"]["边界条件如何处理？"][0]["id"] == "closed"
    assert result["question_options"]["边界条件如何处理？"][-1]["id"] == "undetermined"

    invalid = json.dumps({"hypotheses": [], "questions": [{
        "question": "选择一个？", "options": [{"id": "only", "label": "只有一个", "impact": "不足以形成选择。"}],
    }]}, ensure_ascii=False)
    failed = generator(lambda _: invalid).propose(contract)
    assert failed["status"] == "failed_safe"
    assert failed["error_code"] == "invalid_question_options"
    assert failed["question_options"] == {}


def test_already_confirmed_questions_are_suppressed_without_semantic_overclaim():
    base = ProblemContract.create("信息不足")
    contract = base.record_confirmation("边界条件如何处理？", "按闭区间处理。")
    raw = json.dumps({"hypotheses": [], "questions": [
        "边界条件如何处理？", "边界条件如何处理？"
    ]}, ensure_ascii=False)
    result = generator(lambda _: raw).propose(contract)
    assert result["questions"] == []
    assert result["suppressed_questions"] == ["边界条件如何处理？", "边界条件如何处理？"]


def test_rephrased_question_is_flagged_but_not_silently_removed():
    contract = ProblemContract.create("信息不足").record_confirmation("边界条件如何处理？", "按闭区间处理。")
    raw = json.dumps({"hypotheses": [], "questions": ["边界条件应该如何处理"]}, ensure_ascii=False)
    result = generator(lambda _: raw).propose(contract)
    assert result["questions"] == ["边界条件应该如何处理"]
    assert result["possible_repeated_questions"][0]["status"] == "possible_repeat_not_suppressed"


def test_semantic_hypothesis_set_keeps_competing_interpretations_under_contract_scope():
    contract = ProblemContract.create("研究变量关系，目标可能是预测或解释。")
    semantic = SemanticHypothesisSet.create(contract, [
        {"id": "predict", "interpretation": "目标是条件预测。", "fact_ids": ["statement"],
         "assumptions": ["未来输入可获得"], "open_questions": ["预测窗口多长？"], "status": "candidate"},
        {"id": "explain", "interpretation": "目标是解释变量作用。", "fact_ids": ["statement"],
         "assumptions": ["观测变量具有可比语义"], "open_questions": [], "status": "candidate"},
    ])
    payload = semantic.public()
    payload.pop("digest")
    payload.pop("authority")
    restored = SemanticHypothesisSet.from_payload(payload, contract)
    assert restored.digest == semantic.digest
    assert restored.public()["authority"] == "semantic_hypothesis_only"
    assert len(restored.public()["hypotheses"]) == 2
    ranking = semantic.rank_questions()
    assert ranking["policy"] == "not_formal_information_gain"
    assert ranking["questions"][0]["disagreement_count"] == 1


def test_semantic_hypothesis_set_rejects_fact_rebinding():
    contract = ProblemContract.create("只允许一个事实。")
    with pytest.raises(HypothesisValidationError, match="unknown_semantic_fact_id"):
        SemanticHypothesisSet.create(contract, [{
            "id": "h", "interpretation": "候选解释", "fact_ids": ["invented"],
            "assumptions": [], "open_questions": [], "status": "candidate",
        }])


def test_question_ranking_validates_two_or_three_choices_without_executing_them():
    contract = ProblemContract.create("研究变量关系")
    semantic = SemanticHypothesisSet.create(contract, [
        {"id": "h1", "interpretation": "按实体建模。", "fact_ids": ["statement"],
         "assumptions": [], "open_questions": ["是否按实体分组？"], "status": "candidate"},
        {"id": "h2", "interpretation": "按总体建模。", "fact_ids": ["statement"],
         "assumptions": [], "open_questions": ["是否按实体分组？"], "status": "candidate"},
    ])
    result = semantic.rank_questions(options_by_question={"是否按实体分组？": [
        {"id": "yes", "label": "按实体", "impact": "保留实体层差异。", "hypothesis_ids": ["h1"]},
        {"id": "no", "label": "不分组", "impact": "使用总体关系。", "hypothesis_ids": ["h2"]},
    ]})
    assert len(result["questions"][0]["options"]) == 2
    assert result["questions"][0]["selection_policy"] == "user_choice_creates_new_contract_revision"
    assert result["questions"][0]["information_value"]["status"] == "conditional"
    assert result["questions"][0]["information_value"]["expected_information_gain_bits"] > 0
    with pytest.raises(HypothesisValidationError, match="question_options_must_have_2_to_3_choices"):
        semantic.rank_questions(options_by_question={"是否按实体分组？": [
            {"id": "only", "label": "只有一个选项", "hypothesis_ids": ["h1"]},
        ]})
    with pytest.raises(HypothesisValidationError, match="invalid_question_option_hypothesis_ids"):
        semantic.rank_questions(options_by_question={"是否按实体分组？": [
            {"id": "yes", "label": "按实体", "hypothesis_ids": ["unknown"]},
            {"id": "no", "label": "不分组", "hypothesis_ids": ["h2"]},
        ]})


def test_question_ranking_marks_overlapping_options_without_failing_heuristic():
    contract = ProblemContract.create("研究变量关系")
    semantic = SemanticHypothesisSet.create(contract, [
        {"id": "h1", "interpretation": "解释一。", "fact_ids": ["statement"],
         "assumptions": [], "open_questions": ["如何解释？"], "status": "candidate"},
        {"id": "h2", "interpretation": "解释二。", "fact_ids": ["statement"],
         "assumptions": [], "open_questions": ["如何解释？"], "status": "candidate"},
    ])
    result = semantic.rank_questions(options_by_question={"如何解释？": [
        {"id": "both", "label": "两者皆可", "hypothesis_ids": ["h1", "h2"]},
        {"id": "one", "label": "仅一类", "hypothesis_ids": ["h1"]},
    ]})
    info = result["questions"][0]["information_value"]
    assert info["status"] == "not_assessed"
    assert info["reason"] == "overlapping_answer_partition"


def test_schema_is_versioned_and_model_has_no_verdict_fields():
    schema = proposal_schema()
    assert schema["$id"].endswith(":v1")
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"hypotheses", "questions"}
    assert "descriptor" in schema["properties"]["hypotheses"]["items"]["properties"]["nodes"]["items"]["properties"]["attributes"]["properties"]
    assert schema["properties"]["questions"]["items"]["anyOf"][1]["properties"]["options"]["minItems"] == 2
    assert decode_proposal(json.dumps({"hypotheses": [], "questions": []})) == {"hypotheses": [], "questions": []}


def test_request_budget_is_atomic_under_concurrent_access():
    from concurrent.futures import ThreadPoolExecutor

    controller = SearchController()

    def reserve(_index):
        try:
            controller.reserve_call()
            return True
        except HypothesisValidationError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(12))) == 1
    assert controller.public()["calls_used"] == 1


def test_numerical_evidence_requires_exact_input_and_scope_for_reuse():
    contract = ProblemContract.create("研究变化规律")
    graph = HypothesisIR.from_payload(candidate(contract), contract)
    scope = {"input_hash": "a" * 64, "evaluator_version": "v1", "domain": {"time": [0, 10]}, "seed": 42}
    ledger = EvidenceLedger().append(graph, method="numerical_test", outcome="pass", scope=scope)
    assert ledger.for_hypothesis(graph) == []
    assert len(ledger.for_hypothesis(graph, scope=scope)) == 1
    assert ledger.for_hypothesis(graph, scope={**scope, "input_hash": "b" * 64}) == []
    assert ledger.for_hypothesis(graph, scope={**scope, "domain": {"time": [0, 100]}}) == []
    with pytest.raises(HypothesisValidationError, match="numerical_evidence_context_required"):
        ledger.append(graph, method="numerical_test", outcome="pass", scope={"fake": True})


def test_research_keeps_proposals_out_of_factual_and_numerical_evidence(tmp_path):
    from core.modeling_assistant import MathModelingAssistant

    problem = "研究物体运动，但没有给出运动方程。"
    baseline = MathModelingAssistant(output_dir=str(tmp_path / "baseline")).run(
        problem, {}, run_modeling=False, generate_plots=False)
    result = MathModelingAssistant(output_dir=str(tmp_path / "proposed"), hypothesis_generator=generator(respond)).run(
        problem, {}, run_modeling=False, generate_plots=False)
    assert "model_hypotheses" not in baseline.specialized_results
    proposals = result.specialized_results["model_hypotheses"]
    assert proposals["status"] == "proposed"
    assert result.mathematical_model_spec == baseline.mathematical_model_spec
    assert result.evidence_bundle["claims"] == baseline.evidence_bundle["claims"]
    assert result.specialized_results["mechanistic_model"]["numerical_results"] == []
    artifact = tmp_path / "proposed" / "evidence" / "model_hypotheses.json"
    assert json.loads(artifact.read_text(encoding="utf-8"))["policy"]["may_execute"] is False
    from pathlib import Path
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "候选机制提议（尚未求解，不是事实或数值证据）" in report


def test_proposal_failure_does_not_abort_research_or_expose_provider_errors(tmp_path):
    from core.modeling_assistant import MathModelingAssistant

    def fail(_messages):
        raise RuntimeError("private-token")

    result = MathModelingAssistant(output_dir=str(tmp_path), hypothesis_generator=generator(fail)).run(
        "研究运动规律", {}, run_modeling=False, generate_plots=False)
    assert result.specialized_results["model_hypotheses"]["status"] == "failed_safe"
    assert any("hypothesis_backend_failed" in warning for warning in result.warnings)
    assert "private-token" not in json.dumps(result.to_dict(), ensure_ascii=False)


def test_generated_markdown_and_html_are_presented_as_text(tmp_path):
    from pathlib import Path
    from core.modeling_assistant import MathModelingAssistant

    problem = "研究变化规律"
    payload = candidate(ProblemContract.create(problem))
    payload["assumptions"][0]["text"] = '<script>alert(1)</script> [伪造链接](https://example.invalid)'
    result = MathModelingAssistant(output_dir=str(tmp_path), hypothesis_generator=generator(
        lambda _: json.dumps({"hypotheses": [payload], "questions": []}))).run(
            problem, {}, run_modeling=False, generate_plots=False)
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "<script>" not in report
    assert "&lt;script&gt;" in report
    assert "\\[伪造链接\\]" in report
