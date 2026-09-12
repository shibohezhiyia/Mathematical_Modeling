"""A heldout result is never allowed to train, select, or silently retry a model."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import json
import sqlite3
import threading

import pytest

from test_graph_search import fixture_graph, experiment, cases
from core.confirmation_registry import ConfirmationRegistry, confirm_frozen_model
from core.graph_confirmation import FrozenGraphModel, HeldoutCases, evaluate_frozen_request
from core.graph_evaluator import evaluate_graph_request
from core.graph_experiments import SearchExperiment
from core.graph_search import GraphSearchSession
from core.graph_search_artifacts import run_search_bundle
from core.model_hypotheses import HypothesisValidationError
from core.solver_runtime import SolverLimits, SolverProcessRunner, SolverRuntimeError


def local_runner(monkeypatch):
    def execute(self, key, payload, *, limits=None, cancel=None):
        evaluate = evaluate_frozen_request if key == "scalar_graph_confirm/v1" else evaluate_graph_request
        return evaluate(payload, max_evaluations=(limits or SolverLimits()).max_evaluations)
    monkeypatch.setattr(SolverProcessRunner, "execute", execute)


def make_session():
    contract, graph = fixture_graph("multiply", parameter=True)
    exp = experiment(contract, graph, lambda x: 2*x, parameter_bounds={"a": [0, 5, 1]})
    session = GraphSearchSession(contract, exp)
    result = session.run([graph.payload()], grammar_search=False)
    return session, result, session.freeze(graph.digest)


@pytest.fixture
def frozen(monkeypatch):
    local_runner(monkeypatch)
    return make_session()[2]


def holdout(fn=lambda x: 2*x):
    return {"schema_version": "mathmodel.scalar-holdout/v1", "cases":
            cases([0.137, 0.271, 0.863, 1.123, -0.137, -0.271, -0.863, -1.123], fn, "final")}


def evaluate(model, payload=None):
    return evaluate_frozen_request({"frozen_model": model.public(), "holdout": payload or holdout()}, max_evaluations=1000)


def test_frozen_model_cannot_be_changed_through_public_outcome(monkeypatch):
    local_runner(monkeypatch)
    session, result, frozen = make_session()
    digest = frozen.digest
    result["reports"][0]["parameters"]["a"] = 100
    frozen.public()["parameters"]["a"] = 99
    assert session.freeze(frozen.public()["development_evidence"]["hypothesis_hash"]).digest == digest
    assert frozen.public()["parameters"]["a"] == pytest.approx(2)
    with pytest.raises(FrozenInstanceError):
        frozen._json = "{}"


def test_cannot_freeze_unexecuted_or_failed_candidate(monkeypatch):
    local_runner(monkeypatch)
    contract, graph = fixture_graph()
    session = GraphSearchSession(contract, experiment(contract, graph))
    with pytest.raises(HypothesisValidationError, match="search_not_finished"):
        session.freeze(graph.digest)
    session.run([graph.payload()], grammar_search=False)
    with pytest.raises(HypothesisValidationError, match="candidate_not_selectable"):
        session.freeze(graph.digest)


def test_selection_cannot_change_after_freeze(monkeypatch):
    local_runner(monkeypatch)
    session, _, frozen = make_session()
    with pytest.raises(HypothesisValidationError, match="candidate_selection_already_frozen"):
        session.freeze("0"*64)


def test_holdout_changes_results_not_parameters_or_development(frozen, monkeypatch):
    def must_not_fit(*args, **kwargs):
        pytest.fail("Heldout evaluator tried to fit parameters")
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, "least_squares", must_not_fit)
    before = frozen.public()
    good = evaluate(frozen)
    bad = evaluate(frozen, holdout(lambda x: 20*x))
    assert good["status"] == "passed_finite_heldout_checks"
    assert bad["status"] == "failed_heldout_checks"
    assert good["parameters"] == bad["parameters"] == before["parameters"]
    assert frozen.public() == before
    assert not good["parameters_refitted"] and not good["may_feed_search"]
    assert good["check_count"] == good["planned_check_count"]
    assert good["metrics"][0]["rmse"] < 1e-6
    assert good["metrics"][0]["training_mean_baseline_rmse"] > 0.1
    assert good["semantic_verdict"] == "not_assessed" and not good["continuous_domain_proof"]


@pytest.mark.parametrize("value", [-2, -1.5, 0, -0.0, 3])
def test_holdout_excludes_training_search_and_fixed_probes(frozen, value):
    payload = holdout()
    payload["cases"][0]["bindings"]["x"] = value
    with pytest.raises(HypothesisValidationError, match="holdout_development_overlap"):
        HeldoutCases.from_payload(payload, frozen)


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(training_cases=[]), "unexpected_fields"),
    (lambda p: p.update(absolute_tolerance=1e9), "unexpected_fields"),
    (lambda p: p.update(cases=[]), "holdout_case_budget"),
    (lambda p: p["cases"][0]["bindings"].update(x=99), "point_outside_domain"),
    (lambda p: p["cases"][0]["expected"].update(y=float("inf")), "nonfinite_or_large_number"),
    (lambda p: p["cases"][0].update(id=p["cases"][1]["id"]), "duplicate_holdout_id"),
    (lambda p: p["cases"][0].update(bindings=p["cases"][1]["bindings"]), "duplicate_holdout_point"),
])
def test_holdout_validation_is_strict(frozen, mutation, error):
    payload = holdout()
    mutation(payload)
    with pytest.raises(HypothesisValidationError, match=error):
        HeldoutCases.from_payload(payload, frozen)


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p["parameters"].update(a=3), "frozen_evaluation_mismatch"),
    (lambda p: p["parameters"].update(a=99), "invalid_frozen_parameters"),
    (lambda p: p.update(verified=True), "unexpected_fields"),
    (lambda p: p["experiment"].update(absolute_tolerance=100), "frozen_evidence_scope_mismatch"),
    (lambda p: p["development_evidence"].update(final_data_seen=True), "frozen_selection_policy_mismatch"),
])
def test_frozen_snapshot_detects_accidental_tampering(frozen, mutation, error):
    payload = frozen.public()
    mutation(payload)
    with pytest.raises(HypothesisValidationError, match=error):
        FrozenGraphModel.from_payload(payload)


def test_digest_ignores_row_ids_order_and_integer_spelling(frozen):
    first = holdout()
    first["cases"][0]["expected"]["y"] = 1
    second = deepcopy(first)
    second["cases"].reverse()
    for index, case in enumerate(second["cases"]):
        case["id"] = f"renamed_{index}"
        case["expected"]["y"] = float(case["expected"]["y"])
    assert HeldoutCases.from_payload(first, frozen).digest == HeldoutCases.from_payload(second, frozen).digest


def test_registry_blocks_relabel_reorder_subset_and_restart(frozen, tmp_path):
    path = tmp_path / "usage.sqlite3"
    registry = ConfirmationRegistry(path)
    data = HeldoutCases.from_payload(holdout(), frozen)
    attempt = registry.reserve("study", frozen, data)
    assert registry.inspect(attempt)["status"] == "reserved"
    changed = holdout(lambda x: 100*x)
    changed["cases"].reverse()
    changed["cases"] = changed["cases"][:2]
    for index, case in enumerate(changed["cases"]):
        case["id"] = f"new_{index}"
    with pytest.raises(HypothesisValidationError, match="holdout_already_consumed_in_study"):
        ConfirmationRegistry(path).reserve("study", frozen, HeldoutCases.from_payload(changed, frozen))


def test_registry_transaction_rolls_back_new_points_on_partial_overlap(frozen, tmp_path):
    registry = ConfirmationRegistry(tmp_path / "usage.sqlite3")
    old = holdout()
    registry.reserve("study", frozen, HeldoutCases.from_payload(old, frozen))
    mixed = deepcopy(old)
    mixed["cases"][0]["bindings"]["x"] = 0.4321
    with pytest.raises(HypothesisValidationError, match="holdout_already_consumed_in_study"):
        registry.reserve("study", frozen, HeldoutCases.from_payload(mixed, frozen))
    mixed["cases"] = mixed["cases"][:1]
    assert registry.reserve("study", frozen, HeldoutCases.from_payload(mixed, frozen))


def test_concurrent_registry_reservation_has_only_one_winner(frozen, tmp_path):
    registry = ConfirmationRegistry(tmp_path / "usage.sqlite3")
    data = HeldoutCases.from_payload(holdout(), frozen)
    barrier = threading.Barrier(2)
    def reserve():
        barrier.wait(timeout=5)
        try:
            return registry.reserve("study", frozen, data)
        except HypothesisValidationError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: reserve(), range(2)))
    assert results.count("holdout_already_consumed_in_study") == 1


def test_registry_does_not_modify_an_unrelated_database(tmp_path):
    path = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE personal_notes(note TEXT)")
    with pytest.raises(HypothesisValidationError, match="unrecognized_confirmation_registry"):
        ConfirmationRegistry(path)
    with sqlite3.connect(path) as connection:
        assert [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")] == ["personal_notes"]


def test_successful_confirmation_is_consumed_and_scoped(frozen, tmp_path):
    registry = ConfirmationRegistry(tmp_path / "usage.sqlite3")
    result = confirm_frozen_model(frozen, holdout(), registry=registry, study="study")
    assert result["status"] == "passed_finite_heldout_checks"
    assert result["consumption"]["status"] == "completed"
    assert result["consumption"]["guard_scope"] == "same_study_same_registry"
    with pytest.raises(HypothesisValidationError, match="holdout_already_consumed_in_study"):
        confirm_frozen_model(frozen, holdout(), registry=registry, study="study")


@pytest.mark.parametrize("code", ["timeout", "memory_limit", "cancelled"])
def test_failed_execution_still_consumes_holdout(frozen, tmp_path, monkeypatch, code):
    def failed(*args, **kwargs):
        raise SolverRuntimeError(code)
    monkeypatch.setattr(SolverProcessRunner, "execute", failed)
    registry = ConfirmationRegistry(tmp_path / "usage.sqlite3")
    result = confirm_frozen_model(frozen, holdout(), registry=registry, study="study")
    assert result["status"] == "execution_incomplete"
    with pytest.raises(HypothesisValidationError, match="holdout_already_consumed_in_study"):
        registry.reserve("study", frozen, HeldoutCases.from_payload(holdout(), frozen))


def test_unexpected_interruption_leaves_durable_reservation(frozen, tmp_path, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(SolverProcessRunner, "execute", interrupt)
    path = tmp_path / "usage.sqlite3"
    with pytest.raises(KeyboardInterrupt):
        confirm_frozen_model(frozen, holdout(), registry=ConfirmationRegistry(path), study="study")
    with pytest.raises(HypothesisValidationError, match="holdout_already_consumed_in_study"):
        ConfirmationRegistry(path).reserve("study", frozen, HeldoutCases.from_payload(holdout(), frozen))


def test_forged_worker_parameters_cannot_be_accepted(frozen, tmp_path, monkeypatch):
    original = evaluate(frozen)
    original["parameters"]["a"] = 3
    monkeypatch.setattr(SolverProcessRunner, "execute", lambda *a, **k: original)
    result = confirm_frozen_model(frozen, holdout(), registry=ConfirmationRegistry(tmp_path / "usage.sqlite3"), study="study")
    assert result["status"] == "execution_incomplete" and result["failure_code"] == "invalid_response"


def test_final_failures_are_not_added_to_search_history(monkeypatch, tmp_path):
    local_runner(monkeypatch)
    session, before, frozen = make_session()
    serialized = json.dumps(before, sort_keys=True)
    final = confirm_frozen_model(frozen, holdout(lambda x: 200*x),
        registry=ConfirmationRegistry(tmp_path / "usage.sqlite3"), study="study")
    assert final["status"] == "failed_heldout_checks" and not final["may_feed_search"]
    assert json.dumps(before, sort_keys=True) == serialized
    assert session.freeze(before["pareto_candidates"][0]).digest == frozen.digest


def test_lookalike_mechanism_passes_development_but_fails_fresh_points(frozen):
    import math
    # This single mechanism matches 2*x exactly at ALL observed development
    # inputs, but differs elsewhere. A perfect development fit is insufficient.
    roots = [-2, -1, 1, -1.5, 0, 0.5, 2]
    def mechanism(x):
        return 2*x + 0.1*math.prod(x-root for root in roots)
    assert all(mechanism(x) == 2*x for x in roots)
    result = evaluate(frozen, holdout(mechanism))
    assert result["status"] == "failed_heldout_checks"
    assert result["failed_check_count"] > 0
    assert result["parameters"]["a"] == pytest.approx(2)


def singular_frozen_model():
    from test_graph_search import node
    from core.model_hypotheses import HypothesisIR
    contract, initial = fixture_graph()
    payload = initial.payload()
    payload["nodes"] = [payload["nodes"][0], node("c", "constant", attrs={"value": 0.137}),
                        node("one", "constant", attrs={"value": 1}), node("den", "subtract", ["x", "c"]),
                        node("y", "divide", ["one", "den"])]
    graph = HypothesisIR.from_payload(payload, contract)
    exp = experiment(contract, graph, lambda x: 1/(x-0.137))
    session = GraphSearchSession(contract, exp)
    assert session.run([graph.payload()], grammar_search=False)["pareto_candidates"]
    return session.freeze(graph.digest)


def test_unseen_singularity_is_not_hidden_by_other_correct_predictions(monkeypatch):
    local_runner(monkeypatch)
    frozen = singular_frozen_model()
    data = holdout(lambda x: 0 if x == 0.137 else 1/(x-0.137))
    result = evaluate(frozen, data)
    assert result["status"] == "failed_heldout_checks"
    assert result["failures"][0]["check"] == "finite_output"
    assert result["metrics"] == []
    assert result["check_count"] < result["planned_check_count"]


def test_real_worker_confirmation_budget_cannot_be_evaded_by_fallback(tmp_path):
    frozen = singular_frozen_model()
    data = holdout(lambda x: 0 if x == 0.137 else 1/(x-0.137))
    result = confirm_frozen_model(frozen, data, registry=ConfirmationRegistry(tmp_path / "usage.sqlite3"),
                                 study="budget", limits=SolverLimits(max_evaluations=1))
    assert result["status"] == "execution_incomplete" and result["failure_code"] == "evaluation_limit"


def test_real_worker_confirms_without_refitting_and_respects_budget(tmp_path):
    _, _, model = make_session()
    result = confirm_frozen_model(model, holdout(), registry=ConfirmationRegistry(tmp_path / "usage.sqlite3"), study="real")
    assert result["status"] == "passed_finite_heldout_checks"
    assert result["execution_supervision"]["process_isolated"]
    assert result["parameters_refitted"] is False
    request = {"frozen_model": model.public(), "holdout": holdout()}
    request["frozen_model"]["parameters"]["a"] = 100
    with pytest.raises(SolverRuntimeError, match="invalid_contract"):
        SolverProcessRunner().execute("scalar_graph_confirm/v1", request)


def test_bundle_freezes_before_first_holdout_read_and_saves_certificate(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "examples" / "graph_search_scalar.json").read_text(encoding="utf-8"))
    final_path = tmp_path / "heldout.json"
    final_path.write_text(json.dumps(holdout(lambda x: x*x)), encoding="utf-8")
    events = []
    original_freeze, original_open = GraphSearchSession.freeze, Path.open
    def freeze(self, key):
        value = original_freeze(self, key)
        events.append("freeze")
        return value
    def open_path(self, *args, **kwargs):
        if self == final_path:
            assert events == ["freeze"]
            events.append("read_holdout")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(GraphSearchSession, "freeze", freeze)
    monkeypatch.setattr(Path, "open", open_path)
    result, directory = run_search_bundle(bundle, output_root=tmp_path / "runs", confirmation_path=final_path,
        study="study", registry_path=tmp_path / "usage.sqlite3")
    assert events == ["freeze", "read_holdout"]
    assert result["confirmation"]["status"] == "passed_finite_heldout_checks"
    assert (directory / "evidence" / "frozen_model.json").is_file()
    assert (directory / "evidence" / "heldout_evidence.json").is_file()
    report = (directory / "reports" / "graph_search.md").read_text(encoding="utf-8")
    assert "留出检验通过" in report and "先选定模型并冻结" in report
    evidence = json.loads((directory / "evidence" / "heldout_evidence.json").read_text(encoding="utf-8"))
    assert evidence["records"][0]["scope"]["phase"] == "frozen_heldout"
    assert evidence["records"][0]["scope"]["may_feed_search"] is False


def test_no_candidate_never_opens_holdout_file(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "examples" / "graph_search_scalar.json").read_text(encoding="utf-8"))
    bundle["grammar_search"] = False
    nonexistent = tmp_path / "must_not_open.json"
    result, _ = run_search_bundle(bundle, output_root=tmp_path / "runs", confirmation_path=nonexistent,
                                  study="study", registry_path=tmp_path / "usage.sqlite3")
    assert result["confirmation"]["status"] == "no_selectable_candidate"
    assert result["confirmation"]["holdout_file_read"] is False
    assert not (tmp_path / "usage.sqlite3").exists()


def test_bad_candidate_selection_preserves_search_without_reading_holdout(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "examples" / "graph_search_scalar.json").read_text(encoding="utf-8"))
    result, path = run_search_bundle(bundle, output_root=tmp_path / "runs", confirmation_path=tmp_path / "missing.json",
                                    study="study", candidate_hash="0"*64, registry_path=tmp_path / "usage.sqlite3")
    assert result["pareto_candidates"]
    assert result["confirmation"]["failure_code"] == "candidate_not_selectable"
    assert not result["confirmation"]["holdout_file_read"]
    assert (path / "evidence" / "graph_search.json").is_file()


def test_invalid_study_is_rejected_before_any_output_or_holdout_read(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "examples" / "graph_search_scalar.json").read_text(encoding="utf-8"))
    with pytest.raises(HypothesisValidationError, match="invalid_study_id"):
        run_search_bundle(bundle, output_root=tmp_path / "runs", confirmation_path=tmp_path / "missing.json", study="../bad")
    assert not (tmp_path / "runs").exists()


def test_repeated_bundle_confirmation_preserves_search_and_rejects_reuse(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    root = Path(__file__).resolve().parents[1]
    bundle = json.loads((root / "examples" / "graph_search_scalar.json").read_text(encoding="utf-8"))
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(holdout(lambda x: x*x)), encoding="utf-8")
    kwargs = {"output_root": tmp_path / "runs", "confirmation_path": path, "study": "study",
              "registry_path": tmp_path / "usage.sqlite3"}
    first, _ = run_search_bundle(bundle, **kwargs)
    second, directory = run_search_bundle(bundle, **kwargs)
    assert first["confirmation"]["status"] == "passed_finite_heldout_checks"
    assert second["pareto_candidates"] == first["pareto_candidates"]
    assert second["confirmation"]["failure_code"] == "holdout_already_consumed_in_study"
    assert (directory / "reports" / "graph_search.md").is_file()


def test_registry_reopening_does_not_leak_connections(frozen, tmp_path):
    path = tmp_path / "usage.sqlite3"
    for _ in range(20):
        ConfirmationRegistry(path)
    registry = ConfirmationRegistry(path)
    attempt = registry.reserve("study", frozen, HeldoutCases.from_payload(holdout(), frozen))
    registry.complete(attempt, "passed_finite_heldout_checks")
    with pytest.raises(HypothesisValidationError, match="confirmation_attempt_not_reserved"):
        registry.complete(attempt, "failed_heldout_checks")
    assert registry.inspect(attempt)["outcome"] == "passed_finite_heldout_checks"


def test_registry_schema_creation_is_atomic_across_threads(tmp_path):
    path = tmp_path / "usage.sqlite3"
    barrier = threading.Barrier(2)
    def create(_):
        barrier.wait(timeout=5)
        return ConfirmationRegistry(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert len(list(executor.map(create, range(2)))) == 2
