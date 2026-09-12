"""Paired holdout reservations must not become adaptive model selection."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import pytest

from test_graph_confirmation import frozen, holdout
from core.confirmation_registry import ConfirmationRegistry, confirm_frozen_model
from core.graph_confirmation import evaluate_frozen_request
from core.graph_panel import FrozenModelPanel, confirm_frozen_panel
from core.model_hypotheses import HypothesisValidationError
from core.solver_runtime import SolverProcessRunner, SolverRuntimeError


def panel_for(model):
    # Identical models are intentional: different methods may discover the same graph.
    return FrozenModelPanel.from_models({"baseline": model, "search": model})


def test_panel_is_immutable_and_preserves_method_labels(frozen):
    panel = panel_for(frozen)
    digest = panel.digest
    panel.public()["arms"].clear()
    assert panel.digest == digest and len(panel.public()["arms"]) == 2


@pytest.mark.parametrize("change,error", [
    (lambda p: p.update(arms=p["arms"][:1]), "panel_arm_budget"),
    (lambda p: p["arms"][1].update(id="baseline"), "duplicate_panel_arm"),
    (lambda p: p.update(wall_seconds_per_arm=31), "panel_wall_budget"),
    (lambda p: p.update(evaluations_per_arm=True), "panel_evaluation_budget"),
    (lambda p: p.update(select_best=True), "unexpected_fields"),
])
def test_panel_contract_rejects_invalid_or_adaptive_settings(frozen, change, error):
    payload = panel_for(frozen).public()
    change(payload)
    with pytest.raises(HypothesisValidationError, match=error):
        FrozenModelPanel.from_payload(payload)


def test_panel_checks_all_models(frozen):
    payload = panel_for(frozen).public()
    payload["arms"][1]["model"]["parameters"]["a"] = 99
    with pytest.raises(HypothesisValidationError):
        FrozenModelPanel.from_payload(payload)


def test_paired_checks_use_same_data_and_fixed_quotas(frozen, monkeypatch, tmp_path):
    observed = []
    def execute(self, key, payload, *, limits, cancel):
        observed.append((deepcopy(payload), limits))
        return evaluate_frozen_request(payload, max_evaluations=limits.max_evaluations)
    monkeypatch.setattr(SolverProcessRunner, "execute", execute)
    registry = ConfirmationRegistry(tmp_path / "audit.sqlite3")
    result = confirm_frozen_panel(panel_for(frozen), holdout(), registry=registry, study="paired")
    assert result["all_arms_passed"] and not result["winner_selected"]
    assert not result["may_feed_search"]
    assert observed[0][0]["holdout"] == observed[1][0]["holdout"]
    assert observed[0][1] == observed[1][1]
    assert [a["id"] for a in result["arms"]] == ["baseline", "search"]
    with pytest.raises(HypothesisValidationError, match="holdout_already_consumed"):
        confirm_frozen_model(frozen, holdout(), registry=registry, study="paired")


def test_failure_is_retained_without_retry_or_quota_transfer(frozen, monkeypatch, tmp_path):
    calls = []
    def execute(self, key, payload, *, limits, cancel):
        calls.append(limits)
        if len(calls) == 1:
            raise SolverRuntimeError("timeout")
        return evaluate_frozen_request(payload, max_evaluations=limits.max_evaluations)
    monkeypatch.setattr(SolverProcessRunner, "execute", execute)
    registry = ConfirmationRegistry(tmp_path / "audit.sqlite3")
    result = confirm_frozen_panel(panel_for(frozen), holdout(), registry=registry, study="paired")
    assert len(calls) == 2 and calls[0] == calls[1]
    assert result["status"] == "panel_completed" and not result["all_arms_passed"]
    assert result["arms"][0]["result"]["status"] == "execution_incomplete"
    assert result["arms"][1]["result"]["status"] == "passed_finite_heldout_checks"


def test_interruption_consumes_entire_panel(frozen, monkeypatch, tmp_path):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(SolverProcessRunner, "execute", interrupt)
    registry = ConfirmationRegistry(tmp_path / "audit.sqlite3")
    with pytest.raises(KeyboardInterrupt):
        confirm_frozen_panel(panel_for(frozen), holdout(), registry=registry, study="paired")
    with pytest.raises(HypothesisValidationError, match="holdout_already_consumed"):
        registry.reserve_panel("paired", panel_for(frozen), holdout())


def test_concurrent_panels_share_single_consumption_guard(frozen, tmp_path):
    registry = ConfirmationRegistry(tmp_path / "audit.sqlite3")
    def reserve(_):
        try:
            return registry.reserve_panel("paired", panel_for(frozen), holdout())
        except HypothesisValidationError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, range(2)))
    assert sum(x is not None for x in outcomes) == 1


def test_bad_holdout_does_not_partially_consume(frozen, tmp_path):
    registry = ConfirmationRegistry(tmp_path / "audit.sqlite3")
    payload = holdout()
    payload["cases"][0]["bindings"]["x"] = 0
    with pytest.raises(HypothesisValidationError, match="holdout_development_overlap"):
        registry.reserve_panel("paired", panel_for(frozen), payload)
    assert registry.reserve_panel("paired", panel_for(frozen), holdout())


def test_different_development_protocols_cannot_be_paired(frozen):
    from core.graph_experiments import SearchExperiment, restore_problem
    from core.graph_search import GraphSearchSession
    from core.model_hypotheses import HypothesisIR
    spec = frozen.public()
    contract = restore_problem(spec["problem"])
    exp = spec["experiment"]
    exp["absolute_tolerance"] *= 2
    session = GraphSearchSession(contract, SearchExperiment.from_payload(exp, contract))
    graph = HypothesisIR.from_payload(spec["hypothesis"], contract)
    session.run([graph.payload()], grammar_search=False)
    other = session.freeze(graph.digest)
    with pytest.raises(HypothesisValidationError, match="panel_experiments_differ"):
        FrozenModelPanel.from_models({"first": frozen, "other": other})


def test_real_supervised_panel(frozen, monkeypatch, tmp_path):
    monkeypatch.undo()  # Restore the actual isolated subprocess runner.
    result = confirm_frozen_panel(panel_for(frozen), holdout(),
        registry=ConfirmationRegistry(tmp_path / "audit.sqlite3"), study="real")
    assert result["all_arms_passed"]
    assert all("execution_supervision" in a["result"] for a in result["arms"])
