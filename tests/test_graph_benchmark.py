import json
from pathlib import Path
import threading

import pytest

from test_graph_search import fixture_graph, experiment
from test_graph_confirmation import holdout, local_runner
from core.graph_benchmark import BENCHMARK_VERSION, run_graph_benchmark, validate_benchmark
from core.model_hypotheses import HypothesisValidationError


def spec_for(fn=lambda x: x*x):
    contract, graph = fixture_graph()
    return {"schema_version": BENCHMARK_VERSION, "problem": contract.public(),
        "experiment": experiment(contract, graph, fn).public(), "candidates": [graph.payload()],
        "budget": {"max_candidates": 8, "max_evaluations": 2000, "wall_seconds": 30},
        "arms": [{"id": "baseline", "grammar_search": False, "reuse_intermediates": True},
                 {"id": "search", "grammar_search": True, "reuse_intermediates": True}]}


def execute(spec, tmp_path, fn=lambda x: x*x, cancel=None):
    source = tmp_path / "heldout.json"
    source.write_text(json.dumps(holdout(fn)), encoding="utf-8")
    return run_graph_benchmark(spec, holdout_path=source, output_root=tmp_path / "runs",
        registry_path=tmp_path / "usage.sqlite3", study="paired", cancel=cancel)


def test_failed_method_retained_and_single_success_confirmed(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    result, directory = execute(spec_for(), tmp_path)
    assert len(result["arms"]) == 2 and result["accepted_arms"] == 1
    assert result["arms"][0]["confirmation"]["status"] == "no_candidate_passed"
    assert result["arms"][1]["confirmation"]["status"] == "passed_finite_heldout_checks"
    assert not result["winner_selected"] and result["peak_memory_bytes"] is None
    assert "1 / 2" in (directory / "reports" / "benchmark.md").read_text(encoding="utf-8")
    repeated, _ = execute(spec_for(), tmp_path)
    assert repeated["arms"][1]["confirmation"]["failure_code"] == "holdout_already_consumed_in_study"


def test_all_methods_frozen_before_final_file_open(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    original = Path.open
    reads = []
    def guarded(self, *args, **kwargs):
        if self == tmp_path / "heldout.json" and args and args[0] == "rb":
            selection = next((tmp_path / "runs").glob("*/evidence/frozen_selection.json"))
            payload = json.loads(selection.read_text(encoding="utf-8"))
            assert len(payload["arms"]) == 2 and len(payload["models"]) == 2
            assert len(list((selection.parent).glob("development_*.json"))) == 2
            assert (selection.parent / "frozen_panel.json").exists()
            reads.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    result, _ = execute(spec_for(lambda x: x), tmp_path, lambda x: x)
    assert len(reads) == 1 and result["accepted_arms"] == 2


def test_no_candidate_never_reads_holdout(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    spec = spec_for(lambda x: 123.456)
    result, directory = run_graph_benchmark(spec, holdout_path=tmp_path / "absent.json",
        output_root=tmp_path / "runs", study="paired", registry_path=tmp_path / "usage.sqlite3")
    assert result["accepted_arms"] == 0 and len(result["arms"]) == 2
    assert not (directory / "evidence" / "heldout_input.json").exists()
    assert not (tmp_path / "usage.sqlite3").exists()


def test_single_search_cancelled_manifest_is_valid(tmp_path, monkeypatch):
    from core.graph_search_artifacts import run_search_bundle, BUNDLE_VERSION
    local_runner(monkeypatch)
    spec = spec_for()
    spec.pop("arms")
    spec["schema_version"] = BUNDLE_VERSION
    cancel = threading.Event()
    cancel.set()
    result, directory = run_search_bundle(spec, output_root=tmp_path / "runs", cancel=cancel)
    assert result["termination"] == "cancelled"
    assert json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))["status"] == "incomplete"


def test_cli_real_workers_keep_failed_baseline(tmp_path, monkeypatch, capsys):
    from scripts.benchmark_graph_search import main
    from core.confirmation_registry import ConfirmationRegistry
    import core.graph_benchmark as module
    monkeypatch.setattr(module, "ConfirmationRegistry", lambda path: ConfirmationRegistry(tmp_path / "usage.sqlite3"))
    root = Path(__file__).resolve().parents[1]
    args = [str(root / "examples" / "graph_search_scalar.json"), "--holdout",
            str(root / "examples" / "graph_search_heldout.json"), "--study", "paired",
            "--compare-cache", "--output-root", str(tmp_path / "runs")]
    assert main(args) == 1  # Missing baseline solution is not silently discarded.
    assert "3 / 4" in capsys.readouterr().out
    result_path = next((tmp_path / "runs").glob("*/evidence/benchmark_result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["accepted_arms"] == 3 and len(result["arms"]) == 4
    assert all("execution_supervision" in a["confirmation"] for a in result["arms"][1:])


def test_runtime_failure_is_not_mathematical_rejection(tmp_path, monkeypatch):
    from core.solver_runtime import SolverProcessRunner, SolverRuntimeError
    local_runner(monkeypatch)
    original = SolverProcessRunner.execute
    def first_fails(self, key, payload, **kwargs):
        if not getattr(first_fails, "called", False):
            first_fails.called = True
            raise SolverRuntimeError("timeout")
        return original(self, key, payload, **kwargs)
    monkeypatch.setattr(SolverProcessRunner, "execute", first_fails)
    result, directory = execute(spec_for(), tmp_path)
    assert result["accepted_arms"] == 1
    baseline = json.loads((directory / "evidence" / "development_0.json").read_text(encoding="utf-8"))
    assert baseline["reports"][0]["status"] == "execution_incomplete"
    assert baseline["budget"]["evaluations_charged"] == baseline["budget"]["per_candidate_evaluations"]


def test_cache_arms_receive_equal_quotas_and_keep_same_selection(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    spec = spec_for()
    spec["arms"][0].update(grammar_search=True, reuse_intermediates=False)
    result, _ = execute(spec, tmp_path)
    left, right = result["arms"]
    assert left["selected_hash"] == right["selected_hash"]
    keys = ("max_candidates", "max_evaluations", "per_candidate_evaluations", "wall_seconds", "max_patch_attempts")
    assert all(left["development_budget"][k] == right["development_budget"][k] for k in keys)
    assert left["development_budget"]["evaluations_charged"] == right["development_budget"]["evaluations_charged"]


def test_diagnostic_arm_uses_same_budget_and_records_hint_scope(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    spec = spec_for()
    spec["arms"].append({"id": "diagnostic", "grammar_search": True, "reuse_intermediates": True,
                          "diagnostic_hints": [{"status": "proposal_not_executed", "hard_constraint": False,
                                                "may_feed_search": True, "requires_current_validation": True,
                                                "candidate_primitives": ["sinusoidal_basis"]}]})
    result, directory = execute(spec, tmp_path)
    assert result["arms"][-1]["diagnostic_hint_count"] == 1
    assert result["arms"][-1]["development_budget"]["max_evaluations"] == 2000
    payload = json.loads((directory / "evidence" / "development_2.json").read_text(encoding="utf-8"))
    assert payload["policy"]["diagnostic_hint_count"] == 1


def test_changed_final_targets_cannot_change_development(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    good_dir, bad_dir = tmp_path / "good", tmp_path / "bad"
    good_dir.mkdir()
    bad_dir.mkdir()
    good, _ = execute(spec_for(), good_dir)
    bad, _ = execute(spec_for(), bad_dir, lambda x: 77*x)
    assert good["accepted_arms"] == 1 and bad["accepted_arms"] == 0
    assert [a["selected_hash"] for a in good["arms"]] == [a["selected_hash"] for a in bad["arms"]]
    assert bad["arms"][1]["confirmation"]["status"] == "failed_heldout_checks"


@pytest.mark.parametrize("change,error", [
    (lambda s: s["arms"][1].update(budget={}), "unexpected_fields"),
    (lambda s: s.update(heldout=[]), "unexpected_fields"),
    (lambda s: s["arms"][1].update(id="baseline"), "duplicate_benchmark_arm"),
    (lambda s: s["arms"][1].update(grammar_search=1), "invalid_benchmark_switch"),
    (lambda s: s.update(candidates=[]), "benchmark_candidate_budget"),
])
def test_protocol_rejects_adaptive_or_incomparable_settings(change, error):
    spec = spec_for()
    change(spec)
    with pytest.raises(HypothesisValidationError, match=error):
        validate_benchmark(spec)


def test_cancelled_run_retains_all_methods_without_final_read(tmp_path, monkeypatch):
    local_runner(monkeypatch)
    cancel = threading.Event()
    cancel.set()
    result, directory = execute(spec_for(), tmp_path, cancel=cancel)
    assert len(result["arms"]) == 2 and result["accepted_arms"] == 0
    assert all(a["termination"] == "cancelled" for a in result["arms"])
    assert not (directory / "evidence" / "heldout_input.json").exists()
