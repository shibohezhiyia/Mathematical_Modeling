"""Actual file handoff must freeze every member before opening heldout data."""
import hashlib
import json
from pathlib import Path

import pytest

from test_graph_confirmation import frozen, holdout
from test_graph_panel import panel_for
from core.graph_panel_artifacts import run_panel_confirmation
from core.model_hypotheses import HypothesisValidationError
from core.solver_runtime import SolverProcessRunner


def run(panel, path, root, registry):
    return run_panel_confirmation(panel.public(), holdout_path=path, output_root=root,
                                  registry_path=registry, study="paired")


def test_snapshot_is_saved_before_first_holdout_read(frozen, tmp_path, monkeypatch):
    source = tmp_path / "holdout.json"
    source.write_text(json.dumps(holdout()), encoding="utf-8")
    output = tmp_path / "runs"
    original = Path.open
    reads = []
    panel = panel_for(frozen)
    def checked_open(self, *args, **kwargs):
        if self == source:
            snapshots = list(output.glob("*/evidence/frozen_panel.json"))
            assert len(snapshots) == 1
            assert json.loads(snapshots[0].read_text(encoding="utf-8")) == panel.public()
            reads.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked_open)
    result, directory = run(panel, source, output, tmp_path / "usage.sqlite3")
    assert result["all_arms_passed"] and len(reads) == 1
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        saved = directory / artifact["relative_path"]
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == artifact["sha256"]
    report = (directory / "reports" / "panel_confirmation.md").read_text(encoding="utf-8")
    assert "方法 baseline" in report and "方法 search" in report and "不依据留出表现选冠军" in report


def test_invalid_panel_never_opens_holdout_or_creates_output(frozen, tmp_path, monkeypatch):
    panel = panel_for(frozen).public()
    panel["arms"][1]["id"] = "baseline"
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid panel must not open any input")
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(HypothesisValidationError, match="duplicate_panel_arm"):
        run_panel_confirmation(panel, holdout_path=tmp_path / "secret.json",
                               output_root=tmp_path / "runs", study="paired")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("raw,code", [(b"broken", "invalid_json"),
                                       (b"x" * 256001, "holdout_file_size_limit"),
                                       (b"\xff", "panel_file_unavailable")], ids=["json", "size", "encoding"])
def test_invalid_holdout_preserves_frozen_snapshot(frozen, tmp_path, raw, code):
    source = tmp_path / "holdout.json"
    source.write_bytes(raw)
    result, directory = run(panel_for(frozen), source, tmp_path / "runs", tmp_path / "usage.sqlite3")
    assert result["failure_code"] == code
    assert (directory / "evidence" / "frozen_panel.json").exists()
    assert (directory / "reports" / "panel_confirmation.md").exists()
    assert not (tmp_path / "usage.sqlite3").exists()


def test_output_directory_change_does_not_reset_registry(frozen, tmp_path):
    source = tmp_path / "holdout.json"
    source.write_text(json.dumps(holdout()), encoding="utf-8")
    registry = tmp_path / "usage.sqlite3"
    first, _ = run(panel_for(frozen), source, tmp_path / "runs1", registry)
    second, directory = run(panel_for(frozen), source, tmp_path / "runs2", registry)
    assert first["all_arms_passed"]
    assert second["failure_code"] == "holdout_already_consumed_in_study"
    assert (directory / "reports" / "panel_confirmation.md").exists()


def test_interruption_keeps_snapshot_manifest_and_consumption(frozen, tmp_path, monkeypatch):
    source = tmp_path / "holdout.json"
    source.write_text(json.dumps(holdout()), encoding="utf-8")
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(SolverProcessRunner, "execute", interrupted)
    output, registry = tmp_path / "runs", tmp_path / "usage.sqlite3"
    with pytest.raises(KeyboardInterrupt):
        run(panel_for(frozen), source, output, registry)
    manifest = next(output.glob("*/artifact_manifest.json"))
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "failed"
    retry, _ = run(panel_for(frozen), source, output, registry)
    assert retry["failure_code"] == "holdout_already_consumed_in_study"


@pytest.mark.parametrize("valid_targets,expected_exit", [(True, 0), (False, 1)])
def test_cli_runs_real_workers_and_reports_each_model(frozen, tmp_path, monkeypatch, capsys,
                                                      valid_targets, expected_exit):
    from scripts.confirm_graph_panel import main
    from core.confirmation_registry import ConfirmationRegistry
    import core.graph_panel_artifacts as artifacts
    monkeypatch.undo()  # Use real resource-supervised numerical workers.
    monkeypatch.setattr(artifacts, "ConfirmationRegistry",
                        lambda path: ConfirmationRegistry(tmp_path / "usage.sqlite3"))
    panel_path, source = tmp_path / "panel.json", tmp_path / "holdout.json"
    panel_path.write_text(json.dumps(panel_for(frozen).public()), encoding="utf-8")
    source.write_text(json.dumps(holdout() if valid_targets else holdout(lambda x: 20*x)), encoding="utf-8")
    args = [str(panel_path), "--holdout", str(source), "--study", "paired",
            "--output-root", str(tmp_path / "runs")]
    assert main(args) == expected_exit
    assert "不选择冠军" in capsys.readouterr().out
    assert main(args) == 1  # Same registry, even though another output run is created.


def test_cli_rejects_invalid_panel_without_tracebacks(tmp_path, capsys):
    from scripts.confirm_graph_panel import main
    source = tmp_path / "panel.json"
    source.write_text("broken", encoding="utf-8")
    assert main([str(source), "--holdout", str(tmp_path / "absent"), "--study", "paired"]) == 2
    error = capsys.readouterr().err
    assert "invalid_json" in error and "Traceback" not in error
