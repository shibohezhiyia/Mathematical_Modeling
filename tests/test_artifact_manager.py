import hashlib
import json

import pytest

from core.artifact_manager import (
    ARTIFACT_SCHEMA_VERSION,
    CACHE_SCHEMA_VERSION,
    RunArtifactManager,
)


def test_versioned_manifest_records_strict_paths_hashes_and_formats(tmp_path):
    manager = RunArtifactManager(tmp_path, run_id="run_test_001")
    evidence = manager.write_json(
        "evidence.bundle", "evidence", "evidence_bundle.json",
        {"claim": "supported"}, required=True,
    )
    report = manager.write_text(
        "report.argument", "reports", "mathematical_argument.md", "# Argument\n",
        media_type="text/markdown; charset=utf-8", required=True,
    )
    manager.finalize()

    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["directories"] == {
        "cache": "cache", "charts": "charts", "evidence": "evidence",
        "logs": "logs", "reports": "reports", "temp": "temp",
    }
    records = {item["id"]: item for item in manifest["artifacts"]}
    assert records["evidence.bundle"]["relative_path"] == "evidence/evidence_bundle.json"
    assert records["report.argument"]["relative_path"] == "reports/mathematical_argument.md"
    assert records["evidence.bundle"]["sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert records["report.argument"]["size_bytes"] == report.stat().st_size
    assert not list((tmp_path / "temp").glob("*.tmp"))


def test_cache_is_deterministic_disposable_and_safely_clearable(tmp_path):
    manager = RunArtifactManager(tmp_path, run_id="run_cache_test")
    key_a = manager.cache_key("interactions", {"datasets": ["a", "b"]}, "2")
    key_b = manager.cache_key("interactions", {"datasets": ["a", "b"]}, "2")
    assert key_a == key_b
    cache_path = manager.write_cache(
        "interactions", {"datasets": ["a", "b"]}, {"score": 0.8}, version="2"
    )
    evidence = manager.write_json(
        "evidence.bundle", "evidence", "evidence_bundle.json", {"keep": True}
    )
    manager.finalize()

    cache_manifest = json.loads(
        (tmp_path / "cache" / "cache_manifest.json").read_text("utf-8")
    )
    assert cache_manifest["schema_version"] == CACHE_SCHEMA_VERSION
    assert cache_manifest["entries"][0]["relative_path"].startswith("cache/interactions/")
    assert cache_path.is_file()

    reopened = RunArtifactManager.open_existing(tmp_path)
    preview = reopened.clear_cache(dry_run=True)
    assert preview["deleted_files"] == 1
    assert cache_path.is_file()
    summary = reopened.clear_cache()
    assert summary["deleted_files"] == 1
    assert not cache_path.exists()
    assert evidence.is_file()
    assert (tmp_path / "reports").is_dir()
    assert (tmp_path / "cache" / "cache_manifest.json").is_file()
    manifest = reopened.read_manifest()
    assert manifest["cleanup_history"][-1]["categories"] == ["cache"]


def test_paths_and_cleanup_cannot_escape_or_delete_durable_results(tmp_path):
    manager = RunArtifactManager(tmp_path)
    with pytest.raises(ValueError):
        manager.path("evidence", "../outside.json")
    with pytest.raises(ValueError):
        manager.path("unknown", "file.json")
    with pytest.raises(ValueError):
        manager.cleanup_disposable(("evidence",))
    with pytest.raises(ValueError):
        manager.write_json(
            "bad.cache", "cache", "entry.json", {}, disposable=False
        )
