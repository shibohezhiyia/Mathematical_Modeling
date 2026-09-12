import pytest

from core.execution_manifest import ExecutionManifestError, build_execution_manifest, compare_execution_manifests


def test_execution_manifest_captures_reproducibility_metadata_without_raw_code():
    result = build_execution_manifest(run_id="r1", input_hash="a" * 64, random_seed=42,
                                      dependencies={"numpy": "2.0"}, code="x + 1", log_digest="b" * 64)
    assert result["code_sha256"]
    assert "x + 1" not in str(result)
    assert len(result["manifest_digest"]) == 64


def test_execution_manifest_rejects_invalid_input_hash():
    with pytest.raises(ExecutionManifestError):
        build_execution_manifest(run_id="r1", input_hash="bad", random_seed=None, dependencies={"numpy": "2"})


def test_execution_manifest_comparison_separates_preconditions_from_output_proof():
    left = build_execution_manifest(run_id="r1", input_hash="a" * 64, random_seed=1,
                                    dependencies={"numpy": "2"})
    right = build_execution_manifest(run_id="r2", input_hash="a" * 64, random_seed=1,
                                     dependencies={"numpy": "2"})
    assert compare_execution_manifests(left, right)["compatible"] is True
    changed = build_execution_manifest(run_id="r3", input_hash="b" * 64, random_seed=1,
                                       dependencies={"numpy": "2"})
    result = compare_execution_manifests(left, changed)
    assert result["compatible"] is False and "input_hash" in result["differences"]
    assert "output_equivalence" in result["policy"]
