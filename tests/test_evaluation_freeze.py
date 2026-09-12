from pathlib import Path

from core.evaluation_freeze import EvaluationFreezeError, create_evaluation_freeze, verify_evaluation_freeze


def test_freeze_records_source_digest_and_verifies(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "model.py").write_text("x = 1\n", encoding="utf-8")
    freeze = create_evaluation_freeze(
        tmp_path, protocol_id="pilot-v1", seed=7,
        budget={"max_seconds": 10, "max_candidates": 2}, methods=["baseline", "treatment"],
    )
    assert freeze["status"] == "frozen"
    assert len(freeze["files"]) == 1
    assert verify_evaluation_freeze(tmp_path, freeze)["status"] == "verified"

    (tmp_path / "core" / "model.py").write_text("x = 2\n", encoding="utf-8")
    verified = verify_evaluation_freeze(tmp_path, freeze)
    assert verified["status"] == "mismatch" and verified["mismatch_count"] == 1


def test_freeze_excludes_data_and_workspace(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "workspace").mkdir()
    (tmp_path / "core" / "model.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "data" / "private.txt").write_text("secret", encoding="utf-8")
    freeze = create_evaluation_freeze(
        tmp_path, protocol_id="pilot-v1", seed=7,
        budget={"max_seconds": 10}, methods=["baseline", "treatment"],
    )
    assert all(not row["path"].startswith(("data/", "workspace/")) for row in freeze["files"])


def test_freeze_rejects_duplicate_methods(tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "model.py").write_text("x = 1\n", encoding="utf-8")
    try:
        create_evaluation_freeze(tmp_path, protocol_id="pilot-v1", seed=7,
                                 budget={"max_seconds": 10}, methods=["baseline", "baseline"])
    except EvaluationFreezeError as exc:
        assert str(exc) == "methods_invalid"
    else:
        raise AssertionError("duplicate methods must be rejected")
