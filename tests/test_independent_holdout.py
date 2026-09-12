import json
import subprocess
import sys

import pytest

from core.independent_holdout import score_regression_payload


def test_scorer_is_model_agnostic_and_reproducible():
    result = score_regression_payload({"case_id": "case-1", "truth": [1, 2, 3], "prediction": [1, 3, 2]})
    assert result["status"] == "scored"
    assert result["rmse"] == pytest.approx((2 / 3) ** 0.5)
    assert result["evaluator"] == "independent_holdout_scorer_v1"


def test_scorer_cli_reads_separate_truth_and_prediction_files(tmp_path):
    truth = tmp_path / "truth.json"
    prediction = tmp_path / "prediction.json"
    output = tmp_path / "score.json"
    truth.write_text(json.dumps({"case_id": "case-1", "values": [1, 2, 3]}), encoding="utf-8")
    prediction.write_text(json.dumps({"case_id": "case-1", "values": [1, 2, 2]}), encoding="utf-8")
    completed = subprocess.run([sys.executable, "-m", "core.independent_holdout",
                                "--truth", str(truth), "--prediction", str(prediction),
                                "--output", str(output)], check=False, capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "scored"


def test_scorer_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="prediction_shape_invalid"):
        score_regression_payload({"case_id": "case-1", "truth": [1, 2], "prediction": [1]})
