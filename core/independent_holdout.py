"""独立预测评分器。

评分进程只接收预测值和封存标签，不接收训练数据、模型对象或生成代码。
这不是操作系统级沙箱，也不能把公开标签变成真实未见题；它提供的是一个
可复现、可单独审计的第三方评分边界，避免训练器自己同时声明“预测正确”。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "mathmodel.independent-holdout-score/v1"


class IndependentHoldoutError(ValueError):
    pass


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def score_regression_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Score one prediction bundle without importing any model code."""
    if not isinstance(payload, Mapping) or set(payload) != {"case_id", "truth", "prediction"}:
        raise IndependentHoldoutError("prediction_bundle_fields_invalid")
    case_id = payload["case_id"]
    if type(case_id) is not str or not case_id.strip() or len(case_id) > 160:
        raise IndependentHoldoutError("case_id_invalid")
    try:
        truth = np.asarray(payload["truth"], dtype=float)
        prediction = np.asarray(payload["prediction"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise IndependentHoldoutError("prediction_values_non_numeric") from exc
    if truth.ndim != 1 or prediction.ndim != 1 or truth.shape != prediction.shape or truth.size == 0:
        raise IndependentHoldoutError("prediction_shape_invalid")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise IndependentHoldoutError("prediction_values_non_finite")
    error = prediction - truth
    mse = float(np.mean(np.square(error)))
    truth_mean = float(np.mean(truth))
    denominator = float(np.sum(np.square(truth - truth_mean)))
    r2 = 1.0 - float(np.sum(np.square(error))) / denominator if denominator > 0 else None
    return {
        "schema_version": SCHEMA,
        "case_id": case_id.strip(),
        "status": "scored",
        "row_count": int(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": math.sqrt(mse),
        "r2": r2,
        "truth_digest": _digest([float(item) for item in truth]),
        "prediction_digest": _digest([float(item) for item in prediction]),
        "evaluator": "independent_holdout_scorer_v1",
        "policy": "score_only_prediction_bundle; no_model_or_training_input",
    }


def score_regression_files(truth_path: str | Path, prediction_path: str | Path) -> dict[str, Any]:
    """Read separate JSON bundles; intended for a dedicated scorer process."""
    try:
        truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
        prediction = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentHoldoutError("score_input_unreadable") from exc
    if not isinstance(truth, Mapping) or not isinstance(prediction, Mapping):
        raise IndependentHoldoutError("score_input_must_be_object")
    if truth.get("case_id") != prediction.get("case_id"):
        raise IndependentHoldoutError("case_id_mismatch")
    return score_regression_payload({
        "case_id": truth.get("case_id"),
        "truth": truth.get("values"),
        "prediction": prediction.get("values"),
    })


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent regression holdout scorer")
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = score_regression_files(args.truth, args.prediction)
    except (OSError, IndependentHoldoutError) as exc:
        result = {"schema_version": SCHEMA, "status": "error", "error_code": str(exc)[:120]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["IndependentHoldoutError", "SCHEMA", "score_regression_files", "score_regression_payload"]
