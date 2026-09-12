"""Serializable training worker used by the web process executor."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import joblib

from .integrated_pipeline import IntegratedPipeline


def run_training_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    records = payload.get("records") or []
    df = pd.DataFrame.from_records(records)
    if df.empty and payload.get("columns"):
        df = pd.DataFrame(columns=payload["columns"])
    targets = payload.get("target_cols") or []
    config = dict(payload.get("config") or {})
    raw_options = dict(payload.get("options") or {})
    artifact = Path(payload["artifact_path"])
    artifact.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def build(target):
        perf = config.get("performance", {}) if isinstance(config.get("performance"), dict) else {}
        return IntegratedPipeline(
            strategy_preference=raw_options.get("strategy_preference"), target_col=target,
            task_type=config.get("task_type"), model_keys=config.get("model_keys"),
            encoding=config.get("encoding", "auto"), feature_selection=config.get("feature_selection", "mi"),
            ensemble=config.get("ensemble", "weighted"), feature_engineering=config.get("feature_engineering", False),
            fold_type=config.get("fold_type", "default"), group_col=config.get("group_col"),
            pseudo_labeling=config.get("pseudo_labeling", False), pseudo_label_threshold=config.get("pseudo_label_threshold", .9),
            n_splits=int(config.get("n_splits", 5)), optimize_hyperparams=config.get("optimize_hyperparams", False),
            hyperparam_trials=int(config.get("hyperparam_trials", 20)), auto_decision_mode=config.get("auto_decision_mode", "balanced"),
            user_override_model=config.get("user_override_model"), auto_sample=config.get("auto_sample", True),
            max_samples=int(config.get("max_samples", 50000)), deep_learning=config.get("deep_learning"),
            optimizer=config.get("optimizer", "bayesian"), dim_reduction=config.get("dim_reduction", "none"),
            enable_kernel_approximation=perf.get("enable_kernel_approximation", config.get("enable_kernel_approximation", True)),
            enable_precomputed_kernel_cache=perf.get("enable_precomputed_kernel_cache", config.get("enable_precomputed_kernel_cache", True)),
            allow_disk_write=True,
        )

    results = []
    for target in (targets or [None]):
        result = build(target).run(df)
        results.append({"target": target, "result": result})
    joblib.dump({"results": results, "config": config}, artifact, compress=3)
    return {"artifact_path": str(artifact), "targets": targets, "elapsed": time.time() - started}
