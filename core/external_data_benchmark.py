"""Automated, reproducible scoring for the downloaded public data arm.

This runner evaluates predictive execution only.  It keeps source and split
fingerprints in the report, uses a scorer independent from the model fitting
code, and labels the result descriptive because the labels are public.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .external_dataset_catalog import load_catalog


class ExternalBenchmarkError(ValueError):
    pass


_LEAKAGE_GUARDS: dict[str, tuple[str, ...]] = {
    # cnt is defined as casual + registered; instant is a row identifier.
    "uci-bike-sharing": ("casual", "registered", "instant"),
}


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_split(train: pd.DataFrame, test: pd.DataFrame) -> str:
    payload = ("|".join(map(str, train.index)) + "#" + "|".join(map(str, test.index))).encode()
    return _digest_bytes(payload)


def _read_member(path: Path, member: str, *, sep: str = ",", decimal: str = ".") -> pd.DataFrame:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if member not in names:
                raise ExternalBenchmarkError(f"member_not_found:{member}")
            raw = archive.read(member)
    except zipfile.BadZipFile as exc:
        raise ExternalBenchmarkError("invalid_zip") from exc
    return pd.read_csv(BytesIO(raw), sep=sep, decimal=decimal, encoding="latin1")


def _load_cases(zip_path: Path, dataset_id: str) -> list[tuple[str, pd.DataFrame, str, bool]]:
    """Return (case id, frame, target, time_ordered)."""
    if dataset_id == "uci-wine-quality":
        return [(f"{dataset_id}-red", _read_member(zip_path, "winequality-red.csv", sep=";"), "quality", False),
                (f"{dataset_id}-white", _read_member(zip_path, "winequality-white.csv", sep=";"), "quality", False)]
    if dataset_id == "uci-bike-sharing":
        return [(dataset_id, _read_member(zip_path, "day.csv"), "cnt", True)]
    if dataset_id == "uci-air-quality":
        frame = _read_member(zip_path, "AirQualityUCI.csv", sep=";", decimal=",")
        return [(dataset_id, frame, "CO(GT)", True)]
    if dataset_id == "uci-airfoil-self-noise":
        # UCI's official archive stores a whitespace-delimited .dat file with
        # five physical inputs and one sound-pressure target (1503 rows).
        try:
            with zipfile.ZipFile(zip_path) as archive:
                members = [name for name in archive.namelist()
                           if name.lower().endswith("airfoil_self_noise.dat")]
                if not members:
                    raise ExternalBenchmarkError("airfoil_member_not_found")
                raw = archive.read(members[0])
        except zipfile.BadZipFile as exc:
            raise ExternalBenchmarkError("invalid_zip") from exc
        columns = ["frequency", "attack-angle", "chord-length",
                   "free-stream-velocity", "suction-side-displacement-thickness",
                   "scaled-sound-pressure"]
        frame = pd.read_csv(BytesIO(raw), sep=r"\s+", header=None, names=columns,
                            engine="python", encoding="latin1")
        return [(dataset_id, frame, "scaled-sound-pressure", False)]
    if dataset_id == "uci-concrete-strength":
        try:
            with zipfile.ZipFile(zip_path) as archive:
                raw = archive.read("Concrete_Data.xls")
        except (KeyError, zipfile.BadZipFile) as exc:
            raise ExternalBenchmarkError("concrete_member_not_found") from exc
        return [(dataset_id, pd.read_excel(BytesIO(raw)), "Concrete compressive strength", False)]
    raise ExternalBenchmarkError(f"unsupported_dataset:{dataset_id}")


def _prepare(frame: pd.DataFrame, target: str, *, drop_columns: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    clean = frame.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    target_name = next((column for column in clean.columns if column == target), None)
    if target_name is None and target == "Concrete compressive strength":
        target_name = next((column for column in clean.columns
                            if column.lower().startswith("concrete compressive strength")), None)
    if target_name is None:
        raise ExternalBenchmarkError(f"target_not_found:{target}")
    raw_target = pd.to_numeric(clean.pop(target_name), errors="coerce")
    target_missing_sentinel = int((raw_target == -200).sum())
    y = raw_target.replace([np.inf, -np.inf, -200], np.nan)
    dropped = [column for column in drop_columns if column in clean.columns]
    if dropped:
        clean = clean.drop(columns=dropped)
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf, -200], np.nan)
    valid = y.notna()
    dropped_target_rows = int((~valid).sum())
    clean, y = clean.loc[valid], y.loc[valid]
    # Drop columns with no usable signal; the scorer never guesses text encodings.
    all_missing = [column for column in clean.columns if not clean[column].notna().any()]
    clean = clean.loc[:, clean.notna().any(axis=0)]
    if len(clean) < 40 or clean.shape[1] < 1:
        raise ExternalBenchmarkError("insufficient_numeric_data")
    audit = {"target_column": target_name, "target_missing_sentinel_rows": target_missing_sentinel,
             "dropped_target_rows": dropped_target_rows, "dropped_leakage_columns": dropped,
             "dropped_all_missing_columns": all_missing}
    return clean.reset_index(drop=True), y.reset_index(drop=True), audit


def _split(X: pd.DataFrame, y: pd.Series, *, ordered: bool, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    cut = max(1, min(n - 1, int(math.floor(n * 0.8))))
    if ordered:
        train_idx, test_idx = np.arange(cut), np.arange(cut, n)
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(n)
        train_idx, test_idx = shuffled[:cut], shuffled[cut:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _fit_score(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
               model: Any) -> dict[str, float]:
    pipeline = make_pipeline(SimpleImputer(strategy="median"), model)
    pipeline.fit(X_train, y_train)
    prediction = np.asarray(pipeline.predict(X_test), dtype=float)
    if prediction.shape != y_test.shape or not np.isfinite(prediction).all():
        raise ExternalBenchmarkError("non_finite_prediction")
    return {"mae": float(mean_absolute_error(y_test, prediction)),
            "rmse": float(math.sqrt(mean_squared_error(y_test, prediction))),
            "r2": float(r2_score(y_test, prediction))}


def _fit_predict(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                 model: Any) -> np.ndarray:
    """Fit without passing test labels into the modeling callable."""
    pipeline = make_pipeline(SimpleImputer(strategy="median"), model)
    pipeline.fit(X_train, y_train)
    prediction = np.asarray(pipeline.predict(X_test), dtype=float)
    if prediction.ndim != 1 or not np.isfinite(prediction).all():
        raise ExternalBenchmarkError("non_finite_prediction")
    return prediction


def _score_in_independent_process(case_id: str, truth: pd.Series, prediction: np.ndarray) -> dict[str, Any]:
    """Send only truth/prediction vectors to a separate scorer process."""
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mathmodel_score_") as directory:
        folder = Path(directory)
        truth_path, prediction_path, output_path = folder / "truth.json", folder / "prediction.json", folder / "score.json"
        truth_path.write_text(json.dumps({"case_id": case_id, "values": [float(value) for value in truth]},
                                         ensure_ascii=False), encoding="utf-8")
        prediction_path.write_text(json.dumps({"case_id": case_id, "values": [float(value) for value in prediction]},
                                              ensure_ascii=False), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "core.independent_holdout", "--truth", str(truth_path),
                 "--prediction", str(prediction_path), "--output", str(output_path)],
                cwd=str(root), capture_output=True, text=True, timeout=30, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ExternalBenchmarkError("independent_scorer_unavailable") from exc
        if completed.returncode != 0 or not output_path.is_file():
            raise ExternalBenchmarkError("independent_scorer_failed")
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExternalBenchmarkError("independent_score_unreadable") from exc
        if result.get("status") != "scored":
            raise ExternalBenchmarkError("independent_score_rejected")
        return result


def _paired_statistics(effects: list[float], *, seed: int) -> dict[str, Any]:
    """Compute bounded task-level uncertainty without mixing metric units."""
    if not effects:
        return {"sample_count": 0, "mean": None, "median": None,
                "bootstrap_ci_95": None, "permutation_p_two_sided": None,
                "status": "not_assessed"}
    values = np.asarray(effects, dtype=float)
    rng = np.random.default_rng(seed)
    reps = min(10_000, max(2_000, 400 * len(values)))
    indices = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[indices].mean(axis=1)
    observed = abs(float(values.mean()))
    if len(values) <= 20:
        signs = np.asarray([[(1 if (mask >> i) & 1 else -1) for i in range(len(values))]
                            for mask in range(1 << len(values))], dtype=float)
        null_means = (signs * values).mean(axis=1)
        p_value = float(np.mean(np.abs(null_means) >= observed - 1e-15))
        permutation_method = "exact_sign_flip"
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(min(20_000, reps), len(values)))
        null_means = (signs * values).mean(axis=1)
        p_value = float((np.count_nonzero(np.abs(null_means) >= observed) + 1) /
                        (len(null_means) + 1))
        permutation_method = "monte_carlo_sign_flip"
    return {"sample_count": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)),
            "bootstrap_ci_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "bootstrap_replicates": int(reps), "permutation_p_two_sided": p_value,
            "permutation_method": permutation_method, "seed": int(seed),
            "status": "descriptive_public_labels"}


def run_external_data_benchmark(catalog_path: str | Path = "examples/external_dataset_catalog.json",
                                data_root: str | Path = "data/external", *, seed: int = 20260910,
                                max_rows: int = 100_000, isolated_scoring: bool = True,
                                include_current_system: bool = False,
                                current_system_models: tuple[str, ...] | None = None) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    root = Path(data_root)
    rows: list[dict[str, Any]] = []
    for item in catalog["datasets"]:
        path = root / str(item.get("filename") or (item["id"] + ".zip"))
        if not path.is_file():
            rows.append({"id": item["id"], "status": "failed", "error_code": "dataset_file_missing"})
            continue
        actual_digest = _digest_bytes(path.read_bytes())
        if item.get("sha256") and actual_digest != item["sha256"].lower():
            rows.append({"id": item["id"], "status": "failed", "error_code": "source_hash_mismatch"})
            continue
        try:
            cases = _load_cases(path, item["id"])
        except Exception as exc:
            rows.append({"id": item["id"], "status": "failed", "error_code": type(exc).__name__})
            continue
        for case_id, frame, target, ordered in cases:
            try:
                X, y, data_audit = _prepare(frame, target, drop_columns=_LEAKAGE_GUARDS.get(item["id"], ()))
                if len(X) > max_rows:
                    raise ExternalBenchmarkError("row_budget_exceeded")
                X_train, X_test, y_train, y_test = _split(X, y, ordered=ordered, seed=seed)
                if isolated_scoring:
                    baseline_prediction = _fit_predict(X_train, X_test, y_train,
                                                       make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
                    treatment_prediction = _fit_predict(
                        X_train, X_test, y_train,
                        ExtraTreesRegressor(n_estimators=120, random_state=seed, n_jobs=1,
                                            min_samples_leaf=2),
                    )
                    baseline = _score_in_independent_process(case_id + ":baseline", y_test, baseline_prediction)
                    treatment = _score_in_independent_process(case_id + ":treatment", y_test, treatment_prediction)
                else:
                    baseline = _fit_score(X_train, X_test, y_train, y_test,
                                          make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
                    treatment = _fit_score(X_train, X_test, y_train, y_test,
                                           ExtraTreesRegressor(n_estimators=120, random_state=seed, n_jobs=1,
                                                               min_samples_leaf=2))
                row = {"id": case_id, "dataset_id": item["id"], "status": "completed",
                             "source_sha256": actual_digest, "split_sha256": _digest_split(X_train, X_test),
                             "train_rows": len(X_train), "test_rows": len(X_test),
                             "feature_count": X.shape[1], "target": target, "ordered_split": ordered,
                             "data_audit": data_audit,
                             "baseline": baseline, "treatment": treatment}
                if include_current_system:
                    try:
                        from .system_benchmark import compare_frozen_systems
                        row["current_system_comparison"] = compare_frozen_systems(
                            X_train, y_train, X_test, y_test, random_state=seed,
                            model_keys=current_system_models, n_splits=3,
                        )
                    except Exception as exc:
                        row["current_system_comparison"] = {
                            "schema_version": "mathmodel.system-benchmark/v1",
                            "status": "not_assessed", "failure_code": type(exc).__name__,
                        }
                rows.append(row)
            except Exception as exc:
                rows.append({"id": case_id, "dataset_id": item["id"], "status": "failed",
                             "source_sha256": actual_digest, "error_code": type(exc).__name__})
    completed = [row for row in rows if row.get("status") == "completed"]
    if completed:
        # Raw RMSE cannot be averaged across datasets with different units.
        # Normalize each task against its own baseline before pairing.
        effects = [row["treatment"]["rmse"] / row["baseline"]["rmse"] - 1.0
                   for row in completed if row["baseline"]["rmse"] > 0]
        effect = float(np.mean(effects)) if effects else None
    else:
        effect = None
    statistics = _paired_statistics(effects, seed=seed) if completed else _paired_statistics([], seed=seed)
    current_effects = [row["current_system_comparison"].get("relative_rmse_effect_current_minus_baseline")
                       for row in completed if isinstance(row.get("current_system_comparison"), Mapping)
                       and row["current_system_comparison"].get("status") == "completed"
                       and row["current_system_comparison"].get("relative_rmse_effect_current_minus_baseline") is not None]
    return {"schema_version": "mathmodel.external-data-benchmark/v1", "status": "completed",
            "evaluation_status": "descriptive_public_labels", "seed": seed,
            "data_contract": {"sentinel_values": [-200],
                              "leakage_guards": {key: list(value) for key, value in _LEAKAGE_GUARDS.items()},
                              "time_split": "ordered",
                              "scorer": "independent_holdout_scorer_v1" if isolated_scoring else "in_process_legacy"},
            "case_count": len(rows), "completed_count": len(completed),
            "failed_count": len(rows) - len(completed), "rows": rows,
            "paired_relative_rmse_effect_treatment_minus_baseline": effect,
            "median_relative_rmse_effect_treatment_minus_baseline": (
                float(np.median(effects)) if completed and effects else None
            ),
            "paired_relative_rmse_statistics": statistics,
            "current_system": {
                "enabled": bool(include_current_system),
                "completed_count": len(current_effects),
                "relative_rmse_effect_mean": (float(np.mean(current_effects)) if current_effects else None),
                "relative_rmse_effects": [float(value) for value in current_effects],
                "policy": "paired_frozen_split; current_system_comparison_is_descriptive_public_label_evidence",
            },
            "policy": "public_labels; external_data_evidence_only; no_real_unseen_claim; scorer_separate_process" if isolated_scoring
            else "public_labels; external_data_evidence_only; no_real_unseen_claim"}


__all__ = ["ExternalBenchmarkError", "run_external_data_benchmark"]
