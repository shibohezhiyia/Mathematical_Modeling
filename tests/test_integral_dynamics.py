"""Temporal isolation and honest evaluation of discovered dynamics."""

import numpy as np
import pandas as pd
import pytest

from core.integral_dynamics import discover_integral_dynamics, prepare_dynamics


def oscillator(n=300):
    time = np.arange(n) * 0.2
    return pd.DataFrame({
        "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(time, unit="D"),
        "state": np.sin(0.2 * time),
        "driver": np.cos(0.2 * time),
    })


def discover(frame, **kwargs):
    return discover_integral_dynamics(
        frame, time_column="date", target_column="state", **kwargs,
    )


def test_final_test_values_cannot_change_model_or_rollout():
    original = oscillator()
    changed = original.copy()
    changed.loc[240:, ["state", "driver"]] += 40
    first, second = discover(original), discover(changed)
    assert first is not None and second is not None
    for key in ("state_columns", "standardization", "selected_alpha",
                "active_terms", "system_equations", "candidate_search", "training_fingerprint"):
        assert first[key] == second[key], key
    np.testing.assert_allclose(first["test_rollout_prediction"], second["test_rollout_prediction"])
    assert first["trajectory_test"]["metrics"]["r2"] > 0.98
    assert second["trajectory_test"]["status"] == "fail"


def test_selection_data_cannot_change_training_preprocessing():
    original = oscillator()
    for index in range(5):
        original[f"extra{index}"] = np.random.default_rng(index).normal(size=len(original))
    changed = original.copy()
    changed.loc[180:, "extra4"] = changed.loc[180:, "state"] * 1000
    changed.loc[180:, "state"] *= 100
    first = prepare_dynamics(original, time_column="date", target_column="state")
    second = prepare_dynamics(changed, time_column="date", target_column="state")
    assert first.state_columns == second.state_columns
    np.testing.assert_array_equal(first.center, second.center)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.states[:180], second.states[:180])
    assert first.training_fingerprint == second.training_fingerprint


def test_missing_final_targets_do_not_move_split_or_change_fit():
    original = oscillator()
    changed = original.copy()
    changed.loc[240:, "state"] = np.nan
    first, second = discover(original), discover(changed)
    assert first["split"] == second["split"]
    assert first["active_terms"] == second["active_terms"]
    assert first["candidate_search"] == second["candidate_search"]
    assert second["test_integral_metrics"]["n"] == 0
    assert second["trajectory_test"]["status"] == "not_assessed"
    assert second["credibility_audit"]["status"] != "pass"


def test_imputation_never_reads_across_training_boundary():
    original = oscillator()
    original.loc[178:179, "driver"] = np.nan
    changed = original.copy()
    changed.loc[180, "driver"] = 1e6
    first = prepare_dynamics(original, time_column="date", target_column="state")
    second = prepare_dynamics(changed, time_column="date", target_column="state")
    np.testing.assert_array_equal(first.states[:180], second.states[:180])
    assert first.states[179, 1] == first.states[177, 1]


def test_explicit_target_outside_column_budget_is_retained():
    frame = oscillator()
    for index in range(12):
        frame[f"other{index}"] = np.arange(len(frame)) + index
    frame = frame[["date", *[f"other{i}" for i in range(12)], "driver", "state"]]
    result = discover(frame)
    assert result is not None
    assert result["state_columns"][0] == "state"


def test_partition_windows_do_not_overlap_boundaries():
    prepared = prepare_dynamics(oscillator(), time_column="date", target_column="state")
    for start, end, mask in (
        (0, prepared.train_end, prepared.train_mask),
        (prepared.train_end, prepared.search_end, prepared.search_mask),
        (prepared.search_end, len(prepared.states), prepared.test_mask),
    ):
        assert mask.any()
        assert np.all(prepared.starts[mask] >= start)
        assert np.all(prepared.starts[mask] + prepared.window < end)


def test_cumulative_quadrature_matches_direct_windows_on_irregular_times():
    from core.integral_dynamics import _library

    frame = oscillator()
    times = np.cumsum(np.random.default_rng(5).uniform(0.1, 0.3, len(frame)))
    frame["date"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(times, unit="D")
    prepared = prepare_dynamics(frame, time_column="date", target_column="state")
    library = _library(prepared.states)
    for index in [0, 50, 160, 240]:
        stop = index + prepared.window
        expected = np.sum(0.5 * (library[index:stop] + library[index + 1:stop + 1])
                          * np.diff(prepared.elapsed[index:stop + 1])[:, None], axis=0)
        np.testing.assert_allclose(prepared.design[index], expected, rtol=1e-10, atol=1e-12)


def test_nonfinite_predictions_are_not_silently_dropped_from_a_pass():
    from core.integral_dynamics import _metrics, _metric_status

    truth = np.arange(20, dtype=float)
    predicted = truth.copy()
    predicted[-1] = np.inf
    metrics = _metrics(truth, predicted, np.zeros_like(truth))
    assert metrics["r2"] == 1.0
    assert metrics["invalid_predictions"] == 1
    assert _metric_status(metrics) == "fail"


def test_oscillator_has_separate_integral_and_autonomous_tests():
    result = discover(oscillator())
    assert result["test_integral_metrics"]["r2"] > 0.99
    assert result["trajectory_test"]["metrics"]["r2"] > 0.98
    assert result["trajectory_test"]["uses_future_observations"] is False
    assert result["trajectory_test"]["initial_row"] == 239
    assert len(result["system_equations"]) == 2
    assert result["evaluation_protocol"] == "train_search_locked_test"
    assert result["metrics"]["validation_r2"] == result["test_integral_metrics"]["r2"]
    assert result["selected_alpha"] == result["candidate_search"][
        np.argmin([item["selection_objective"] for item in result["candidate_search"]])
    ]["alpha"]


def test_good_integral_fit_cannot_hide_unpredicted_future_driver():
    frame = oscillator()
    frame.loc[240:, "driver"] += 3
    driver = frame["driver"].to_numpy()
    frame["state"] = np.r_[0, np.cumsum(0.2 * 0.2 * (driver[1:] + driver[:-1]) / 2)]
    result = discover(frame)
    assert result["test_integral_metrics"]["r2"] > 0.98
    assert result["trajectory_test"]["status"] == "fail"
    assert result["credibility_audit"]["status"] == "fail"


def test_rollout_exhaustion_is_not_a_pass():
    result = discover(oscillator(), max_rollout_evaluations=1)
    assert result["trajectory_test"]["status"] == "fail"
    assert result["trajectory_test"]["reason"] == "evaluation_budget_exhausted"
    assert result["credibility_audit"]["status"] == "fail"


def test_rollout_deadline_is_reported_without_false_success(monkeypatch):
    from itertools import count
    import core.integral_dynamics as dynamics

    ticks = count()
    monkeypatch.setattr(dynamics, "monotonic", lambda: next(ticks))
    result = discover(oscillator(), rollout_timeout_seconds=0.5)
    assert result["trajectory_test"]["status"] == "fail"
    assert result["trajectory_test"]["reason"] == "rollout_deadline_exceeded"


def test_parameter_search_interface_only_receives_train_and_search(monkeypatch):
    import core.integral_dynamics as dynamics

    original_fit = dynamics._fit_system
    prepared = prepare_dynamics(oscillator(), time_column="date", target_column="state")

    def checked_fit(x_train, y_train, x_search, y_search, random_state):
        np.testing.assert_array_equal(x_train, prepared.design[prepared.train_mask])
        np.testing.assert_array_equal(y_train, prepared.response[prepared.train_mask])
        np.testing.assert_array_equal(x_search, prepared.design[prepared.search_mask])
        np.testing.assert_array_equal(y_search, prepared.response[prepared.search_mask])
        return original_fit(x_train, y_train, x_search, y_search, random_state)

    monkeypatch.setattr(dynamics, "_fit_system", checked_fit)
    assert discover(oscillator())["trajectory_test"]["uses_future_observations"] is False


def test_missing_initial_state_cannot_be_filled_from_test():
    frame = oscillator()
    frame.loc[239, "driver"] = np.nan
    result = discover(frame)
    assert result["trajectory_test"]["status"] == "not_assessed"
    assert result["trajectory_test"]["reason"] == "initial_state_not_observed"


def test_noise_does_not_become_a_validated_mechanism():
    frame = oscillator()
    frame[["state", "driver"]] = np.random.default_rng(2).normal(size=(300, 2))
    result = discover(frame)
    assert result["credibility_audit"]["status"] == "fail"


@pytest.mark.parametrize("n", [50, 80, 300])
def test_small_series_and_single_state_are_supported(n):
    frame = oscillator(n).drop(columns="driver")
    frame["state"] = 2 * np.exp(-np.arange(n) * 0.003)
    result = discover(frame)
    assert result is not None
    assert result["trajectory_test"]["metrics"]["r2"] > 0.97


def test_duplicate_times_invalid_values_and_resource_cap():
    frame = oscillator(5100)
    frame = pd.concat([frame.iloc[:2], frame], ignore_index=True)
    frame.loc[0, "date"] = pd.NaT
    frame.loc[1, "driver"] = np.inf
    prepared = prepare_dynamics(frame, time_column="date", target_column="state")
    assert len(prepared.states) == 5000
    assert np.all(np.diff(prepared.elapsed) > 0)
    assert np.isfinite(prepared.center).all()


def test_too_short_or_unobserved_training_target_is_not_fit():
    assert discover(oscillator(20)) is None
    frame = oscillator()
    frame.loc[:179, "state"] = np.nan
    assert discover(frame) is None


def test_timestamp_order_and_duplicate_index_do_not_change_results():
    original = oscillator()
    shuffled = original.sample(frac=1, random_state=7)
    shuffled.index = np.zeros(len(shuffled), dtype=int)
    first, second = discover(original), discover(shuffled)
    assert first["training_fingerprint"] == second["training_fingerprint"]
    assert first["system_equations"] == second["system_equations"]


def test_missing_test_does_not_leak_into_json_metrics():
    import json
    from core.modeling_assistant import _plain

    frame = oscillator()
    frame.loc[240:, ["state", "driver"]] = np.nan
    result = _plain(discover(frame))
    json.dumps(result, allow_nan=False)
    assert result["credibility_audit"]["status"] != "pass"
