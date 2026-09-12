import pytest

from core.parameter_uncertainty import (
    ParameterUncertaintyError,
    bootstrap_parameter_uncertainty,
    profile_likelihood_uncertainty,
)


def test_profile_likelihood_reports_bounded_level_set():
    result = profile_likelihood_uncertainty(
        lambda value: (value - 2.0) ** 2, [0, 1, 2, 3, 4], delta_threshold=1.0,
    )
    assert result["status"] == "assessed"
    assert result["minimum"]["parameter"] == 2.0
    assert result["interval"] == {"lower": 1.0, "upper": 3.0}


def test_profile_likelihood_keeps_objective_failure_unassessed():
    result = profile_likelihood_uncertainty(
        lambda value: (_ for _ in ()).throw(RuntimeError()), [0, 1, 2],
    )
    assert result["status"] == "not_assessed"
    assert result["evaluated_points"] == 0


def test_bootstrap_reports_empirical_parameter_spread():
    def fit(indices):
        return {"stable": 2.0, "noisy": 2.0 + (sum(indices) % 5) / 10}

    result = bootstrap_parameter_uncertainty(fit, sample_count=20, bootstrap_replicates=24)
    assert result["status"] == "assessed"
    assert result["successful_replicates"] == 24
    by_name = {item["parameter"]: item for item in result["parameters"]}
    assert by_name["stable"]["std"] == 0
    assert by_name["noisy"]["std"] > 0
    assert "not_posterior" in result["policy"]


def test_bootstrap_keeps_schema_and_fit_failures_explicit():
    def fit(indices):
        if sum(indices) % 3 == 0:
            raise RuntimeError("solver did not converge")
        return {"a": 1.0}

    result = bootstrap_parameter_uncertainty(fit, sample_count=12, bootstrap_replicates=16,
                                             max_failures=16)
    assert result["failed_replicates"] > 0
    assert all("reason" in failure for failure in result["failures"])

    calls = [0]
    def changing_schema(indices):
        calls[0] += 1
        return {"a": 1.0} if calls[0] == 1 else {"b": 1.0}

    schema = bootstrap_parameter_uncertainty(changing_schema, sample_count=8, bootstrap_replicates=8)
    assert schema["status"] in {"not_assessed", "few_successful_replicates"}
    assert any("schema_changed" in item["reason"] for item in schema["failures"])


def test_bootstrap_returns_not_assessed_when_all_fits_fail():
    result = bootstrap_parameter_uncertainty(lambda indices: (_ for _ in ()).throw(RuntimeError()),
                                             sample_count=8, bootstrap_replicates=8)
    assert result["status"] == "not_assessed"
    assert result["successful_replicates"] == 0


def test_bootstrap_rejects_ambiguous_parameter_names_in_fit_output():
    result = bootstrap_parameter_uncertainty(
        lambda indices: {"a": 1.0, " a ": 2.0}, sample_count=8, bootstrap_replicates=8,
    )
    assert result["status"] == "not_assessed"
    assert any("parameter_names_must_be_unique" in item["reason"] for item in result["failures"])


@pytest.mark.parametrize("kwargs", [
    {"sample_count": 3}, {"bootstrap_replicates": 7}, {"seed": -1},
    {"max_failures": -1},
])
def test_bootstrap_rejects_invalid_contract(kwargs):
    arguments = {"sample_count": 8}
    arguments.update(kwargs)
    with pytest.raises(ParameterUncertaintyError):
        bootstrap_parameter_uncertainty(lambda indices: {"a": 1.0}, **arguments)
