import numpy as np
import pytest

from core.identifiability import IdentifiabilityError, assess_local_identifiability


def test_local_identifiability_detects_full_rank_and_redundant_parameters():
    full = assess_local_identifiability(
        lambda p: np.array([p["a"], p["b"], p["a"] + p["b"]]), {"a": 1.0, "b": 2.0}
    )
    redundant = assess_local_identifiability(
        lambda p: np.array([p["a"] + p["b"], 2 * (p["a"] + p["b"])]), {"a": 1.0, "b": 2.0}
    )
    assert full["status"] == "locally_identified"
    assert redundant["status"] == "weakly_identified"
    assert redundant["jacobian_rank"] == 1


def test_local_identifiability_uses_one_sided_bound_and_respects_budget():
    result = assess_local_identifiability(lambda p: [p["x"] ** 2], {"x": 0}, bounds={"x": [0, 2]})
    assert result["status"] == "locally_identified"
    assert result["difference_methods"] == ["forward"]
    limited = assess_local_identifiability(lambda p: [p["x"]], {"x": 1}, max_evaluations=1)
    assert limited["status"] == "not_assessed"


def test_local_identifiability_rejects_invalid_prediction_contract():
    with pytest.raises(IdentifiabilityError, match="numeric_array"):
        assess_local_identifiability(lambda p: "bad", {"x": 1})
