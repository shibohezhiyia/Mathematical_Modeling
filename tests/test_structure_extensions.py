import pytest

from core.structure_extensions import StructureExtensionError, build_pde_library_contract, build_ude_contract, fit_ude_correction, screen_interactions


def test_structure_extensions_keep_interactions_noncausal_and_build_typed_contracts():
    result = screen_interactions([[1, 2, 0], [2, 4, 1], [3, 6, 2], [4, 8, 3]], ["x", "y", "z"], threshold=0.9)
    assert result["edges"][0]["source"] == "x"
    assert "not_causal" in result["policy"]
    ude = build_ude_contract(["x"], ["-k*x"])
    assert ude["status"] == "typed_not_fitted"
    pde = build_pde_library_contract([4, 4], ["x", "t"])
    assert "d1_x(field)" in pde["terms"]


def test_structure_extensions_reject_unbounded_or_invalid_contracts():
    with pytest.raises(StructureExtensionError):
        screen_interactions([[1, 2], [2, 4]], ["x", "y"])
    with pytest.raises(StructureExtensionError):
        build_ude_contract(["x", "y"], ["f"])
    with pytest.raises(StructureExtensionError):
        build_pde_library_contract([2, 4], ["x", "t"])


def test_ude_fit_reports_conditioning_and_rejects_invalid_condition_limit():
    fit = fit_ude_correction(
        [1, 2, 3, 4, 5, 6, 7, 8], [0] * 8,
        [[1], [2], [3], [4], [5], [6], [7], [8]],
        max_condition_number=1e8,
    )
    assert fit["design_condition_number"] >= 1
    assert fit["stability_status"] == "assessed"
    with pytest.raises(StructureExtensionError, match="ude_condition_limit_invalid"):
        fit_ude_correction([1] * 8, [0] * 8, [[1]] * 8, max_condition_number=1)
