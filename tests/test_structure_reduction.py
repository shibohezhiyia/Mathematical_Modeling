import pytest

from core.structure_reduction import StructureReductionError, analyze_design_structure


def test_structure_reduction_detects_sparse_blocks_and_symmetry():
    result = analyze_design_structure([
        [1, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3],
    ])
    assert result["sparse_candidate"] is True
    assert result["separable_candidate"] is True
    assert result["symmetry_groups"][0] == [0, 1]
    assert "symmetry_reuse_after_certificate" in result["plan"]
    assert "full_cold_comparison" in result["policy"]


def test_structure_reduction_rejects_ragged_or_nonfinite_input():
    with pytest.raises(StructureReductionError, match="rectangular"):
        analyze_design_structure([[1], [1, 2]])
    with pytest.raises(StructureReductionError, match="finite"):
        analyze_design_structure([[float("nan")]])
