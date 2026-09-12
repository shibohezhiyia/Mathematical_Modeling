import numpy as np
import pytest

from core.pde_discovery import discover_1d_pde, discover_2d_pde, discover_3d_pde


def test_pde_discovery_returns_holdout_and_boundary_residuals():
    times = np.linspace(0.0, 2.0, 20)
    coordinates = np.linspace(0.0, 1.0, 10)
    field = np.array([[np.exp(-t) * np.sin(x) for x in coordinates] for t in times])
    result = discover_1d_pde(times, coordinates, field, residual_tolerance=2.0, boundary_tolerance=2.0)
    assert result["status"] in {"candidate_found", "candidate_rejected_by_validation"}
    assert result["validation_time_rows"] > 0
    assert "boundary_rmse" in result
    assert result["forward_rollout_status"] in {"rollout_completed", "rejected_nonfinite_rollout"}
    assert "forward_stability" in result
    assert result["grid_convergence"]["status"] == "assessed"
    assert result["grid_convergence"]["coarse_space_points"] < len(coordinates)


def test_pde_discovery_records_sparse_term_selection():
    times = np.linspace(0.0, 2.0, 20)
    coordinates = np.linspace(0.0, 1.0, 10)
    field = np.array([[np.exp(-t) * np.sin(x) for x in coordinates] for t in times])
    result = discover_1d_pde(times, coordinates, field, sparsity_threshold=0.2,
                              residual_tolerance=2.0, boundary_tolerance=2.0)
    assert result["sparsity_threshold"] == 0.2
    assert result["active_terms"]
    assert set(result["active_terms"]).issubset(set(result["terms"]))


def test_pde_contract_records_units_noise_and_boundary_declarations():
    times = np.linspace(0.0, 2.0, 20)
    coordinates = np.linspace(0.0, 1.0, 10)
    field = np.array([[np.exp(-t) * np.sin(x) for x in coordinates] for t in times])
    result = discover_1d_pde(
        times, coordinates, field, residual_tolerance=2.0, boundary_tolerance=2.0,
        field_dimensions={"Q": 1}, coordinate_dimensions={"L": 1},
        noise_scale=0.01, boundary_values=field[:, [0, -1]],
        field_mask=np.ones_like(field, dtype=bool),
    )
    assert result["unit_status"] == "declared"
    assert result["noise_status"] == "declared"
    assert result["boundary_status"] == "declared_applied"
    assert result["boundary_condition_rmse"] == 0.0
    assert result["missing_status"] == "none_observed"


def test_pde_rejects_missing_mask_instead_of_interpolating_silently():
    times = np.linspace(0.0, 2.0, 20)
    coordinates = np.linspace(0.0, 1.0, 10)
    field = np.ones((times.size, coordinates.size))
    mask = np.ones_like(field, dtype=bool)
    mask[3, 4] = False
    with pytest.raises(ValueError, match="missing_field_values_not_supported"):
        discover_1d_pde(times, coordinates, field, field_mask=mask)


def test_pde_applies_declared_time_varying_boundary_in_rollout():
    times = np.linspace(0.0, 1.0, 20)
    coordinates = np.linspace(0.0, 1.0, 9)
    boundary = np.column_stack((np.sin(times), np.cos(times)))
    field = np.zeros((times.size, coordinates.size))
    field[:, 0], field[:, -1] = boundary[:, 0], boundary[:, 1]
    result = discover_1d_pde(
        times, coordinates, field, include_advection=False,
        boundary_values=boundary, residual_tolerance=1e-2,
        boundary_tolerance=1e-2,
    )
    assert result["boundary_status"] == "declared_applied"
    assert result["boundary_condition_rmse"] == 0.0


def test_pde_2d_discovers_regular_grid_diffusion_candidate():
    times = np.linspace(0.0, 0.8, 20)
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 9)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = np.asarray([np.exp(-0.4 * t) * np.sin(np.pi * xx) * np.sin(np.pi * yy) for t in times])
    result = discover_2d_pde(times, x, y, field, include_advection=False, residual_tolerance=0.2)
    assert result["terms"] == ["u", "u_xx", "u_yy"]
    assert result["grid"]["field_shape"] == [20, 9, 9]
    assert result["forward_stability"]["explicit_cfl_satisfied"] is True
    assert result["grid_convergence"]["status"] == "assessed"


def test_pde_2d_applies_declared_time_varying_boundaries():
    times = np.linspace(0.0, 0.8, 20)
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 9)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = np.asarray([
        np.exp(-0.4 * t) * np.sin(np.pi * xx) * np.sin(np.pi * yy)
        for t in times
    ])
    boundaries = {
        "left": field[:, 0, :], "right": field[:, -1, :],
        "bottom": field[:, :, 0], "top": field[:, :, -1],
    }
    result = discover_2d_pde(
        times, x, y, field, include_advection=False,
        boundary_values=boundaries, residual_tolerance=0.2,
        boundary_tolerance=1e-12,
    )
    assert result["boundary_status"] == "declared_applied"
    assert result["boundary_condition_rmse"] == 0.0


def test_pde_3d_discovers_regular_grid_candidate():
    times = np.linspace(0.0, 0.8, 20)
    x = np.linspace(0.0, 1.0, 5); y = np.linspace(0.0, 1.0, 5); z = np.linspace(0.0, 1.0, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    field = np.asarray([
        np.exp(-0.4 * t) * np.sin(np.pi * xx) * np.sin(np.pi * yy) * np.sin(np.pi * zz)
        for t in times
    ])
    result = discover_3d_pde(times, x, y, z, field, include_advection=False, residual_tolerance=0.5)
    assert result["terms"] == ["u", "u_xx", "u_yy", "u_zz"]
    assert result["grid"]["field_shape"] == [20, 5, 5, 5]
    assert result["forward_stability"]["explicit_cfl_satisfied"] is True
