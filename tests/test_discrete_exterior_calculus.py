import pytest

from core.discrete_exterior_calculus import DECError, apply_exterior_derivative, build_simplicial_complex


def test_triangle_complex_has_oriented_boundary_squared_zero():
    complex_payload = build_simplicial_complex(
        vertex_count=3, edges=[(0, 1), (1, 2), (2, 0)], faces=[(0, 1, 2)]
    )
    assert complex_payload["boundary_squared_zero"] is True
    assert apply_exterior_derivative(complex_payload, 0, [1, 2, 4]) == [1.0, 2.0, -3.0]


def test_missing_face_edge_is_rejected():
    with pytest.raises(DECError, match="face_boundary_edge_missing"):
        build_simplicial_complex(vertex_count=3, edges=[(0, 1), (1, 2)], faces=[(0, 1, 2)])
