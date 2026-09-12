"""Small typed discrete exterior-calculus backend.

This is a finite simplicial-complex contract, not a claim of a complete
Decapodes category-theoretic compiler.  It makes form degree, orientation,
mesh topology and the d∘d=0 obligation explicit before a PDE solver is called.
"""

from __future__ import annotations

from typing import Any, Sequence


class DECError(ValueError):
    pass


def _matrix_multiply(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    if not left or not right or len(left[0]) != len(right):
        return []
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def build_simplicial_complex(*, vertex_count: int, edges: Sequence[Sequence[int]],
                             faces: Sequence[Sequence[int]] = ()) -> dict[str, Any]:
    if type(vertex_count) is not int or not 1 <= vertex_count <= 10_000:
        raise DECError("vertex_count_out_of_bounds")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)) or not edges:
        raise DECError("edges_required")
    normalized_edges = []
    for edge in edges:
        if not isinstance(edge, Sequence) or len(edge) != 2 or any(type(item) is not int for item in edge):
            raise DECError("edge_must_have_two_integer_vertices")
        if edge[0] == edge[1] or any(item < 0 or item >= vertex_count for item in edge):
            raise DECError("edge_vertex_out_of_bounds")
        normalized_edges.append(tuple(edge))
    normalized_faces = []
    for face in faces:
        if not isinstance(face, Sequence) or len(face) != 3 or any(type(item) is not int for item in face):
            raise DECError("face_must_have_three_integer_vertices")
        if len(set(face)) != 3 or any(item < 0 or item >= vertex_count for item in face):
            raise DECError("face_vertex_out_of_bounds")
        normalized_faces.append(tuple(face))
    edge_index = {edge: index for index, edge in enumerate(normalized_edges)}
    d0 = [[0 for _ in range(vertex_count)] for _ in normalized_edges]
    for row, (start, end) in enumerate(normalized_edges):
        d0[row][start] = -1
        d0[row][end] = 1
    d1 = [[0 for _ in normalized_edges] for _ in normalized_faces]
    for row, face in enumerate(normalized_faces):
        for start, end, sign in ((face[0], face[1], 1), (face[1], face[2], 1), (face[2], face[0], 1)):
            if (start, end) in edge_index:
                d1[row][edge_index[(start, end)]] += sign
            elif (end, start) in edge_index:
                d1[row][edge_index[(end, start)]] -= sign
            else:
                raise DECError("face_boundary_edge_missing")
    composition = _matrix_multiply(d1, d0)
    if any(value != 0 for row in composition for value in row):
        raise DECError("boundary_squared_not_zero")
    return {"schema_version": "mathmodel.dec-complex/v1", "vertex_count": vertex_count,
            "edges": [list(edge) for edge in normalized_edges],
            "faces": [list(face) for face in normalized_faces],
            "d0": d0, "d1": d1, "boundary_squared_zero": True,
            "policy": "typed_discrete_exterior_contract_not_complete_decapodes_compiler"}


def apply_exterior_derivative(complex_payload: dict[str, Any], degree: int,
                              coefficients: Sequence[float]) -> list[float]:
    if not isinstance(complex_payload, dict) or degree not in {0, 1}:
        raise DECError("degree_must_be_zero_or_one")
    matrix = complex_payload.get("d0" if degree == 0 else "d1")
    if not isinstance(matrix, list) or len(coefficients) != (len(matrix[0]) if matrix else 0):
        raise DECError("form_shape_mismatch")
    try:
        values = [float(value) for value in coefficients]
    except (TypeError, ValueError) as exc:
        raise DECError("form_coefficients_must_be_numeric") from exc
    return [sum(float(value) * row[index] for index, value in enumerate(values)) for row in matrix]


__all__ = ["DECError", "build_simplicial_complex", "apply_exterior_derivative"]
