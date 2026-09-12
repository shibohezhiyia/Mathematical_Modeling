"""Guarded, auditable rewrites for a small mathematical expression language.

The graph records *why* two forms are related and which conditions are still
required.  It is intentionally conservative: a rewrite with missing domain,
initial-condition, or optimization-sense information is returned as a
proposal, never silently treated as an executable equivalence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


class RewriteError(ValueError):
    pass


def _digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RewriteError("expression_not_finite_json") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _node(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 32:
        raise RewriteError("expression_depth_limit")
    if not isinstance(value, Mapping) or not isinstance(value.get("op"), str):
        raise RewriteError("invalid_expression_node")
    op = str(value["op"])
    allowed = {"symbol", "constant", "add", "mul", "sub", "div", "derivative", "integral",
               "implicit_equation", "explicit_equation", "minimize", "maximize"}
    if op not in allowed:
        raise RewriteError("unsupported_expression_operator")
    args = value.get("args", [])
    if not isinstance(args, list) or len(args) > 8:
        raise RewriteError("invalid_expression_args")
    arities = {"symbol": 0, "constant": 0, "add": 2, "mul": 2, "sub": 2, "div": 2,
               "derivative": 2, "integral": 2, "implicit_equation": 2,
               "explicit_equation": 2, "minimize": 1, "maximize": 1}
    if len(args) != arities[op]:
        raise RewriteError("invalid_expression_arity")
    if op == "symbol" and (type(value.get("name")) is not str or not value["name"].strip() or len(value["name"]) > 128):
        raise RewriteError("invalid_symbol_name")
    if op == "constant":
        try:
            numeric = float(value["value"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RewriteError("invalid_constant") from exc
        import math
        if not math.isfinite(numeric):
            raise RewriteError("invalid_constant")
    result = {"op": op, "args": [_node(item, depth=depth + 1) for item in args]}
    for key in ("name", "value", "variable", "lower", "upper", "sense", "condition"):
        if key in value:
            raw = value[key]
            if key in {"name", "variable", "sense", "condition"}:
                if type(raw) is not str or not raw.strip() or len(raw) > 128:
                    raise RewriteError("invalid_expression_metadata")
                result[key] = raw.strip()
            elif key in {"lower", "upper"}:
                try:
                    numeric = float(raw)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RewriteError("invalid_expression_metadata") from exc
                import math
                if not math.isfinite(numeric):
                    raise RewriteError("invalid_expression_metadata")
                result[key] = numeric
            elif key == "value":
                # Constants are canonicalized so 1, 1.0 and "1" cannot
                # produce different expression identities.
                if op != "constant":
                    raise RewriteError("invalid_expression_metadata")
                result[key] = numeric
    return result


@dataclass(frozen=True)
class RewriteEdge:
    source_hash: str
    target_hash: str
    rule: str
    conditions: tuple[str, ...]
    executable: bool

    def as_dict(self) -> dict[str, Any]:
        return {"source_hash": self.source_hash, "target_hash": self.target_hash,
                "rule": self.rule, "conditions": list(self.conditions),
                "executable": self.executable}


class EquivalenceGraph:
    def __init__(self, expression: Mapping[str, Any]):
        self.root = _node(expression)
        self.forms: dict[str, dict[str, Any]] = {_digest(self.root): deepcopy(self.root)}
        self.edges: list[RewriteEdge] = []

    def _add(self, source: dict[str, Any], target: dict[str, Any], rule: str,
             conditions: tuple[str, ...], executable: bool) -> dict[str, Any]:
        source_hash, target_hash = _digest(source), _digest(target)
        self.forms.setdefault(target_hash, deepcopy(target))
        edge = RewriteEdge(source_hash, target_hash, rule, conditions, executable)
        if edge not in self.edges:
            self.edges.append(edge)
        return {"expression": deepcopy(target), "edge": edge.as_dict()}

    def rewrite(self, rule: str, *, condition: str | None = None,
                source_hash: str | None = None) -> dict[str, Any]:
        """Apply a guarded rewrite to any previously recorded form.

        The default remains the initial expression for compatibility.  A
        caller must explicitly select a known source hash to continue a
        rewrite chain; this prevents silently applying a rule to an unrelated
        expression while preserving a complete provenance edge.
        """
        if source_hash is None:
            source = deepcopy(self.root)
        else:
            if not isinstance(source_hash, str) or source_hash not in self.forms:
                raise RewriteError("unknown_source_form")
            source = deepcopy(self.forms[source_hash])
        op = source["op"]
        if rule == "commute_add" and op == "add" and len(source["args"]) == 2:
            target = {**source, "args": [source["args"][1], source["args"][0]]}
            return self._add(source, target, rule, (), True)
        if rule == "commute_mul" and op == "mul" and len(source["args"]) == 2:
            target = {**source, "args": [source["args"][1], source["args"][0]]}
            return self._add(source, target, rule, (), True)
        if rule == "cancel_self_division" and op == "div" and len(source["args"]) == 2 and source["args"][0] == source["args"][1]:
            if not condition:
                return self._add(source, {"op": "constant", "value": 1}, rule,
                                 ("nonzero_denominator_required",), False)
            if condition != "nonzero_denominator":
                raise RewriteError("unsupported_domain_condition")
            return self._add(source, {"op": "constant", "value": 1}, rule, (), True)
        if rule == "explicit_to_implicit" and op == "explicit_equation" and len(source["args"]) == 2:
            target = {**source, "op": "implicit_equation"}
            return self._add(source, target, rule, (), True)
        if rule == "implicit_to_explicit" and op == "implicit_equation" and len(source["args"]) == 2:
            if condition != "unique_solution_for_target":
                return self._add(source, {**source, "op": "explicit_equation"}, rule,
                                 ("unique_solution_for_target_required",), False)
            return self._add(source, {**source, "op": "explicit_equation"}, rule, (), True)
        if rule == "minimize_to_maximize_negated" and op == "minimize" and len(source["args"]) == 1:
            target = {
                "op": "maximize",
                "args": [{"op": "mul", "args": [{"op": "constant", "value": -1}, source["args"][0]]}],
                "sense": "same_feasible_set",
            }
            return self._add(source, target, rule, (), True)
        if rule == "derivative_to_integral" and op == "derivative" and len(source["args"]) == 2:
            target = {"op": "integral", "args": [source["args"][0], source["args"][1]],
                      "condition": "integration_constant_required"}
            return self._add(source, target, rule, ("initial_or_boundary_condition_required", "integration_constant_required"), False)
        raise RewriteError("rewrite_not_applicable")

    def summary(self) -> dict[str, Any]:
        # Only executable edges may collapse forms for candidate de-duplication.
        # Proposed edges with missing conditions remain visible but do not
        # assert semantic equivalence.
        parent = {key: key for key in self.forms}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for edge in self.edges:
            if edge.executable and edge.source_hash in parent and edge.target_hash in parent:
                union(edge.source_hash, edge.target_hash)
        classes: dict[str, list[str]] = {}
        for key in self.forms:
            classes.setdefault(find(key), []).append(key)
        return {
            "forms": len(self.forms), "edges": [edge.as_dict() for edge in self.edges],
            "executable_edges": sum(edge.executable for edge in self.edges),
            "verified_equivalence_classes": [sorted(values) for values in classes.values()],
            "policy": "conditions_are_preserved_and_unproved_edges_are_not_executable",
        }


__all__ = ["RewriteError", "RewriteEdge", "EquivalenceGraph"]
