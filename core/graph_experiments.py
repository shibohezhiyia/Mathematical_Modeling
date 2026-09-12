"""Immutable, explicitly bound experiments and narrow patches for typed graphs.

An experiment is supplied by the application/user, never by the mutator. Its
observations, domain, properties and tolerances remain fixed throughout search.
Finite checks are not proofs, and this interface deliberately has no test set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Iterable, Mapping

from .model_hypotheses import (
    HypothesisIR, ProblemContract, _canonical, _id, _keys, _require,
)

EXPERIMENT_VERSION = "mathmodel.graph-experiment/v1"
SCALAR_OPERATORS = frozenset({"variable", "parameter", "constant", "add", "subtract",
                              "multiply", "divide", "minimum", "maximum", "negate", "abs",
                              "sqrt", "exp", "log", "sin", "cos", "observation"})
COMMUTATIVE_OPERATORS = frozenset({"add", "multiply", "minimum", "maximum"})


def fingerprint(value) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def algebraic_equivalence_key(graph_or_payload) -> str:
    """返回受限代数等价键，当前只折叠交换律，不改写定义域或精度语义。"""
    payload = graph_or_payload.payload() if isinstance(graph_or_payload, HypothesisIR) else graph_or_payload
    normalized = json.loads(_canonical(payload))
    # ``id`` is an experiment label, not part of the mathematical object.
    # Omitting it makes the key stable across generated rewrite labels while
    # retaining node identities and provenance in the executable graph.
    normalized.pop("id", None)
    for node in normalized["nodes"]:
        if node["op"] in COMMUTATIVE_OPERATORS:
            node["inputs"] = sorted(node["inputs"])
    return fingerprint(normalized)


def mechanism_diversity_key(graph_or_payload) -> str:
    """忽略符号名称的机制拓扑键，用于发现伪多样性而非强制去重。"""
    payload = graph_or_payload.payload() if isinstance(graph_or_payload, HypothesisIR) else graph_or_payload
    by_id = {node["id"]: node for node in payload["nodes"]}
    memo = {}

    def signature(identifier):
        if identifier in memo:
            return memo[identifier]
        node = by_id[identifier]
        op = node["op"]
        descriptor = node.get("attributes", {}).get("descriptor")
        if op in ("variable", "parameter"):
            attrs = node.get("attributes", {})
            value = _canonical({"op": op, "role": attrs.get("role"), "type": node["type"],
                                "source": descriptor.get("source") if descriptor else None})
        elif op == "constant":
            value = _canonical({"op": op, "type": node["type"],
                                "value": node.get("attributes", {}).get("value")})
        else:
            children = [signature(child) for child in node.get("inputs", [])]
            if op in COMMUTATIVE_OPERATORS:
                children.sort()
            attrs = node.get("attributes", {})
            value = _canonical({"op": op, "type": node["type"], "inputs": children,
                                "allowed_operators": attrs.get("allowed_operators", [])
                                if op == "unknown_mechanism" else None})
        memo[identifier] = value
        return value

    return fingerprint({"outputs": [signature(identifier) for identifier in payload["outputs"]]})


def diversity_audit(graphs: Iterable[HypothesisIR | Mapping[str, Any]]) -> dict:
    """报告只改变量名/交换顺序造成的候选聚集，不自动淘汰候选。"""
    groups = {}
    for graph in graphs:
        digest = graph.digest if isinstance(graph, HypothesisIR) else fingerprint(
            {key: value for key, value in graph.items() if key != "id"}
        )
        key = mechanism_diversity_key(graph)
        groups.setdefault(key, []).append(digest)
    collisions = [{"mechanism_key": key, "hypothesis_hashes": sorted(set(values)),
                   "reason": "same_operator_topology_after_symbol_name_erasure"}
                  for key, values in groups.items() if len(set(values)) > 1]
    return {"candidate_count": sum(len(values) for values in groups.values()),
            "distinct_mechanism_count": len(groups), "collision_groups": collisions,
            "policy": "diagnostic_only_no_automatic_rejection"}


def commutative_rewrites(graph: HypothesisIR, contract: ProblemContract, *, max_rewrites: int = 16) -> list[dict]:
    """生成可审计的交换律候选；不声称浮点逐位相等。"""
    _require(type(max_rewrites) is int and 1 <= max_rewrites <= 32, "invalid_rewrite_budget")
    payload = graph.payload()
    rewrites = []
    for node in payload["nodes"]:
        if node["op"] not in COMMUTATIVE_OPERATORS or len(node["inputs"]) != 2 or node["inputs"][0] == node["inputs"][1]:
            continue
        revised = json.loads(_canonical(payload))
        target = next(item for item in revised["nodes"] if item["id"] == node["id"])
        target["inputs"] = [target["inputs"][1], target["inputs"][0]]
        revised["id"] = f"rewrite_{node['id']}_{len(rewrites)}"
        candidate = HypothesisIR.from_payload(revised, contract)
        rewrites.append({
            "rule": f"commutativity:{node['op']}",
            "source_hash": graph.digest,
            "target_hash": candidate.digest,
            "algebraic_equivalence_key": algebraic_equivalence_key(candidate),
            "exact_algebraic": True,
            "floating_point_equivalence": "within_tolerance_only",
            "candidate": candidate,
        })
        if len(rewrites) >= max_rewrites:
            break
    return rewrites


def point_fingerprint(point: dict) -> str:
    # Scalar real inputs have the same meaning across JSON integer/float/zero
    # spellings; these must not bypass partition-overlap checks.
    return fingerprint({key: float(value) if value != 0 else 0.0 for key, value in point.items()})


def restore_problem(payload: dict) -> ProblemContract:
    _keys(payload, {"schema_version", "revision", "parent_hash", "statement", "facts", "hard_constraint_ids"})
    _require(payload["schema_version"] == "mathmodel.problem-contract/v1", "problem_version_mismatch")
    return ProblemContract.create(payload["statement"], facts=payload["facts"],
                                  hard_constraint_ids=payload["hard_constraint_ids"],
                                  revision=payload["revision"], parent_hash=payload["parent_hash"])


def finite_number(value, *, maximum=1e12) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= maximum


def ensure_scalar_executable(graph: HypothesisIR) -> None:
    for node in graph.payload()["nodes"]:
        _require(node["op"] in SCALAR_OPERATORS, "graph_operator_not_executable", node["id"])
        _require(node["type"]["shape"] == [] and node["type"]["dtype"] == "real",
                 "scalar_real_graph_required", node["id"])
        _require(node["type"]["dimensions"] is not None, "graph_units_unresolved", node["id"])


def interface_matches(graph: HypothesisIR, template: HypothesisIR) -> None:
    candidate, original = graph.payload(), template.payload()
    _require(candidate["contract_hash"] == original["contract_hash"], "contract_hash_mismatch")
    _require(candidate["assumptions"] == original["assumptions"], "protected_assumptions")
    _require(candidate["outputs"] == original["outputs"], "protected_outputs")
    symbols = lambda p: {n["id"]: n for n in p["nodes"] if n["op"] in ("variable", "parameter")}
    _require(symbols(candidate) == symbols(original), "protected_symbol_bindings")
    old_types = {n["id"]: n["type"] for n in original["nodes"]}
    new_types = {n["id"]: n["type"] for n in candidate["nodes"]}
    _require(all(new_types[key] == old_types[key] for key in original["outputs"]), "protected_output_types")


@dataclass(frozen=True)
class SearchExperiment:
    _json: str = field(repr=False)

    @classmethod
    def create(cls, contract: ProblemContract, template: HypothesisIR, *, domain: dict,
               parameter_bounds: dict | None = None, training_cases: list | None = None,
               search_cases: list | None = None, properties: list | None = None,
               absolute_tolerance: float = 1e-6, relative_tolerance: float = 1e-4,
               probe_count: int = 24, seed: int = 42) -> "SearchExperiment":
        return cls.from_payload({"schema_version": EXPERIMENT_VERSION, "contract_hash": contract.digest,
            "template": template.payload(), "domain": domain,
            "parameter_bounds": {} if parameter_bounds is None else parameter_bounds,
            "training_cases": [] if training_cases is None else training_cases,
            "search_cases": [] if search_cases is None else search_cases,
            "properties": [] if properties is None else properties, "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance, "probe_count": probe_count, "seed": seed}, contract)

    @classmethod
    def from_payload(cls, payload: dict, contract: ProblemContract) -> "SearchExperiment":
        payload = json.loads(_canonical(payload))
        _keys(payload, {"schema_version", "contract_hash", "template", "domain", "parameter_bounds",
                        "training_cases", "search_cases", "properties", "absolute_tolerance",
                        "relative_tolerance", "probe_count", "seed"})
        _require(payload["schema_version"] == EXPERIMENT_VERSION, "experiment_version_mismatch")
        _require(payload["contract_hash"] == contract.digest, "contract_hash_mismatch")
        template = HypothesisIR.from_payload(payload["template"], contract)
        # A template may have a typed hole; only completed candidates execute.
        for node in template.payload()["nodes"]:
            _require(node["type"]["shape"] == [] and node["type"]["dtype"] == "real" and
                     node["type"]["dimensions"] is not None, "scalar_known_units_required")
        symbols = {n["id"] for n in template.payload()["nodes"] if n["op"] == "variable"}
        parameters = {n["id"] for n in template.payload()["nodes"] if n["op"] == "parameter"}
        _require(len(symbols) <= 8 and len(parameters) <= 8, "experiment_symbol_limit")
        _keys(payload["domain"], symbols)
        for interval in payload["domain"].values():
            _require(type(interval) is list and len(interval) == 2 and all(finite_number(v) for v in interval)
                     and interval[0] < interval[1], "invalid_domain")
        _keys(payload["parameter_bounds"], parameters)
        for bound in payload["parameter_bounds"].values():
            _require(type(bound) is list and len(bound) == 3 and all(finite_number(v) for v in bound)
                     and bound[0] < bound[1] and bound[0] <= bound[2] <= bound[1], "invalid_parameter_bounds")
        for key in ("absolute_tolerance", "relative_tolerance"):
            _require(finite_number(payload[key], maximum=1e6) and payload[key] >= 0, "invalid_check_tolerance")
        _require(payload["absolute_tolerance"] > 0 or payload["relative_tolerance"] > 0, "positive_tolerance_required")
        _require(type(payload["probe_count"]) is int and 0 <= payload["probe_count"] <= 64, "invalid_probe_budget")
        _require(type(payload["seed"]) is int and 0 <= payload["seed"] < 2**32, "invalid_experiment_seed")
        outputs = set(template.payload()["outputs"])
        _require(len(outputs) <= 4, "experiment_output_limit")
        ids, training_inputs = set(), set()
        for partition, maximum in (("training_cases", 256), ("search_cases", 128)):
            cases = payload[partition]
            _require(type(cases) is list and len(cases) <= maximum, "case_budget_exceeded")
            for case in cases:
                _keys(case, {"id", "bindings", "expected"})
                _id(case["id"])
                _require(case["id"] not in ids, "duplicate_case_id")
                ids.add(case["id"])
                validate_point(case["bindings"], payload["domain"])
                _keys(case["expected"], outputs)
                _require(all(finite_number(v) for v in case["expected"].values()), "invalid_expected_value")
                key = point_fingerprint(case["bindings"])
                if partition == "training_cases":
                    training_inputs.add(key)
                else:
                    _require(key not in training_inputs, "training_search_overlap")
        _require(not parameters or len(payload["training_cases"]) >= 3, "parameter_training_data_required")
        properties = payload["properties"]
        _require(type(properties) is list and len(properties) <= 16, "property_budget_exceeded")
        property_ids = set()
        for prop in properties:
            _keys(prop, {"id", "output", "lower", "upper", "source"})
            _id(prop["id"])
            _require(prop["id"] not in property_ids, "duplicate_property_id")
            property_ids.add(prop["id"])
            _require(type(prop["output"]) is str and prop["output"] in outputs, "unknown_property_output")
            _require(all(v is None or finite_number(v) for v in (prop["lower"], prop["upper"])) and
                     (prop["lower"] is not None or prop["upper"] is not None), "invalid_property_bound")
            _require(prop["lower"] is None or prop["upper"] is None or prop["lower"] <= prop["upper"], "invalid_property_bound")
            _keys(prop["source"], {"kind", "id"})
            kind = prop["source"]["kind"]
            _require(kind in ("fact", "assumption"), "property_source_required")
            sources = contract.public()["facts"] if kind == "fact" else template.payload()["assumptions"]
            _require(type(prop["source"]["id"]) is str and prop["source"]["id"] in {s["id"] for s in sources},
                     "unknown_property_source")
        _require(bool(payload["search_cases"] or properties), "independent_check_required")
        _require(not properties or payload["probe_count"] > 0, "property_probes_required")
        return cls(_canonical(payload))

    @property
    def digest(self) -> str:
        return sha256(self._json.encode("utf-8")).hexdigest()

    def public(self) -> dict:
        return json.loads(self._json)


def validate_point(point: dict, domain: dict) -> None:
    _keys(point, set(domain))
    _require(all(finite_number(point[key]) and bounds[0] <= point[key] <= bounds[1]
                 for key, bounds in domain.items()), "point_outside_domain")


def apply_graph_patch(parent: HypothesisIR, patch: dict, contract: ProblemContract, *,
                      max_changes: int = 4) -> HypothesisIR:
    """Untrusted JSON patch cannot touch facts, observations or the graph interface."""
    patch = json.loads(_canonical(patch))
    _keys(patch, {"parent_hash", "id", "replace_nodes", "add_nodes", "remove_node_ids"})
    _require(patch["parent_hash"] == parent.digest, "stale_patch_parent")
    _id(patch["id"])
    _require(type(max_changes) is int and 1 <= max_changes <= 8, "invalid_patch_budget")
    for field_name in ("replace_nodes", "add_nodes", "remove_node_ids"):
        _require(type(patch[field_name]) is list, "invalid_patch_operations")
    _require(1 <= sum(len(patch[key]) for key in ("replace_nodes", "add_nodes", "remove_node_ids")) <= max_changes,
             "patch_budget_exceeded")
    payload = parent.payload()
    by_id = {node["id"]: node for node in payload["nodes"]}
    touched = set()
    for action in ("replace_nodes", "add_nodes", "remove_node_ids"):
        for item in patch[action]:
            identifier = _id(item) if action == "remove_node_ids" else _id(item.get("id") if type(item) is dict else None)
            _require(identifier not in touched, "duplicate_patch_target")
            touched.add(identifier)
            _require((identifier not in by_id) if action == "add_nodes" else (identifier in by_id), "invalid_patch_target")
            if action != "add_nodes":
                _require(by_id[identifier]["op"] not in ("variable", "parameter"), "protected_symbol_bindings")
                if action == "replace_nodes":
                    _require(item.get("type") == by_id[identifier]["type"], "protected_node_type")
                    if by_id[identifier]["op"] == "unknown_mechanism":
                        _require(item.get("op") in by_id[identifier]["attributes"]["allowed_operators"],
                                 "mechanism_search_space_violation")
            if action == "remove_node_ids":
                del by_id[identifier]
            else:
                by_id[identifier] = item
    payload["id"], payload["nodes"] = patch["id"], list(by_id.values())
    result = HypothesisIR.from_payload(payload, contract)
    interface_matches(result, parent)
    _require(result.digest != parent.digest, "no_effect_patch")
    return result
