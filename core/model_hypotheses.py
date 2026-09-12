"""Versioned, non-executable contracts for open-world model proposals.

These objects protect provenance and type boundaries, not OS sandbox boundaries.
An accepted expression graph is a hypothesis, never a numerical result or proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import json
import math
import re
from threading import Lock
from typing import Any, Iterable, Mapping, Optional


SCHEMA_VERSION = "mathmodel.hypothesis-ir/v1"
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_BASE_DIMENSIONS = ("M", "L", "T", "I", "Theta", "N", "J")
OPERATORS = ("variable", "parameter", "constant", "add", "subtract", "multiply",
             "divide", "minimum", "maximum", "negate", "abs", "sqrt", "exp", "log", "sin", "cos",
             "derivative", "integral", "equal", "observation", "unknown_mechanism")


class HypothesisValidationError(ValueError):
    """Machine-readable errors intentionally omit raw model output and secrets."""

    def __init__(self, code: str, node_id: Optional[str] = None):
        self.code, self.node_id = code, node_id
        super().__init__(code if node_id is None else f"{code}:{node_id}")


def _require(condition: bool, code: str, node_id: Optional[str] = None) -> None:
    if not condition:
        raise HypothesisValidationError(code, node_id)


def _keys(value: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    _require(type(value) is dict, "expected_object")
    _require(required <= value.keys(), "missing_fields")
    _require(value.keys() <= required | optional, "unexpected_fields")


def _text(value: Any, limit: int = 2000) -> str:
    _require(type(value) is str and 0 < len(value) <= limit, "invalid_text")
    return value


def _id(value: Any) -> str:
    _require(type(value) is str and bool(_IDENTIFIER.fullmatch(value)), "invalid_identifier")
    return value


def _canonical(value: Any) -> str:
    # Bound depth and shape even for in-process callers, without accepting custom
    # Mapping objects, overloaded numeric objects, or non-finite JSON values.
    stack, count = [(value, 0)], 0
    while stack:
        current, depth = stack.pop()
        count += 1
        _require(depth <= 20 and count <= 15000, "payload_complexity_limit")
        if type(current) is dict:
            _require(len(current) + len(stack) + count <= 15000, "payload_complexity_limit")
            _require(all(type(key) is str and len(key) <= 200 for key in current), "invalid_keys")
            stack.extend((item, depth + 1) for item in current.values())
        elif type(current) in (list, tuple):
            _require(len(current) + len(stack) + count <= 15000, "payload_complexity_limit")
            stack.extend((item, depth + 1) for item in current)
        else:
            _require(current is None or type(current) in (str, bool, int, float), "non_json_value")
            if type(current) in (int, float):
                _require(abs(current) <= 1e100 and math.isfinite(current), "nonfinite_or_large_number")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    _require(len(encoded.encode("utf-8")) <= 256000, "payload_size_limit")
    return encoded


def decode_proposal(raw: str) -> dict[str, Any]:
    _require(type(raw) is str and len(raw.encode("utf-8")) <= 256000, "response_size_limit")
    depth, quoted, escaped = 0, False, False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            _require(depth <= 20, "payload_complexity_limit")
        elif character in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw, object_pairs_hook=pairs)
        _canonical(parsed)
    except HypothesisValidationError:
        raise
    except (ValueError, OverflowError, RecursionError, TypeError) as exc:
        raise HypothesisValidationError("invalid_json") from exc
    _require(type(parsed) is dict, "expected_object")
    return parsed


@dataclass(frozen=True)
class ProblemContract:
    """Immutable statement facts. Exact quotation is not semantic proof."""

    _json: str = field(repr=False)

    @classmethod
    def create(cls, statement: str, *, facts: Optional[list[dict[str, Any]]] = None,
               hard_constraint_ids: tuple[str, ...] = (), revision: int = 1,
               parent_hash: Optional[str] = None) -> "ProblemContract":
        _text(statement, 64000)
        _require(type(revision) is int and revision >= 1, "invalid_revision")
        _require((revision == 1 and parent_hash is None) or
                 (revision > 1 and type(parent_hash) is str and bool(re.fullmatch(r"[0-9a-f]{64}", parent_hash))),
                 "invalid_parent_revision")
        if facts is None:
            facts = [{"id": "statement", "start": 0, "end": len(statement), "text": statement}]
        _require(type(facts) is list and len(facts) <= 128, "invalid_facts")
        seen = set()
        for fact in facts:
            _keys(fact, {"id", "start", "end", "text"})
            identifier = _id(fact["id"])
            _require(identifier not in seen, "duplicate_fact_id")
            seen.add(identifier)
            start, end = fact["start"], fact["end"]
            _require(type(start) is int and type(end) is int and 0 <= start < end <= len(statement), "invalid_fact_span")
            _require(fact["text"] == statement[start:end], "fact_quote_mismatch")
        _require(type(hard_constraint_ids) in (tuple, list) and all(type(item) is str for item in hard_constraint_ids), "invalid_constraint_references")
        _require(set(hard_constraint_ids) <= seen, "unknown_constraint_fact")
        return cls(_canonical({"schema_version": "mathmodel.problem-contract/v1", "revision": revision,
                               "parent_hash": parent_hash, "statement": statement, "facts": facts,
                               "hard_constraint_ids": list(dict.fromkeys(hard_constraint_ids))}))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProblemContract":
        """Rehydrate a public contract only after a full canonical validation.

        This is deliberately stricter than accepting a dictionary and trusting
        its spans.  Re-creating the contract verifies every quoted fact,
        constraint reference and revision field; the final equality check also
        rejects duplicate constraint IDs that would otherwise be normalized.
        Parent hashes establish lineage identifiers, not proof that an unseen
        parent actually existed, so callers should additionally bind a payload
        to their server-side session or evidence store.
        """
        _require(type(payload) is dict, "invalid_problem_contract")
        _keys(payload, {
            "schema_version", "revision", "parent_hash", "statement",
            "facts", "hard_constraint_ids",
        })
        _require(payload["schema_version"] == "mathmodel.problem-contract/v1",
                 "problem_contract_schema_mismatch")
        contract = cls.create(
            payload["statement"],
            facts=payload["facts"],
            hard_constraint_ids=payload["hard_constraint_ids"],
            revision=payload["revision"],
            parent_hash=payload["parent_hash"],
        )
        _require(contract.public() == payload, "noncanonical_problem_contract")
        return contract

    def public(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def digest(self) -> str:
        return sha256(self._json.encode("utf-8")).hexdigest()

    def revise(self, statement: str, *, facts=None, hard_constraint_ids=None) -> "ProblemContract":
        """Only the application/user path should call this, never a model patch."""
        previous = self.public()
        if hard_constraint_ids is None:
            hard_constraint_ids = previous["hard_constraint_ids"]
        if facts is None and hard_constraint_ids:
            # Do not silently erase protected facts when only the statement is
            # revised. If a quote changes, the caller must explicitly rebind it.
            facts = previous["facts"]
        return self.create(statement, facts=facts, hard_constraint_ids=hard_constraint_ids,
                           revision=previous["revision"] + 1, parent_hash=self.digest)

    def record_confirmation(
        self, question: str, answer: str, *, fact_id: Optional[str] = None,
        hard_constraint: bool = False,
    ) -> "ProblemContract":
        """Create a new contract revision from an explicit user clarification.

        The original statement and facts remain byte-for-byte stable.  The
        appended confirmation is a new quoted fact, so downstream compilers
        can invalidate dependent candidates by comparing ``contract_hash``;
        this method never edits an existing hypothesis or evidence ledger.
        """
        _text(question, 2000)
        _text(answer, 4000)
        previous = self.public()
        identifier = _id(fact_id or f"user_confirmation_{previous['revision']}")
        if any(fact["id"] == identifier for fact in previous["facts"]):
            raise HypothesisValidationError("duplicate_fact_id")
        suffix = f"\n\n[用户确认] {question}\n回答：{answer}"
        statement = previous["statement"] + suffix
        start = len(previous["statement"])
        facts = [*previous["facts"], {
            "id": identifier,
            "start": start,
            "end": len(statement),
            "text": suffix,
        }]
        hard_ids = list(previous["hard_constraint_ids"])
        if hard_constraint:
            hard_ids.append(identifier)
        return self.revise(statement, facts=facts, hard_constraint_ids=hard_ids)


@dataclass(frozen=True)
class SemanticHypothesisSet:
    """同一题意的可竞争解释；解释不是事实，也不是数学求解结果。"""

    _json: str = field(repr=False)

    @classmethod
    def create(cls, contract: ProblemContract, hypotheses: Iterable[Mapping[str, Any]]) -> "SemanticHypothesisSet":
        if not isinstance(contract, ProblemContract):
            raise TypeError("contract_must_be_problem_contract")
        items = list(hypotheses)
        _require(1 <= len(items) <= 16, "semantic_hypothesis_count_limit")
        facts = {item["id"] for item in contract.public()["facts"]}
        seen = set()
        normalized = []
        for item in items:
            _require(type(item) is dict, "semantic_hypothesis_must_be_object")
            _keys(item, {"id", "interpretation", "fact_ids", "assumptions", "open_questions", "status"})
            identifier = _id(item["id"])
            _require(identifier not in seen, "duplicate_semantic_hypothesis_id")
            seen.add(identifier)
            _text(item["interpretation"], 4000)
            fact_ids, assumptions, questions = item["fact_ids"], item["assumptions"], item["open_questions"]
            _require(type(fact_ids) is list and len(fact_ids) <= 32 and all(type(value) is str for value in fact_ids),
                     "invalid_semantic_fact_ids")
            _require(set(fact_ids) <= facts, "unknown_semantic_fact_id")
            _require(type(assumptions) is list and len(assumptions) <= 16 and
                     all(type(value) is str and 0 < len(value) <= 1000 for value in assumptions),
                     "invalid_semantic_assumptions")
            _require(type(questions) is list and len(questions) <= 8 and
                     all(type(value) is str and 0 < len(value) <= 1000 for value in questions),
                     "invalid_semantic_questions")
            _require(item["status"] in ("candidate", "user_confirmed", "rejected"), "invalid_semantic_status")
            normalized.append({"id": identifier, "interpretation": item["interpretation"],
                               "fact_ids": list(dict.fromkeys(fact_ids)), "assumptions": assumptions,
                               "open_questions": questions, "status": item["status"]})
        return cls(_canonical({"schema_version": "mathmodel.semantic-hypotheses/v1",
                              "contract_hash": contract.digest, "contract_revision": contract.public()["revision"],
                              "hypotheses": sorted(normalized, key=lambda value: value["id"])}))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], contract: ProblemContract) -> "SemanticHypothesisSet":
        _require(type(payload) is dict and set(payload) ==
                 {"schema_version", "contract_hash", "contract_revision", "hypotheses"},
                 "invalid_semantic_hypothesis_set")
        _require(payload["schema_version"] == "mathmodel.semantic-hypotheses/v1", "semantic_hypothesis_schema_mismatch")
        _require(payload["contract_hash"] == contract.digest and payload["contract_revision"] == contract.public()["revision"],
                 "semantic_hypothesis_contract_mismatch")
        return cls.create(contract, payload["hypotheses"])

    @property
    def digest(self) -> str:
        return sha256(self._json.encode("utf-8")).hexdigest()

    def public(self) -> dict[str, Any]:
        return {**json.loads(self._json), "digest": self.digest, "authority": "semantic_hypothesis_only"}

    def rank_questions(self, *, max_questions: int = 3,
                       cost_by_question: Mapping[str, float] | None = None,
                       options_by_question: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """按假设分歧与获取成本给出启发式澄清顺序。

        ``options_by_question`` 是可选的交互层补充：每个问题可以附带
        2--3 个用户可选答案、答案影响说明和受影响假设 ID。它只帮助用户
        选择题意，不会把任何选项自动写回契约或数值证据。
        """
        _require(type(max_questions) is int and 1 <= max_questions <= 3, "invalid_question_budget")
        costs = {" ".join(str(key).split()): value for key, value in dict(cost_by_question or {}).items()}
        if options_by_question is not None:
            _require(isinstance(options_by_question, Mapping), "invalid_question_options")
        option_map = {" ".join(str(key).split()): value for key, value in dict(options_by_question or {}).items()}
        items = [item for item in self.public()["hypotheses"] if item["status"] != "rejected"]
        total = len(items)
        hypothesis_ids = {item["id"] for item in items}
        occurrences: dict[str, set[str]] = {}
        for item in items:
            for question in item["open_questions"]:
                normalized = " ".join(question.split())
                occurrences.setdefault(normalized, set()).add(item["id"])
        _require(not (set(option_map) - set(occurrences)), "question_options_reference_unknown_question")
        ranked = []
        for question, affected in occurrences.items():
            cost = costs.get(question, 1.0)
            _require(type(cost) in (int, float) and math.isfinite(float(cost)) and cost > 0,
                     "invalid_question_cost")
            absent = total - len(affected)
            score = (len(affected) * absent) / float(cost)
            record = {"question": question, "affected_hypothesis_ids": sorted(affected),
                      "affected_count": len(affected), "disagreement_count": absent,
                      "cost": float(cost), "heuristic_score": score}
            if question in option_map:
                raw_options = option_map[question]
                _require(type(raw_options) is list and 2 <= len(raw_options) <= 3,
                         "question_options_must_have_2_to_3_choices")
                normalized_options = []
                seen_option_ids, seen_labels = set(), set()
                for option in raw_options:
                    _keys(option, {"id", "label", "hypothesis_ids"}, {"impact"})
                    option_id = _id(option["id"])
                    label = _text(option["label"], 300)
                    _require(option_id not in seen_option_ids and label not in seen_labels,
                             "duplicate_question_option")
                    seen_option_ids.add(option_id); seen_labels.add(label)
                    ids = option["hypothesis_ids"]
                    _require(type(ids) is list and len(ids) <= 16 and
                             all(type(value) is str for value in ids) and set(ids) <= hypothesis_ids,
                             "invalid_question_option_hypothesis_ids")
                    impact = option.get("impact", "")
                    if impact:
                        impact = _text(impact, 1000)
                    normalized_options.append({"id": option_id, "label": label,
                                               "hypothesis_ids": list(dict.fromkeys(ids)),
                                               "impact": impact})
                record["options"] = normalized_options
                record["selection_policy"] = "user_choice_creates_new_contract_revision"
                # Options are often only a UI aid and may overlap.  When they
                # form a valid answer partition, expose a conditional entropy
                # calculation as an additional ranking signal; otherwise keep
                # the heuristic result and explain why information value is
                # unavailable.  Equal weights are deliberately explicit and
                # never presented as semantic posterior probabilities.
                try:
                    from .information_value import estimate_question_information_gain
                    iv = estimate_question_information_gain(
                        {identifier: 1.0 for identifier in sorted(hypothesis_ids)},
                        [{"question": question,
                          "cost": float(cost),
                          "answers": {option["id"]: option["hypothesis_ids"]
                                      for option in normalized_options}}],
                    )
                    summary = iv["questions"][0]
                    record["information_value"] = {
                        "status": "conditional",
                        "expected_information_gain_bits": summary["expected_information_gain_bits"],
                        "gain_per_cost": summary["gain_per_cost"],
                        "policy": "uniform_declared_weights_not_posterior",
                    }
                except (ImportError, ValueError) as exc:
                    code = getattr(exc, "code", "invalid_answer_partition")
                    record["information_value"] = {
                        "status": "not_assessed", "reason": str(code)[:80],
                        "policy": "options_are_not_a_valid_disjoint_answer_partition",
                    }
            ranked.append(record)
        ranked.sort(key=lambda item: (-item["heuristic_score"], item["question"]))
        return {"schema_version": "mathmodel.semantic-question-ranking/v1",
                "semantic_hypothesis_digest": self.digest, "questions": ranked[:max_questions],
                "method": "disagreement_times_coverage_over_declared_cost",
                "authority": "heuristic_question_priority",
                "policy": "not_formal_information_gain"}


@dataclass(frozen=True)
class MathType:
    dtype: str
    shape: tuple[int, ...]
    dimensions: Optional[tuple[Fraction, ...]]

    @classmethod
    def parse(cls, payload: Any) -> "MathType":
        _keys(payload, {"dtype", "shape", "dimensions"})
        _require(payload["dtype"] in ("real", "bool"), "unsupported_dtype")
        shape = payload["shape"]
        _require(type(shape) is list and len(shape) <= 4 and
                 all(type(size) is int and 1 <= size <= 100000 for size in shape), "invalid_shape")
        _require(math.prod(shape) <= 1000000, "shape_size_limit")
        raw, dimensions = payload["dimensions"], None
        if raw is not None:
            _keys(raw, set(), set(_BASE_DIMENSIONS))
            exponents = []
            for key in _BASE_DIMENSIONS:
                value = raw.get(key, 0)
                _require(type(value) is int or (type(value) is str and
                         bool(re.fullmatch(r"-?\d{1,2}(?:/[1-9]\d?)?", value))), "invalid_dimension_exponent")
                exponent = Fraction(value)
                _require(abs(exponent) <= 32 and exponent.denominator <= 16, "dimension_exponent_limit")
                exponents.append(exponent)
            dimensions = tuple(exponents)
        if payload["dtype"] == "bool":
            _require(dimensions == (0,) * 7, "boolean_must_be_dimensionless")
        return cls(payload["dtype"], tuple(shape), dimensions)

    def public(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape), "dimensions": None if self.dimensions is None else {
            key: int(value) if value.denominator == 1 else str(value)
            for key, value in zip(_BASE_DIMENSIONS, self.dimensions) if value}}


@dataclass(frozen=True)
class VariableDescriptor:
    """变量语义补充信息；不替代 MathType 的形状和量纲检查。"""

    value_domain: dict[str, Any]
    time_semantics: str
    observability: str
    source: dict[str, str]

    @classmethod
    def parse(cls, payload: Any) -> "VariableDescriptor":
        _keys(payload, {"value_domain", "time_semantics", "observability", "source"})
        domain = payload["value_domain"]
        _require(type(domain) is dict, "invalid_value_domain")
        kind = domain.get("kind")
        _require(kind in ("unbounded", "interval", "nonnegative", "positive", "finite_set"),
                 "invalid_value_domain")
        if kind == "interval":
            _keys(domain, {"kind", "lower", "upper", "closed_lower", "closed_upper"})
            lower, upper = domain["lower"], domain["upper"]
            _require(type(lower) in (int, float) and type(upper) in (int, float) and
                     math.isfinite(float(lower)) and math.isfinite(float(upper)) and lower < upper,
                     "invalid_value_domain")
            _require(type(domain["closed_lower"]) is bool and type(domain["closed_upper"]) is bool,
                     "invalid_value_domain")
        elif kind == "finite_set":
            _keys(domain, {"kind", "values"})
            values = domain["values"]
            _require(type(values) is list and 1 <= len(values) <= 256 and
                     all(type(value) in (int, float, str, bool) for value in values),
                     "invalid_value_domain")
        else:
            _keys(domain, {"kind"})
        time_semantics = payload["time_semantics"]
        observability = payload["observability"]
        _require(time_semantics in ("static", "instant", "ordered", "duration", "unknown"),
                 "invalid_time_semantics")
        _require(observability in ("observed", "latent", "derived", "unknown"),
                 "invalid_observability")
        source = payload["source"]
        _keys(source, {"kind", "id"})
        _require(source["kind"] in ("fact", "field", "user_confirmation", "derived"),
                 "invalid_variable_source")
        _id(source["id"])
        return cls(json.loads(_canonical(domain)), time_semantics, observability,
                   {"kind": source["kind"], "id": source["id"]})

    def public(self) -> dict[str, Any]:
        return {"value_domain": json.loads(_canonical(self.value_domain)),
                "time_semantics": self.time_semantics,
                "observability": self.observability,
                "source": dict(self.source)}


@dataclass(frozen=True)
class UnknownMechanism:
    node_id: str
    inputs: tuple[str, ...]
    output_type: MathType
    allowed_operators: tuple[str, ...]
    properties: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "inputs": list(self.inputs), "output_type": self.output_type.public(),
                "allowed_operators": list(self.allowed_operators), "properties": list(self.properties),
                "status": "unfilled", "physical_meaning_verified": False}


def _infer_type(op: str, inputs: list[MathType], declared: MathType, node_id: str) -> None:
    """Conservative expression typing; dynamic feedback uses state symbols."""
    if op in ("variable", "parameter", "constant", "unknown_mechanism"):
        return
    _require(all(item.dtype == "real" for item in inputs), "numeric_input_required", node_id)
    if op in ("negate", "observation", "abs", "sqrt", "exp", "log", "sin", "cos"):
        shape = inputs[0].shape
    elif op in ("derivative", "integral"):
        _require(inputs[1].shape == (), "calculus_coordinate_must_be_scalar", node_id)
        shape = inputs[0].shape
    else:
        left, right = inputs
        if op == "multiply" or op == "divide":
            _require(left.shape == right.shape or not left.shape or not right.shape, "shape_mismatch", node_id)
            shape = left.shape or right.shape
        else:
            _require(left.shape == right.shape, "shape_mismatch", node_id)
            shape = left.shape
    _require(declared.shape == shape, "output_shape_mismatch", node_id)
    _require(declared.dtype == ("bool" if op == "equal" else "real"), "output_dtype_mismatch", node_id)
    dimensions = [item.dimensions for item in inputs]
    if op in ("exp", "log", "sin", "cos"):
        _require(dimensions[0] is None or dimensions[0] == (0,) * 7, "transcendental_requires_dimensionless", node_id)
        expected = (0,) * 7
    elif op == "sqrt":
        expected = None if dimensions[0] is None else tuple(value / 2 for value in dimensions[0])
    elif op in ("add", "subtract", "equal", "minimum", "maximum"):
        _require(None in dimensions or dimensions[0] == dimensions[1], "dimension_mismatch", node_id)
        expected = (0,) * 7 if op == "equal" else dimensions[0] if None not in dimensions else None
    elif op in ("multiply", "divide", "derivative", "integral"):
        sign = -1 if op in ("divide", "derivative") else 1
        expected = None if None in dimensions else tuple(a + sign * b for a, b in zip(*dimensions))
    else:
        expected = dimensions[0]
    _require(expected is None or declared.dimensions is None or expected == declared.dimensions, "output_dimension_mismatch", node_id)


@dataclass(frozen=True)
class HypothesisIR:
    _json: str = field(repr=False)
    _obligations: tuple[str, ...] = field(repr=False)
    unknown_mechanisms: tuple[UnknownMechanism, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any], contract: ProblemContract, *, max_nodes: int = 64) -> "HypothesisIR":
        payload = json.loads(_canonical(payload))
        _keys(payload, {"id", "contract_hash", "nodes", "outputs", "assumptions"})
        _id(payload["id"])
        _require(payload["contract_hash"] == contract.digest, "contract_hash_mismatch")
        assumptions = payload["assumptions"]
        _require(type(assumptions) is list and 1 <= len(assumptions) <= 24, "assumptions_required")
        assumption_ids = set()
        for assumption in assumptions:
            _keys(assumption, {"id", "text"})
            identifier = _id(assumption["id"])
            _require(identifier not in assumption_ids, "duplicate_assumption_id")
            assumption_ids.add(identifier)
            _text(assumption["text"])
        nodes = payload["nodes"]
        _require(type(max_nodes) is int and 1 <= max_nodes <= 128, "invalid_node_budget")
        _require(type(nodes) is list and 1 <= len(nodes) <= max_nodes, "node_budget_exceeded")
        facts = {fact["id"] for fact in contract.public()["facts"]}
        by_id, types, names = {}, {}, set()
        obligations = ["numerical_validation_required", "semantic_validation_required"]
        unknown = []
        for node in nodes:
            _keys(node, {"id", "op", "inputs", "type", "attributes", "assumption_ids", "context_fact_ids"})
            identifier, op = _id(node["id"]), node["op"]
            _require(identifier not in by_id, "duplicate_node_id", identifier)
            _require(type(op) is str and op in OPERATORS, "unsupported_operator", identifier)
            _require(type(node["inputs"]) is list and len(node["inputs"]) <= 16, "invalid_inputs", identifier)
            for item in node["inputs"]:
                _id(item)
            for field_name, allowed in (("assumption_ids", assumption_ids), ("context_fact_ids", facts)):
                refs = node[field_name]
                _require(type(refs) is list and all(type(ref) is str for ref in refs) and set(refs) <= allowed,
                         "unknown_provenance_reference", identifier)
            _require(bool(node["assumption_ids"]), "node_assumption_required", identifier)
            declared = MathType.parse(node["type"])
            node["type"] = declared.public()
            types[identifier], by_id[identifier] = declared, node
            if declared.dimensions is None:
                obligations.append(f"units_unresolved:{identifier}")
            attrs = node["attributes"]
            if op in ("variable", "parameter"):
                _keys(attrs, {"name", "role"}, {"descriptor"})
                name = _text(attrs["name"], 100)
                _require(name not in names, "duplicate_symbol_name", identifier)
                names.add(name)
                _require(attrs["role"] in ("state", "latent_state", "parameter", "control", "coordinate", "observed"), "invalid_role", identifier)
                _require(op != "parameter" or attrs["role"] == "parameter", "parameter_role_mismatch", identifier)
                _require(declared.dtype == "real", "numeric_symbol_required", identifier)
                if "descriptor" in attrs:
                    descriptor = VariableDescriptor.parse(attrs["descriptor"])
                    if attrs["role"] == "latent_state":
                        _require(descriptor.observability in ("latent", "unknown"),
                                 "latent_descriptor_mismatch", identifier)
                    if attrs["role"] in ("observed", "coordinate"):
                        _require(descriptor.observability in ("observed", "unknown"),
                                 "observed_descriptor_mismatch", identifier)
                obligations.append(f"symbol_binding_required:{identifier}")
                if attrs["role"] == "latent_state":
                    obligations.append(f"latent_semantics_unverified:{identifier}")
            elif op == "constant":
                _keys(attrs, {"value"})
                value = attrs["value"]
                _require(type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e12,
                         "invalid_constant", identifier)
                _require(declared.dtype == "real" and declared.shape == (), "scalar_constant_required", identifier)
                if declared.dimensions not in (None, (0,) * 7):
                    obligations.append(f"assumed_quantity_requires_verification:{identifier}")
            elif op == "unknown_mechanism":
                _keys(attrs, {"allowed_operators", "properties"})
                ops, properties = attrs["allowed_operators"], attrs["properties"]
                _require(type(ops) is list and 1 <= len(ops) <= 16 and all(type(item) is str and item in OPERATORS for item in ops), "invalid_search_space", identifier)
                _require(type(properties) is list and len(properties) <= 12, "invalid_properties", identifier)
                for item in properties:
                    _text(item, 300)
                unknown.append(UnknownMechanism(identifier, tuple(node["inputs"]), declared, tuple(ops), tuple(properties)))
                obligations.append(f"mechanism_unfilled:{identifier}")
            else:
                _keys(attrs, set())
            if op in ("divide", "log", "integral", "derivative"):
                obligations.append(f"domain_or_boundary_check_required:{identifier}")
        outputs = payload["outputs"]
        _require(type(outputs) is list and 1 <= len(outputs) <= 16 and all(type(item) is str and item in by_id for item in outputs), "invalid_outputs")
        done, visiting = set(), set()

        def visit(identifier: str) -> None:
            _require(identifier in by_id, "unknown_input_node")
            _require(identifier not in visiting, "expression_dependency_cycle", identifier)
            if identifier in done:
                return
            visiting.add(identifier)
            node = by_id[identifier]
            op, inputs = node["op"], node["inputs"]
            arity = 0 if op in ("variable", "parameter", "constant") else 1 if op in (
                "negate", "abs", "sqrt", "exp", "log", "sin", "cos", "observation"
            ) else 2
            _require(op == "unknown_mechanism" or len(inputs) == arity, "operator_arity_mismatch", identifier)
            for child in inputs:
                visit(child)
            _infer_type(op, [types[child] for child in inputs], types[identifier], identifier)
            visiting.remove(identifier)
            done.add(identifier)

        for output in outputs:
            visit(output)
        _require(len(done) == len(nodes), "unreachable_nodes")
        return cls(_canonical(payload), tuple(obligations), tuple(unknown))

    @property
    def contract_hash(self) -> str:
        return json.loads(self._json)["contract_hash"]

    def payload(self) -> dict[str, Any]:
        """Detached canonical input, without public status/authority metadata."""
        return json.loads(self._json)

    @property
    def digest(self) -> str:
        payload = json.loads(self._json)
        payload.pop("id")
        return sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def public(self) -> dict[str, Any]:
        return {**json.loads(self._json), "schema_version": SCHEMA_VERSION, "hypothesis_hash": self.digest,
                "authority": "hypothesis_only", "execution_status": "not_executed",
                "validation_status": "needs_bindings" if self.unknown_mechanisms or
                any(item.startswith("units_unresolved:") for item in self._obligations) else "type_checked",
                "obligations": list(self._obligations),
                "unknown_mechanisms": [item.public() for item in self.unknown_mechanisms]}


@dataclass(frozen=True)
class EvidenceLedger:
    """Append-only snapshots, bound to exact contract and hypothesis versions."""

    _records: tuple[str, ...] = field(default=(), repr=False)

    def append(self, hypothesis: HypothesisIR, *, method: str, outcome: str,
               scope: dict[str, Any]) -> "EvidenceLedger":
        _require(method in ("type_check", "numerical_test", "counterexample"), "unsupported_evidence_method")
        _require(outcome in ("pass", "fail", "not_assessed"), "invalid_evidence_outcome")
        _require(method != "counterexample" or outcome == "fail", "invalid_counterexample_outcome")
        _require(type(scope) is dict and bool(scope), "evidence_scope_required")
        if method != "type_check":
            _require({"input_hash", "evaluator_version", "domain", "seed"} <= scope.keys(), "numerical_evidence_context_required")
            _require(type(scope["input_hash"]) is str and bool(re.fullmatch(r"[0-9a-f]{64}", scope["input_hash"])), "invalid_evidence_input_hash")
            _text(scope["evaluator_version"], 100)
            _require(type(scope["domain"]) is dict and bool(scope["domain"]), "evidence_domain_required")
            _require(scope["seed"] is None or (type(scope["seed"]) is int and scope["seed"] >= 0), "invalid_evidence_seed")
        encoded_scope = _canonical(scope)
        _require(len(self._records) < 256, "evidence_budget_exceeded")
        entry = {"contract_hash": hypothesis.contract_hash, "hypothesis_hash": hypothesis.digest,
                 "method": method, "outcome": outcome, "scope": scope,
                 "scope_hash": sha256(encoded_scope.encode("utf-8")).hexdigest()}
        return EvidenceLedger((*self._records, _canonical(entry)))

    def for_hypothesis(self, hypothesis: HypothesisIR, *, scope: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        # A model match alone must not reuse numerical evidence from other data,
        # domains, seeds or evaluator versions. An explicit context is required.
        scope_hash = sha256(_canonical(scope).encode("utf-8")).hexdigest() if scope is not None else None
        return [item for item in self.public()["records"] if item["contract_hash"] == hypothesis.contract_hash
                and item["hypothesis_hash"] == hypothesis.digest
                and (item["scope_hash"] == scope_hash if scope_hash is not None else item["method"] == "type_check")]

    def public(self) -> dict[str, Any]:
        return {"schema_version": "mathmodel.evidence-ledger/v1", "records": [json.loads(item) for item in self._records]}


class SearchController:
    """Server-owned request budgets; failed model calls still consume budget."""

    def __init__(self, *, max_calls: int = 1, max_candidates: int = 3, max_nodes: int = 64):
        for value, upper in ((max_calls, 8), (max_candidates, 8), (max_nodes, 128)):
            _require(type(value) is int and 1 <= value <= upper, "invalid_search_budget")
        self.max_calls, self.max_candidates, self.max_nodes = max_calls, max_candidates, max_nodes
        self._calls, self._candidate_hashes = 0, set()
        self._lock = Lock()

    def reserve_call(self) -> None:
        with self._lock:
            _require(self._calls < self.max_calls, "model_call_budget_exhausted")
            self._calls += 1

    def register(self, hypothesis: HypothesisIR) -> bool:
        with self._lock:
            if hypothesis.digest in self._candidate_hashes:
                return False
            _require(len(self._candidate_hashes) < self.max_candidates, "candidate_budget_exhausted")
            self._candidate_hashes.add(hypothesis.digest)
            return True

    def public(self) -> dict[str, Any]:
        with self._lock:
            return {"schema_version": "mathmodel.search-controller/v1", "max_calls": self.max_calls,
                    "max_candidates": self.max_candidates, "max_nodes": self.max_nodes,
                    "calls_used": self._calls, "candidates_registered": len(self._candidate_hashes),
                    "can_execute": False}
