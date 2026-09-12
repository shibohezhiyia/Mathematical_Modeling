import math

import pytest

from core.math_primitives import PRIMITIVES, PrimitiveValidationError, validate_primitive_node


@pytest.mark.parametrize("op,spec", list(PRIMITIVES.items()))
def test_every_primitive_has_valid_min_and_max_arity_contract(op, spec):
    for arity in {spec.min_inputs, spec.max_inputs}:
        kind = spec.output_kind
        result = validate_primitive_node({
            "id": f"{op}_{arity}", "op": op, "inputs": [f"x{i}" for i in range(arity)],
            "kind": kind, "dimensions": {}, "attributes": {},
        })
        assert result["op"] == op
        assert result["input_count"] == arity


@pytest.mark.parametrize("op,spec", list(PRIMITIVES.items()))
def test_every_primitive_rejects_nonfinite_and_out_of_scale_dimensions(op, spec):
    base = {"id": f"{op}_bad", "op": op, "inputs": [f"x{i}" for i in range(spec.min_inputs)],
            "kind": spec.output_kind, "attributes": {}}
    with pytest.raises(PrimitiveValidationError, match="invalid_dimensions"):
        validate_primitive_node({**base, "dimensions": {"L": math.inf}})
    with pytest.raises(PrimitiveValidationError, match="invalid_dimensions"):
        validate_primitive_node({**base, "dimensions": {"L": 1_000_001}})
