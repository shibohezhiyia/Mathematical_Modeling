import pytest

from core.ir_schema import (
    IRSchemaError,
    ir_digest,
    migrate_ir,
    unwrap_ir,
    wrap_ir,
)


def test_digest_is_order_independent_and_envelope_roundtrips():
    first = {"b": [2, 3], "a": {"x": 1}}
    second = {"a": {"x": 1}, "b": [2, 3]}
    assert ir_digest(first) == ir_digest(second)
    envelope = wrap_ir(first, kind="hypothesis_ir")
    assert unwrap_ir(envelope, expected_kind="hypothesis_ir") == first
    assert migrate_ir(envelope) == envelope


def test_tampering_is_rejected():
    envelope = wrap_ir({"nodes": []}, kind="primitive_graph")
    envelope["payload"]["nodes"].append({"op": "variable"})
    with pytest.raises(IRSchemaError, match="ir_digest_mismatch"):
        migrate_ir(envelope)


def test_legacy_payload_requires_kind_and_migrates_losslessly():
    legacy = {"nodes": [{"id": "x", "kind": "variable"}]}
    with pytest.raises(IRSchemaError, match="legacy_ir_kind_required"):
        migrate_ir(legacy)
    migrated = migrate_ir(legacy, kind="primitive_graph")
    assert unwrap_ir(migrated, expected_kind="primitive_graph") == legacy
    with pytest.raises(IRSchemaError, match="ir_kind_mismatch"):
        unwrap_ir(migrated, expected_kind="hypothesis_ir")


def test_nonfinite_and_unknown_kind_are_rejected():
    with pytest.raises(IRSchemaError, match="ir_nonfinite_number"):
        wrap_ir({"value": float("nan")}, kind="hypothesis_ir")
    with pytest.raises(IRSchemaError, match="unsupported_ir_kind"):
        wrap_ir({}, kind="unknown")
