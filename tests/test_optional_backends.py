import pytest

from core.optional_backends import OptionalBackendError, discover_optional_backends


def test_optional_backend_discovery_is_bounded_and_non_authorizing():
    result = discover_optional_backends(["scipy", "sympy"])
    assert set(result["backends"]) == {"scipy", "sympy"}
    assert all(item["execution_authorized"] is False for item in result["backends"].values())


def test_unknown_backend_is_rejected():
    with pytest.raises(OptionalBackendError):
        discover_optional_backends(["not_a_backend"])


def test_backend_discovery_rejects_duplicate_or_untyped_requests():
    with pytest.raises(OptionalBackendError):
        discover_optional_backends(["scipy", "scipy"])
    with pytest.raises(OptionalBackendError):
        discover_optional_backends([["scipy"]])
