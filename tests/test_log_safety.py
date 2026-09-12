import json

from core.log_safety import safe_diagnostic


def test_public_diagnostic_omits_tracebacks_secrets_and_raw_rows():
    result = safe_diagnostic({
        "code": "solver_failed", "message": "first line\nsecret traceback",
        "traceback": "private stack", "api_key": "sk-private",
        "counterexample": {"path": r"C:\Users\Alice\secret.xlsx", "rows": [1, 2]},
    })
    encoded = json.dumps(result, ensure_ascii=False)
    assert "private stack" not in encoded
    assert "sk-private" not in encoded
    assert result["traceback_included"] is False


def test_diagnostic_is_bounded():
    result = safe_diagnostic({"message": "ok", "counterexample": {"x": "a" * 10000}}, max_bytes=256)
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 256
