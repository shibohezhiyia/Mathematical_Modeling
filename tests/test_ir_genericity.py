from core.ir_genericity import audit_ir_genericity


def test_generic_ir_passes():
    assert audit_ir_genericity({"nodes": [{"op": "integral", "variable": "t"} ]})["status"] == "pass"


def test_domain_specific_tokens_are_blocked():
    result = audit_ir_genericity({"problem": "无人机 FY1"})
    assert result["status"] == "blocked"
    assert {item["code"] for item in result["findings"]} == {"drone_identifier"}
