from core.gradient_validation import gradient_optimization_gate


def test_gradient_gate_blocks_failed_or_missing_check():
    assert gradient_optimization_gate({"status": "fail"})["use_gradient"] is False
    assert gradient_optimization_gate({"status": "pass"})["use_gradient"] is True
