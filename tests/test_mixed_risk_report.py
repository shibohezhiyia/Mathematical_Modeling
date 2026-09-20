from scripts.audit_mixed_risk_report import derive_erratum


def test_erratum_reclassifies_correct_refusal_without_mutating_frozen_rows():
    report = {
        "schema_version": "mathmodel.mixed-risk-confirmation/v1",
        "compiler_summary": {"modeling_multi_table": {"false_abstain_count": 1}},
        "compiler_rows": [{"family": "modeling_multi_table", "status": "needs_input", "valid": True}],
        "product_summary": {"single_solver": {"solver_arm_launch_count": 20}},
        "freeze": {"freeze_digest": "a" * 64},
        "freeze_before": {"status": "verified"}, "freeze_after": {"status": "verified"},
    }
    audit = derive_erratum(report, source="frozen.json", source_sha256="b" * 64)
    assert audit["corrected_compiler_summary"]["modeling_multi_table"]["false_abstain_count"] == 0
    assert audit["corrected_product_summary"]["single_solver"]["solver_arm_launch_count"] is None
    assert audit["corrected_product_summary"]["single_solver"]["reported_numerical_solver_calls"] == 20
    assert report["compiler_summary"]["modeling_multi_table"]["false_abstain_count"] == 1
    assert report["product_summary"]["single_solver"]["solver_arm_launch_count"] == 20
