from core.cegis_controller import CEGISConfig
from core.model_family_adapters import ModelFamilyAdapter, replay_adapter_witnesses, run_model_family_cegis


def test_adapter_loop_separates_compile_failure_from_counterexample():
    def compile_candidate(candidate):
        if candidate.get("bad"):
            raise ValueError("compile")
        return dict(candidate)

    adapter = ModelFamilyAdapter(
        family="toy",
        compile=compile_candidate,
        evaluate=lambda compiled, cases: {"status": "pass", "score": 1.0, "violations": [], "cost_units": 1},
        diagnose=lambda feedback: {"counterexamples": feedback.get("counterexample_archive", [])},
        patch=lambda candidate, diagnosis: [],
        replay=lambda compiled, witnesses: {"status": "all_replayed", "count": len(witnesses)},
    )
    result = run_model_family_cegis(adapter, [{"bad": True}, {"x": 1}], [{}],
                                    config=CEGISConfig(max_rounds=4, max_candidates=4, max_repairs=1))
    assert result["adapter_family"] == "toy"
    assert result["policy"]["resource_failure_is_not_counterexample"] is True
    assert any(row["status"] == "not_assessed" for row in result["records"])
    assert any(row["status"] == "pass" for row in result["records"])
    assert replay_adapter_witnesses(adapter, {"x": 1}, [{"id": "w"}])["status"] == "all_replayed"
