from core.modeling_assistant import MathModelingAssistant


def test_no_dataset_problem_does_not_crash_ranking_side_channel(tmp_path):
    result = MathModelingAssistant(
        output_dir=str(tmp_path),
        feedback_optimization=False,
        credibility_audit=False,
    ).run(
        "建立可持续旅游管理模型，优化游客数量、收入和环境压力之间的平衡。",
        datasets={},
        run_modeling=True,
        generate_plots=False,
    )
    assert result.to_dict()["input_mode"] == "mechanistic_no_dataset"
    assert result.report_path
