"""Serializable process entry for the main research pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def run_research_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("research_payload_invalid")
    datasets: dict[str, pd.DataFrame] = {}
    for name, spec in (payload.get("datasets") or {}).items():
        if not isinstance(name, str) or not isinstance(spec, Mapping):
            raise ValueError("research_dataset_payload_invalid")
        frame = pd.DataFrame(spec.get("records", []))
        if len(frame) > 100_000:
            raise ValueError("research_dataset_too_large")
        frame.attrs["source_rows"] = int(spec.get("source_rows", len(frame)))
        datasets[name] = frame
    from .modeling_assistant import MathModelingAssistant
    from .artifact_manager import create_run_id
    options = dict(payload.get("options") or {})
    assistant = MathModelingAssistant(
        output_dir=str(payload["output_dir"]),
        max_analysis_rows=int(options.get("max_analysis_rows", 50_000)),
        feedback_optimization=bool(options.get("feedback_optimization", True)),
        feedback_trials=int(options.get("feedback_trials", 6)),
        credibility_audit=bool(options.get("credibility_audit", True)),
        enable_gnn_screen=bool(options.get("enable_gnn_screen", False)),
        enable_graph_search=bool(options.get("enable_graph_search", False)),
        enable_dynamic_competition=bool(options.get("enable_dynamic_competition", False)),
        enable_symbolic_portfolio=bool(options.get("enable_symbolic_portfolio", True)),
        symbolic_solver_arm_budget=int(options.get("symbolic_solver_arm_budget", 2)),
    )
    problem_contract = None
    if isinstance(payload.get("problem_contract"), Mapping):
        from .model_hypotheses import ProblemContract
        problem_contract = ProblemContract.from_payload(payload["problem_contract"])
    result = assistant.run(
        problem=str(payload["description"]), datasets=datasets,
        target=payload.get("target"), run_modeling=bool(options.get("run_modeling", True)),
        generate_plots=bool(options.get("generate_plots", True)),
        mechanistic_ir=payload.get("mechanistic_ir"), problem_images=payload.get("problem_images", []),
        problem_contract=problem_contract,
        dynamic_contract=payload.get("dynamic_contract"),
    ).to_dict()
    result["run_id"] = str(payload.get("run_id") or create_run_id())
    result["execution_policy"] = "main_research_spawn_worker"
    return result


__all__ = ["run_research_payload"]
