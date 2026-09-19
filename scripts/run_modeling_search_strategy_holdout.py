"""Freeze and compare exhaustive versus equal-topology-budget random search."""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from core.evaluation_freeze import create_evaluation_freeze,verify_evaluation_freeze
from core.modeling_benchmark_suite import run_compositional_search_strategy_comparison
from core.modeling_confirmation_suite import reserve_modeling_confirmation,complete_modeling_confirmation
from core.modeling_search_strategy_holdout import build_search_strategy_holdout
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=20260925);p.add_argument("--output",default="artifacts/modeling-search-strategy-holdout/confirmation-20260925.json");p.add_argument("--registry-dir",default="artifacts/modeling-confirmation/consumed");a=p.parse_args()
    marker=reserve_modeling_confirmation(a.registry_dir,protocol_id="modeling-search-strategy-holdout-v6",seed=a.seed)
    budget={"topology_evaluations_per_case":21,"max_wall_seconds_per_case":30,"parameter_max_nfev":1200}
    methods=("exhaustive-depth-two/v8","random-depth-three-equal-topology-budget/v8")
    freeze=create_evaluation_freeze(ROOT,protocol_id="modeling-search-strategy-holdout-v6",seed=a.seed,budget=budget,methods=methods)
    before=verify_evaluation_freeze(ROOT,freeze);cases=build_search_strategy_holdout(seed=a.seed)
    commitment=sha256(json.dumps([c.public_metadata() for c in cases],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    report=run_compositional_search_strategy_comparison(cases=cases,seed=a.seed)
    after=verify_evaluation_freeze(ROOT,freeze);status="completed_without_source_change" if before["status"]==after["status"]=="verified" else "invalidated"
    report["confirmation_protocol"]={"status":status,"seed":a.seed,"suite_commitment":commitment,"freeze":freeze,"freeze_before":before,"freeze_after":after,"case_count":len(cases),"structure_group_count":len({c.structure_group for c in cases}),"scope":"first_execution_after_freeze;remaining_depth_two_topologies;equal_topology_proposal_budget;in_process_resource_usage_not_comparable"}
    d=Path(a.output);d.parent.mkdir(parents=True,exist_ok=True);d.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");complete_modeling_confirmation(marker,outcome="completed" if status.startswith("completed") else "failed",report_path=str(d));print(json.dumps({"status":status,"summary":report["summary"],"effect":report["paired_effect"],"output":str(d)},ensure_ascii=False));return 0 if status.startswith("completed") else 2
if __name__=="__main__":raise SystemExit(main())
