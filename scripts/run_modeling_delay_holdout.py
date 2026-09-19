"""Freeze then first-run delayed differential structure confirmation."""
from __future__ import annotations
import argparse,json
from hashlib import sha256
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.evaluation_freeze import create_evaluation_freeze,verify_evaluation_freeze
from core.modeling_benchmark_suite import current_problem_compiler_adapter,isolated_baseline_adapter,run_three_arm_modeling_comparison
from core.modeling_confirmation_suite import reserve_modeling_confirmation,complete_modeling_confirmation
from core.modeling_delay_holdout import build_delay_holdout
def dg(paths):
 h=sha256()
 for p in paths:h.update(p.read_bytes())
 return h.hexdigest()
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=20260926);p.add_argument("--output",default="artifacts/modeling-delay-holdout/confirmation-20260926.json");p.add_argument("--registry-dir",default="artifacts/modeling-confirmation/consumed");a=p.parse_args();m=reserve_modeling_confirmation(a.registry_dir,protocol_id="modeling-delay-holdout-v7",seed=a.seed)
 budget={"per_case_wall_seconds":30,"memory_mb":1024,"max_model_api_calls":0,"max_numerical_solver_calls":2,"max_manual_interventions":1};methods=("typed-interface-probe/v9","delay-feature-grammar/v9","transparent-tools/v9")
 f=create_evaluation_freeze(ROOT,protocol_id="modeling-delay-holdout-v7",seed=a.seed,budget=budget,methods=methods);before=verify_evaluation_freeze(ROOT,f);cases=build_delay_holdout(seed=a.seed);commit=sha256(json.dumps([c.public_metadata() for c in cases],sort_keys=True,separators=(",",":")).encode()).hexdigest();suite=ROOT/"core"/"modeling_benchmark_suite.py";sup=[ROOT/"core"/n for n in ("solver_runtime.py","solver_worker.py","solver_process_limits.py")]
 r=run_three_arm_modeling_comparison({"frozen_old":isolated_baseline_adapter("frozen_old"),"candidate_new":current_problem_compiler_adapter,"simple_tool_baseline":isolated_baseline_adapter("simple_tool_baseline")},cases=cases,fixed_budget={**budget,"seed":a.seed},system_versions={"frozen_old":{"version_id":methods[0],"source_digest":dg([suite,*sup])},"candidate_new":{"version_id":methods[1],"source_digest":dg([suite,ROOT/"core"/"automatic_modeling.py",ROOT/"core"/"dynamic_model_compiler.py",*sup])},"simple_tool_baseline":{"version_id":methods[2],"source_digest":dg([suite,*sup])}});after=verify_evaluation_freeze(ROOT,f);status="completed_without_source_change" if before["status"]==after["status"]=="verified" else "invalidated";r["confirmation_protocol"]={"status":status,"seed":a.seed,"suite_commitment":commit,"freeze":f,"freeze_before":before,"freeze_after":after,"case_count":len(cases),"structure_group_count":len(cases),"scope":"first_execution_after_freeze;sparse_delay_feature_grammar;internal_generator"};d=Path(a.output);d.parent.mkdir(parents=True,exist_ok=True);d.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8");complete_modeling_confirmation(m,outcome="completed" if status.startswith("completed") else "failed",report_path=str(d));print(json.dumps({"status":status,"summary":r["arm_summary"],"output":str(d)},ensure_ascii=False));return 0 if status.startswith("completed") else 2
if __name__=="__main__":raise SystemExit(main())
