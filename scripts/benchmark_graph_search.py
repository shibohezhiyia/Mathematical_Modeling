"""Compare no-mutation and grammar-search baselines on a shared graph bundle."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.graph_benchmark import BENCHMARK_VERSION, run_graph_benchmark
from core.graph_search_artifacts import BUNDLE_VERSION
from core.model_hypotheses import HypothesisValidationError, _keys, _require, decode_proposal


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="同一数据和初始图，等预算比较固定结构、通用搜索与诊断驱动搜索；不调用 API。")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--study", required=True)
    parser.add_argument("--compare-cache", action="store_true", help="增加关闭中间量缓存的语法搜索对照")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "workspace" / "graph_benchmarks")
    args = parser.parse_args(argv)
    try:
        with args.bundle.open("rb") as source:
            raw = source.read(256001)
        _require(len(raw) <= 256000, "benchmark_file_size_limit")
        bundle = decode_proposal(raw.decode("utf-8-sig"))
        _keys(bundle, {"schema_version", "problem", "experiment", "candidates", "budget"}, {"grammar_search"})
        _require(bundle["schema_version"] == BUNDLE_VERSION, "bundle_version_mismatch")
        if "grammar_search" in bundle:
            _require(type(bundle["grammar_search"]) is bool, "invalid_benchmark_switch")
        # Switches are deliberately supplied by this paired protocol, not by the
        # original single-arm bundle. Numeric quota fields are shared unchanged.
        _keys(bundle["budget"], set(), {"max_candidates", "max_patch_attempts", "max_evaluations",
            "per_candidate_evaluations", "wall_seconds", "reuse_intermediates"})
        budget = dict(bundle["budget"])
        if "reuse_intermediates" in budget:
            _require(type(budget["reuse_intermediates"]) is bool, "invalid_benchmark_switch")
            budget.pop("reuse_intermediates")
        diagnostic_hint = {"status": "proposal_not_executed", "hard_constraint": False,
                           "may_feed_search": True, "requires_current_validation": True,
                           "candidate_primitives": ["sinusoidal_basis", "variance_link"]}
        arms = [{"id": "fixed_structure", "grammar_search": False, "reuse_intermediates": True},
                {"id": "grammar_search", "grammar_search": True, "reuse_intermediates": True},
                {"id": "diagnostic_search", "grammar_search": True, "reuse_intermediates": True,
                 "diagnostic_hints": [diagnostic_hint]}]
        if args.compare_cache:
            arms.append({"id": "grammar_no_cache", "grammar_search": True, "reuse_intermediates": False})
        spec = {k: bundle[k] for k in ("problem", "experiment", "candidates")}
        spec.update(schema_version=BENCHMARK_VERSION, budget=budget, arms=arms)
        result, destination = run_graph_benchmark(spec, holdout_path=args.holdout, output_root=args.output_root,
                                                 study=args.study)
    except HypothesisValidationError as exc:
        print(f"对照协议未通过：{exc.code}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("无法读取或保存文件，请检查路径、权限与 UTF-8 编码。", file=sys.stderr)
        return 2
    print(f"留出检查通过 {result['accepted_arms']} / {len(result['arms'])} 个方法；失败方法未从分母删除。")
    print("这不是总体准确率或提速结论；没有使用留出数据评选冠军。")
    print(f"报告：{destination / 'reports' / 'benchmark.md'}")
    return 0 if result["accepted_arms"] == len(result["arms"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
