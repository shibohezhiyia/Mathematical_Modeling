"""Run an explicitly bound typed-graph experiment; no model API calls."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.graph_search_artifacts import confirmation_status_label, run_search_bundle
from core.model_hypotheses import HypothesisValidationError, decode_proposal


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="运行本地、显式绑定的数学图实验；不会调用模型 API。")
    parser.add_argument("input", type=Path, help="mathmodel.graph-search-bundle/v1 JSON 文件")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "workspace" / "graph_search_runs")
    parser.add_argument("--confirm", type=Path, help="冻结候选后再读取的留出 JSON 文件")
    parser.add_argument("--study", help="持久的研究标识；重新命名研究不会使旧数据重新独立")
    parser.add_argument("--candidate", help="冻结的完整候选指纹；有多个 Pareto 候选时必填")
    args = parser.parse_args(argv)
    if args.confirm is not None and not args.study:
        parser.error("--confirm 必须同时提供稳定的 --study 标识")
    try:
        with args.input.open("rb") as source:
            raw = source.read(256001)
        if len(raw) > 256000:
            raise HypothesisValidationError("bundle_size_limit")
        bundle = decode_proposal(raw.decode("utf-8-sig"))
        result, directory = run_search_bundle(bundle, output_root=args.output_root, confirmation_path=args.confirm,
                                             study=args.study, candidate_hash=args.candidate)
    except HypothesisValidationError as exc:
        print(f"实验校验未通过：{exc.code}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("无法读取或保存实验文件，请检查路径、权限和 UTF-8 编码。", file=sys.stderr)
        return 2
    print(f"搜索结束：{result['termination']}；待确认候选 {len(result['pareto_candidates'])} 个。")
    if result.get("confirmation"):
        print(confirmation_status_label(result["confirmation"]))
        print("留出阶段不重新拟合参数；有限检查不证明整体正确性。")
    else:
        print("最终独立确认未执行；有限检查通过不等于现实或数学证明。")
    print(f"运行目录：{directory}")
    if result.get("confirmation"):
        return 0 if result["confirmation"]["status"] == "passed_finite_heldout_checks" else 1
    return 0 if result["pareto_candidates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
