"""Confirm a pre-frozen panel, never search using final outcomes."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.graph_panel_artifacts import run_panel_confirmation
from core.model_hypotheses import HypothesisValidationError, decode_proposal


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description="确认已冻结的模型组；不调用 API、不重新拟合、不选冠军。")
    parser.add_argument("panel", type=Path, help="mathmodel.frozen-model-panel/v1 JSON")
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--study", required=True, help="稳定研究标识；改名不会使旧数据重新独立")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "workspace" / "graph_panel_runs")
    args = parser.parse_args(argv)
    try:
        with args.panel.open("rb") as source:
            raw = source.read(256001)
        if len(raw) > 256000:
            raise HypothesisValidationError("panel_file_size_limit")
        result, destination = run_panel_confirmation(decode_proposal(raw.decode("utf-8-sig")),
            holdout_path=args.holdout, study=args.study, output_root=args.output_root)
    except HypothesisValidationError as exc:
        print(f"面板校验未通过：{exc.code}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError):
        print("无法读取或保存文件，请检查路径、权限和 UTF-8 编码。", file=sys.stderr)
        return 2
    print("配对记录已保存；请逐项查看通过、失败和未完成状态。")
    print("本次不选择冠军；有限检验不证明现实或整体数学正确性。")
    print(f"报告：{destination / 'reports' / 'panel_confirmation.md'}")
    return 0 if result.get("all_arms_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
