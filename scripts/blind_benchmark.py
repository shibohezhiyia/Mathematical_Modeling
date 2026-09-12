"""管理封存真实未见题基准，不执行题目求解。

典型流程：
1. ``seal`` 在仓库外生成公开哈希清单，题面/答案和令牌留在安全目录；
2. ``scan`` 在运行前检查公开仓库有没有题面泄漏；
3. ``record`` 以清单中的固定预算记录一次运行；
4. ``score`` 运行结束后由独立评测者使用仓外令牌解封；
5. ``report`` 生成解封前后都不夸大证据的汇总；
6. ``statistics`` 在满足独立真实未见题门槛后计算准确率和配对显著性。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.blind_benchmark import (
    BlindBenchmarkError,
    BlindManifest,
    build_blind_report,
    build_sealed_case,
    record_blind_run,
    scan_leakage,
    unlock_and_score,
)
from core.blind_statistics import BlindStatisticsError, assess_blind_accuracy


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _budget(args):
    return {
        "max_seconds": args.max_seconds, "max_memory_mb": args.max_memory_mb,
        "max_api_calls": args.max_api_calls, "max_candidates": args.max_candidates,
        "seed": args.seed,
    }


def _cmd_seal(args) -> int:
    case, token = build_sealed_case(
        case_id=args.case_id, family=args.family, statement_path=args.statement,
        attachments=args.attachments, answer_path=args.answer, split=args.split,
        provenance=args.provenance, tags=args.tags, unlock_token=args.unlock_token,
    )
    manifest = BlindManifest.create([case], budget=_budget(args), name=args.name, revision=args.revision)
    _write(args.output, manifest.public())
    if args.token_file:
        args.token_file.parent.mkdir(parents=True, exist_ok=True)
        args.token_file.write_text(token + "\n", encoding="utf-8")
    else:
        print("unlock_token=" + token)
    print(json.dumps({"status": "sealed", "manifest": str(args.output), "digest": manifest.digest,
                      "case_id": case.case_id, "token_written": bool(args.token_file)}, ensure_ascii=False))
    return 0


def _cmd_scan(args) -> int:
    manifest = BlindManifest.from_payload(_json(args.manifest))
    # 公开清单本身必然包含 case_id 和摘要，扫描时自动排除它；用户仍可
    # 通过 --exclude 排除其他公开运行/报告元数据。
    excludes = [*args.exclude, args.manifest]
    result = scan_leakage(args.project_root, manifest, exclude_paths=excludes,
                          protected_files=args.protected)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "pass" else 2


def _cmd_record(args) -> int:
    manifest = BlindManifest.from_payload(_json(args.manifest))
    result = record_blind_run(
        manifest, case_id=args.case_id, run_id=args.run_id, status=args.status,
        budget=_budget(args), system_version=args.system_version,
        artifact_digest=args.artifact_digest, output_digest=args.output_digest,
        elapsed_seconds=args.elapsed_seconds, api_calls=args.api_calls,
        candidate_count=args.candidate_count, failure_code=args.failure_code,
    )
    _write(args.output, result)
    print(json.dumps({"status": "recorded", "output": str(args.output)}, ensure_ascii=False))
    return 0


def _cmd_score(args) -> int:
    manifest = BlindManifest.from_payload(_json(args.manifest))
    run = _json(args.run)
    reference = _json(args.reference)
    token = args.token_file.read_text(encoding="utf-8").strip()
    result = unlock_and_score(manifest, run, reference=reference, unlock_token=token)
    _write(args.output, result)
    print(json.dumps({"status": "scored", "output": str(args.output)}, ensure_ascii=False))
    return 0


def _cmd_report(args) -> int:
    manifest = BlindManifest.from_payload(_json(args.manifest))
    runs = [_json(path) for path in args.runs]
    scores = [_json(path) for path in args.scores]
    result = build_blind_report(manifest, runs, scores)
    _write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_statistics(args) -> int:
    manifest = BlindManifest.from_payload(_json(args.manifest))
    runs = [_json(path) for path in args.runs]
    scores = [_json(path) for path in args.scores]
    result = assess_blind_accuracy(
        manifest, runs, scores,
        baseline_system=args.baseline_system,
        treatment_system=args.treatment_system,
        primary_score=args.primary_score,
        success_threshold=args.success_threshold,
        min_cases=args.min_cases,
        alpha=args.alpha,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.statistics_seed,
        independent_evaluation_attested=args.independent_evaluation_attested,
        failure_as_incorrect=not args.exclude_failures,
    )
    _write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _common_budget(parser):
    parser.add_argument("--max-seconds", type=int, default=300)
    parser.add_argument("--max-memory-mb", type=int, default=1024)
    parser.add_argument("--max-api-calls", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)


def main(argv=None) -> int:
    # Windows 默认代码页可能无法输出中文帮助；命令仍保持机器可读 JSON。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="封存真实未见题基准管理工具")
    sub = parser.add_subparsers(dest="command", required=True)

    seal = sub.add_parser("seal", help="只写哈希清单，不复制题面")
    seal.add_argument("--case-id", required=True); seal.add_argument("--family", required=True)
    seal.add_argument("--statement", type=Path, required=True); seal.add_argument("--answer", type=Path)
    seal.add_argument("--attachments", type=Path, nargs="*", default=[])
    seal.add_argument("--split", choices=("unseen", "adversarial", "structure_transform"), default="unseen")
    seal.add_argument("--provenance", choices=("external_real", "synthetic_fixture"), default="external_real")
    seal.add_argument("--tags", nargs="*", default=[]); seal.add_argument("--name", default="real-unseen-benchmark")
    seal.add_argument("--revision", type=int, default=1); seal.add_argument("--unlock-token")
    seal.add_argument("--token-file", type=Path); seal.add_argument("--output", type=Path, required=True)
    _common_budget(seal); seal.set_defaults(func=_cmd_seal)

    scan = sub.add_parser("scan", help="扫描仓库泄漏")
    scan.add_argument("--manifest", type=Path, required=True); scan.add_argument("--project-root", type=Path, required=True)
    scan.add_argument("--exclude", type=Path, nargs="*", default=[]); scan.add_argument("--protected", type=Path, nargs="*", default=[])
    scan.set_defaults(func=_cmd_scan)

    record = sub.add_parser("record", help="记录固定预算运行")
    record.add_argument("--manifest", type=Path, required=True); record.add_argument("--case-id", required=True)
    record.add_argument("--run-id", required=True); record.add_argument("--status", choices=("completed", "failed", "blocked", "timed_out", "not_run"), required=True)
    record.add_argument("--system-version", required=True); record.add_argument("--artifact-digest"); record.add_argument("--output-digest")
    record.add_argument("--elapsed-seconds", type=float); record.add_argument("--api-calls", type=int, default=0); record.add_argument("--candidate-count", type=int, default=0); record.add_argument("--failure-code")
    record.add_argument("--output", type=Path, required=True); _common_budget(record); record.set_defaults(func=_cmd_record)

    score = sub.add_parser("score", help="解封并核验参考解承诺")
    score.add_argument("--manifest", type=Path, required=True); score.add_argument("--run", type=Path, required=True); score.add_argument("--reference", type=Path, required=True); score.add_argument("--token-file", type=Path, required=True); score.add_argument("--output", type=Path, required=True); score.set_defaults(func=_cmd_score)

    report = sub.add_parser("report", help="汇总运行和解封分数")
    report.add_argument("--manifest", type=Path, required=True); report.add_argument("--runs", type=Path, nargs="*", default=[]); report.add_argument("--scores", type=Path, nargs="*", default=[]); report.add_argument("--output", type=Path, required=True); report.set_defaults(func=_cmd_report)

    statistics = sub.add_parser("statistics", help="计算真实未见题准确率与配对显著性")
    statistics.add_argument("--manifest", type=Path, required=True)
    statistics.add_argument("--runs", type=Path, nargs="+", required=True)
    statistics.add_argument("--scores", type=Path, nargs="+", required=True)
    statistics.add_argument("--baseline-system", required=True)
    statistics.add_argument("--treatment-system", required=True)
    statistics.add_argument("--primary-score", default="numerically_correct")
    statistics.add_argument("--success-threshold", type=float, default=0.5)
    statistics.add_argument("--min-cases", type=int, default=20)
    statistics.add_argument("--alpha", type=float, default=0.05)
    statistics.add_argument("--bootstrap-replicates", type=int, default=2000)
    statistics.add_argument("--statistics-seed", type=int, default=20260910)
    statistics.add_argument("--independent-evaluation-attested", action="store_true")
    statistics.add_argument("--exclude-failures", action="store_true", help="缺失/失败格不按错误计入时保持 not_assessed")
    statistics.add_argument("--output", type=Path, required=True)
    statistics.set_defaults(func=_cmd_statistics)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, UnicodeError, json.JSONDecodeError, BlindBenchmarkError, BlindStatisticsError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
