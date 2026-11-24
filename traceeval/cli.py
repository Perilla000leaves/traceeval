"""命令行入口：`python -m traceeval run --suite ... --traces ...`。

默认只输出报告、**不因为失败而返回非零**——它是分析工具，不是门禁。
需要当门禁用时加 `--strict`（有失败即退出 1）或 `--fail-under 90`（通过率低于阈值退出 1）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import report, suite


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traceeval",
        description="评测 tool-calling agent 的调用轨迹（支持假成功与未判定检测）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="对一组轨迹跑评测")
    run_parser.add_argument("--suite", required=True, help="用例集 jsonl")
    run_parser.add_argument("--traces", required=True, help="轨迹集 jsonl")
    run_parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    run_parser.add_argument(
        "--match",
        choices=("exact", "in_order", "any_order"),
        default="any_order",
        help="工具序列的匹配语义（可按用例覆盖）",
    )
    run_parser.add_argument(
        "--args-mode",
        choices=("keys", "full"),
        default="keys",
        help="keys=只比期望提到的参数名（默认，容忍值漂移）；full=连值一起比",
    )
    run_parser.add_argument(
        "--allow-ghosts",
        action="store_true",
        help="不把「发了调用但没有返回」判成假成功（仅用于只有请求侧日志的场景）",
    )
    run_parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="不把重复调用同一工具判成失败",
    )
    run_parser.add_argument("--strict", action="store_true", help="有任何失败就以退出码 1 结束")
    run_parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="PCT",
        help="通过率（已判定范围内）低于该百分比时以退出码 1 结束",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cases = suite.load_cases(args.suite)
    traces = suite.load_traces(args.traces)
    results = suite.run(
        cases,
        traces,
        match=args.match,
        args_mode=args.args_mode,
        require_results=not args.allow_ghosts,
        fail_on_duplicates=not args.allow_duplicates,
    )
    summary = report.summarize(results)

    if args.format == "json":
        print(
            json.dumps(
                {"summary": summary, "results": report.to_dicts(results)},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(report.to_markdown(results))

    if args.strict and summary["failed"]:
        return 1
    if args.fail_under is not None:
        rate = summary["pass_rate"]
        if rate is None or rate < args.fail_under:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
