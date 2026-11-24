"""跑 case study：对每个任务调用 agent，录轨迹，再用 traceeval 打分。

产物（`out/`，不进仓库）：

- `suite.jsonl`：用例集（只写期望，不含模型输出）
- `traces_<variant>.jsonl`：录下来的真实轨迹（**有它就能离线复现打分，不需要 key**）
- `report_<variant>.md` / `summary_<variant>.json`：评测结果

用法::

    python run.py --variant baseline --runs 3
    python run.py --variant fixed --runs 3
    python compare.py            # 生成对比表
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # data.py / tools_*.py

from agent import build_agent, load_api_key  # noqa: E402

from traceeval import TraceCallbackHandler, report, score  # noqa: E402
from traceeval.trajectory import Trajectory  # noqa: E402

OUT = HERE / "out"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_once(agent: Any, question: str) -> tuple[Trajectory, str, str]:
    """跑一次，返回 (轨迹, 最终回复, 错误)。"""
    handler = TraceCallbackHandler()
    started = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [("user", question)]},
            config={"callbacks": [handler], "recursion_limit": 16},
        )
        elapsed = (time.perf_counter() - started) * 1000
        final = result["messages"][-1].content
        trajectory = handler.trajectory()
        trajectory.latency_ms = round(elapsed, 1)
        return trajectory, str(final), ""
    except Exception as error:  # noqa: BLE001 - 要把失败记进结果，而不是中断整批
        elapsed = (time.perf_counter() - started) * 1000
        trajectory = handler.trajectory()
        trajectory.latency_ms = round(elapsed, 1)
        return trajectory, "", f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description="LangGraph ReAct case study")
    parser.add_argument("--variant", choices=("baseline", "fixed"), required=True)
    parser.add_argument("--tasks", default=str(HERE / "tasks.jsonl"))
    parser.add_argument("--runs", type=int, default=1, help="每个任务重复跑几次")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    if args.limit:
        tasks = tasks[: args.limit]

    model_kind = "scripted" if os.environ.get("TRACEEVAL_MODEL", "").lower() == "fake" else "deepseek"
    if model_kind != "scripted" and not load_api_key():
        print("缺少 DEEPSEEK_API_KEY，见 agent.py 顶部的说明", file=sys.stderr)
        return 2

    OUT.mkdir(exist_ok=True)
    agent = build_agent(args.variant)

    results = []
    trace_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for task in tasks:
        for run_index in range(args.runs):
            trajectory, final, error = run_once(agent, task["question"])
            result = score(
                task["id"],
                task["expected_tools"],
                trajectory,
                match=task.get("match") or "any_order",
            )
            if error and result.status.value != "fail":
                result = score(task["id"], task["expected_tools"], trajectory, error=error)
            results.append(result)
            trace_rows.append(
                {
                    "id": task["id"],
                    "format": "calls",
                    "latency_ms": trajectory.latency_ms,
                    "payload": [
                        {"name": call.name, "args": call.args, "result_ok": call.result_ok}
                        for call in trajectory.calls
                    ],
                }
            )
            details.append(
                {
                    "id": task["id"],
                    "run": run_index + 1,
                    "status": result.status.value,
                    "calls": trajectory.names(),
                    "reasons": result.reasons,
                    "final": final[:200],
                    "error": error,
                }
            )
            print(f"  {task['id']:<28} run{run_index + 1} {result.status.value:<13} {' -> '.join(trajectory.names())}")

    summary = report.summarize(results)
    summary["model"] = model_kind
    summary["variant"] = args.variant
    summary["runs_per_task"] = args.runs

    (OUT / "suite.jsonl").write_text(
        "\n".join(json.dumps(task, ensure_ascii=False) for task in tasks) + "\n", encoding="utf-8"
    )
    tag = f"{args.variant}_{model_kind}"
    (OUT / f"traces_{tag}.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in trace_rows) + "\n", encoding="utf-8"
    )
    (OUT / f"summary_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / f"details_{tag}.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / f"report_{tag}.md").write_text(report.to_markdown(results), encoding="utf-8")

    print()
    print(f"variant={args.variant} model={model_kind} runs={args.runs}")
    print(json.dumps({k: summary[k] for k in ("total", "passed", "failed", "undetermined", "pass_rate", "undetermined_rate", "ghost_calls")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
