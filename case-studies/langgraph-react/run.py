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
from data import TICKETS, reset, snapshot  # noqa: E402

from traceeval import TraceCallbackHandler, report, score  # noqa: E402
from traceeval.diagnose import diagnose, summary_by_code  # noqa: E402
from traceeval.diagnose import to_json as diagnose_json  # noqa: E402
from traceeval.diagnose import to_markdown as diagnose_md  # noqa: E402
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


def check_side_effects(task: dict[str, Any], snap: dict[str, Any]) -> tuple[bool, str]:
    """检查"到底做成了没有"，而不是听模型说自己做成了。"""
    rule = task.get("side_effects") or {}
    missing_closed = [tid for tid in (rule.get("expect_closed") or []) if tid not in snap["closed"]]
    if missing_closed:
        return False, f"应该关闭但没有关闭：{missing_closed}"
    missing_notified = [tid for tid in (rule.get("expect_notified") or []) if tid not in snap["notified"]]
    if missing_notified:
        return False, f"应该通知但没有通知：{missing_notified}"
    if rule.get("forbid_notify_missing_ticket"):
        bad = [tid for tid in snap["notified"] if tid not in TICKETS]
        if bad:
            return False, f"给不存在的工单发了通知：{bad}"
    return True, ""


def check_answer(task: dict[str, Any], final: str) -> tuple[bool, str]:
    """答复核验：调用对了不代表说对了。

    只做关键词级检查——这不是语义判定，case study 里要说清楚这个边界。
    """
    rule = task.get("answer_rules") or {}
    text = final or ""
    required = rule.get("require_any") or []
    if required and not any(phrase in text for phrase in required):
        return False, f"答复没说明真实原因（应含 {'/'.join(required)}）"
    forbidden = [phrase for phrase in (rule.get("forbid_any") or []) if phrase in text]
    if forbidden:
        return False, f"答复给出了错误结论（出现 {'/'.join(forbidden)}）"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="LangGraph ReAct case study")
    parser.add_argument("--variant", choices=("baseline", "fixed", "improved"), required=True)
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
            reset()  # 写操作任务必须从干净状态开始
            trajectory, final, error = run_once(agent, task["question"])
            snap = snapshot()
            side_ok, side_reason = check_side_effects(task, snap)
            answer_ok, answer_reason = check_answer(task, final)
            result = score(
                task["id"],
                task["expected_tools"],
                trajectory,
                match=task.get("match") or "any_order",
                args_mode=task.get("args_mode") or "keys",
                expect_no_calls=bool(task.get("expect_no_calls")),
                forbid_tools=task.get("forbid_tools") or (),
                fail_on_duplicates=not task.get("allow_duplicates", False),
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
                    "answer_ok": answer_ok,
                    "answer_reason": answer_reason,
                    "side_effect_ok": side_ok,
                    "side_effect_reason": side_reason,
                    "side_effects": snap,
                }
            )
            marker = ""
            if not side_ok:
                marker += "  [副作用违规] " + side_reason
            if not answer_ok:
                marker += "  [答复错误] " + answer_reason
            print(
                f"  {task['id']:<28} run{run_index + 1} {result.status.value:<13} "
                f"{' -> '.join(trajectory.names())}{marker}"
            )

    summary = report.summarize(results)
    summary["model"] = model_kind
    summary["variant"] = args.variant
    summary["runs_per_task"] = args.runs
    violations = [row for row in details if not row["side_effect_ok"]]
    summary["side_effect_violations"] = len(violations)
    summary["side_effect_violation_ids"] = sorted({row["id"] for row in violations})
    answer_violations = [row for row in details if not row["answer_ok"]]
    by_category: dict[str, dict[str, int]] = {}
    for task, result in zip(
        [task for task in tasks for _ in range(args.runs)], results, strict=False
    ):
        bucket = by_category.setdefault(
            task.get("category") or "other", {"passed": 0, "failed": 0, "undetermined": 0}
        )
        key = {"pass": "passed", "fail": "failed"}.get(result.status.value, "undetermined")
        bucket[key] += 1
    summary["by_category"] = by_category
    summary["answer_violations"] = len(answer_violations)
    summary["answer_violation_ids"] = sorted({row["id"] for row in answer_violations})

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
    diagnoses = diagnose(results)
    (OUT / f"diagnosis_{tag}.json").write_text(diagnose_json(diagnoses), encoding="utf-8")
    (OUT / f"diagnosis_{tag}.md").write_text(diagnose_md(diagnoses), encoding="utf-8")

    print()
    print(f"variant={args.variant} model={model_kind} runs={args.runs}")
    print(json.dumps({k: summary[k] for k in ("total", "passed", "failed", "undetermined", "pass_rate", "undetermined_rate", "ghost_calls")}, ensure_ascii=False))
    print("按类别:", json.dumps(summary.get("by_category", {}), ensure_ascii=False))
    print("失败码分布:", json.dumps(summary_by_code(results), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
