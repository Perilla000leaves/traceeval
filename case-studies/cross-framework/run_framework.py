"""跑单一框架的全量任务：`python run_framework.py --framework lg --runs 1`。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from checks import check_answer, check_side_effects  # noqa: E402
from data import TICKETS, reset, snapshot  # noqa: E402
from frameworks import BUILDERS, run_task  # noqa: E402
from tools_common import load_tasks  # noqa: E402

from traceeval import report, score  # noqa: E402
from traceeval.diagnose import diagnose, summary_by_code  # noqa: E402
from traceeval.diagnose import to_markdown as diagnose_md  # noqa: E402

OUT = HERE / "out"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=tuple(BUILDERS), required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    OUT.mkdir(exist_ok=True)
    agent = BUILDERS[args.framework]()

    results = []
    details: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for task in tasks:
        for run_index in range(args.runs):
            reset()
            trajectory, final, error = run_task(agent, args.framework, task["question"])
            snap = snapshot()
            side_ok, side_reason = check_side_effects(task, snap, set(TICKETS))
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
                error=error,
            )
            results.append(result)
            details.append(
                {
                    "id": task["id"],
                    "run": run_index + 1,
                    "status": result.status.value,
                    "calls": trajectory.names(),
                    "reasons": result.reasons,
                    "final": final[:300],
                    "answer_ok": answer_ok,
                    "answer_reason": answer_reason,
                    "side_effect_ok": side_ok,
                    "side_effect_reason": side_reason,
                    "side_effects": snap,
                    "error": error,
                }
            )
            traces.append(
                {
                    "id": task["id"],
                    "format": "calls",
                    "latency_ms": trajectory.latency_ms,
                    "payload": [
                        {"name": c.name, "args": c.args, "result_ok": c.result_ok} for c in trajectory.calls
                    ],
                }
            )
            flag = ""
            if not answer_ok:
                flag += "  [答复] " + answer_reason
            if not side_ok:
                flag += "  [副作用] " + side_reason
            print(
                f"  {task['id']:<28} run{run_index + 1} {result.status.value:<13} "
                f"{' -> '.join(trajectory.names())}{flag}"
            )

    summary = report.summarize(results)
    summary["framework"] = args.framework
    summary["runs_per_task"] = args.runs
    summary["answer_violations"] = sum(1 for row in details if not row["answer_ok"])
    summary["answer_violation_ids"] = sorted({row["id"] for row in details if not row["answer_ok"]})
    summary["side_effect_violations"] = sum(1 for row in details if not row["side_effect_ok"])
    summary["side_effect_violation_ids"] = sorted({row["id"] for row in details if not row["side_effect_ok"]})
    summary["failures_by_code"] = summary_by_code(results)

    (OUT / f"summary_{args.framework}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"details_{args.framework}.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"traces_{args.framework}.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in traces) + "\n", encoding="utf-8"
    )
    (OUT / f"diagnosis_{args.framework}.md").write_text(diagnose_md(diagnose(results)), encoding="utf-8")

    print()
    print(
        json.dumps(
            {k: summary[k] for k in ("framework", "total", "passed", "failed", "pass_rate", "answer_violations", "side_effect_violations")},
            ensure_ascii=False,
        )
    )
    print("失败码:", json.dumps(summary["failures_by_code"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
