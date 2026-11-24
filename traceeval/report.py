"""报告：把逐条结果聚合成一份能贴进 issue 的数字。

**通过率必须和未判定率一起看。** 只报 `pass_rate` 的项目，往往把"没测的用例"要么算成失败
（数字难看），要么算成通过（数字好看），两种做法都让数字失去解释力。这里同时给出：

- `pass_rate`：只在本判定的用例里算（pass / (pass + fail)）；
- `undetermined_rate`：多少用例压根没测成——它本身就是评测质量的指标。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .metrics import CaseResult
from .trajectory import Status


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def summarize(results: Sequence[CaseResult]) -> dict[str, Any]:
    """聚合指标。"""
    total = len(results)
    passed = sum(1 for item in results if item.status is Status.PASS)
    failed = sum(1 for item in results if item.status is Status.FAIL)
    undetermined = total - passed - failed
    judged = passed + failed

    reason_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for item in results:
        for code in item.codes:
            reason_counts[code] += 1
            if len(examples[code]) < 3:
                examples[code].append(item.case_id)

    ghosts = sum(len(item.metrics.get("ghosts") or []) for item in results)
    latencies = [item.metrics["latency_ms"] for item in results if item.metrics.get("latency_ms")]
    costs = [item.metrics["cost_usd"] for item in results if item.metrics.get("cost_usd")]

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "undetermined": undetermined,
        "judged": judged,
        "pass_rate": round(100.0 * passed / judged, 1) if judged else None,
        "undetermined_rate": round(100.0 * undetermined / total, 1) if total else None,
        "ghost_calls": ghosts,
        "reasons": {
            code: {"count": count, "examples": examples[code]}
            for code, count in reason_counts.most_common()
        },
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "cost_usd_total": round(sum(costs), 6) if costs else None,
    }


def to_markdown(results: Sequence[CaseResult], *, title: str = "轨迹评测报告") -> str:
    """生成 markdown 报告：总览 + 失败归因 + 逐条明细。"""
    summary = summarize(results)
    pass_rate = "n/a" if summary["pass_rate"] is None else f"{summary['pass_rate']}%"
    undetermined = "-" if summary["undetermined_rate"] is None else f"{summary['undetermined_rate']}%"

    lines = [
        f"# {title}",
        "",
        "| 用例 | 通过 | 失败 | 未判定 | 通过率（已判定） | 未判定率 | 假成功调用 |",
        "|---|---|---|---|---|---|---|",
        f"| {summary['total']} | {summary['passed']} | {summary['failed']} | "
        f"{summary['undetermined']} | {pass_rate} | {undetermined} | {summary['ghost_calls']} |",
        "",
    ]

    if summary["reasons"]:
        lines += ["## 失败归因", "", "| 类型 | 数量 | 用例（最多 3 个） |", "|---|---|---|"]
        for code, info in summary["reasons"].items():
            lines.append(f"| `{code}` | {info['count']} | {', '.join(info['examples'])} |")
        lines.append("")

    if summary["latency_p50_ms"] is not None:
        lines += [
            "## 成本与时延",
            "",
            f"- P50 时延：{summary['latency_p50_ms']} ms",
            f"- P95 时延：{summary['latency_p95_ms']} ms",
        ]
        if summary["cost_usd_total"] is not None:
            lines.append(f"- 总成本：${summary['cost_usd_total']}")
        lines.append("")

    lines += ["## 逐条明细", "", "| 用例 | 结果 | 原因 |", "|---|---|---|"]
    for item in results:
        reason = "；".join(item.reasons) if item.reasons else "-"
        lines.append(f"| {item.case_id} | {item.status.value} | {reason} |")
    lines.append("")
    return "\n".join(lines)


def to_dicts(results: Sequence[CaseResult]) -> list[dict[str, Any]]:
    """逐条结果的 JSON 形态（供 CI 或下游工具消费）。"""
    return [
        {**asdict(item), "status": item.status.value, "passed": item.passed} for item in results
    ]


__all__ = ["summarize", "to_dicts", "to_markdown"]
