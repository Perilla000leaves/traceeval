"""任务集：可版本化的用例 + 可回放的轨迹。

设计上刻意**把用例与轨迹分开**：

- `suite`（用例集）只写「问题 + 期望调用的工具」，可以人工维护、可以 code review；
- `traces`（轨迹集）是某次运行录下来的真实调用序列。

这样别人不需要 API key 也能复现你的评测结果：拿你的 suite + traces 重新打分即可。
要对比自己的模型，只需要按同样格式录一份 traces——这就是本项目能做「公开基线」的原因。
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import adapters
from .metrics import CaseResult, score
from .trajectory import Status, Trajectory


@dataclass
class Case:
    """一条用例。`match` / `args_mode` 可按用例覆盖全局默认值。"""

    case_id: str
    expected_tools: list[Any]
    question: str = ""
    match: str | None = None
    args_mode: str | None = None
    tags: list[str] = field(default_factory=list)


def load_cases(path: str | Path) -> list[Case]:
    """读 jsonl 用例集；缺 id 或期望工具的条目直接报错（避免静默跳过）。"""
    cases: list[Case] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        case_id = str(row.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"{path}:{line_no} 缺少 id")
        expected = row.get("expected_tools")
        if expected is None:
            raise ValueError(f"{path}:{line_no}（{case_id}）缺少 expected_tools")
        cases.append(
            Case(
                case_id=case_id,
                expected_tools=list(expected),
                question=str(row.get("question") or ""),
                match=row.get("match"),
                args_mode=row.get("args_mode"),
                tags=list(row.get("tags") or []),
            )
        )
    return cases


def load_traces(path: str | Path) -> dict[str, Trajectory]:
    """读 jsonl 轨迹集：每行 `{id, format, payload, latency_ms?, cost_usd?}`。"""
    traces: dict[str, Trajectory] = {}
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        case_id = str(row.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"{path}:{line_no} 缺少 id")
        fmt = str(row.get("format") or "calls")
        payload = row.get("payload")
        if not isinstance(payload, (list, tuple)):
            raise ValueError(f"{path}:{line_no}（{case_id}）payload 必须是列表")
        trajectory = adapters.parse(payload, fmt)
        trajectory.latency_ms = row.get("latency_ms")
        trajectory.cost_usd = row.get("cost_usd")
        traces[case_id] = trajectory
    return traces


def run(
    cases: Iterable[Case],
    traces: Mapping[str, Trajectory],
    *,
    match: str = "any_order",
    args_mode: str = "keys",
    require_results: bool = True,
    fail_on_duplicates: bool = True,
) -> list[CaseResult]:
    """按用例逐条打分；**缺轨迹的用例记成未判定**，不要算成失败。"""
    results: list[CaseResult] = []
    for case in cases:
        trajectory = traces.get(case.case_id)
        if trajectory is None:
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    status=Status.UNDETERMINED,
                    reasons=["缺该用例的轨迹记录"],
                )
            )
            continue
        results.append(
            score(
                case.case_id,
                case.expected_tools,
                trajectory,
                match=case.match or match,
                args_mode=case.args_mode or args_mode,
                require_results=require_results,
                fail_on_duplicates=fail_on_duplicates,
            )
        )
    return results


__all__ = ["Case", "load_cases", "load_traces", "run"]
