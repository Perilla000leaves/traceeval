"""任务集：可版本化的用例 + 可回放的轨迹。

设计上刻意**把用例与轨迹分开**：

- `suite`（用例集）只写「问题 + 期望调用的工具」，可以人工维护、可以 code review；
- `traces`（轨迹集）是某次运行录下来的真实调用序列。

这样别人不需要 API key 也能复现你的评测结果：拿你的 suite + traces 重新打分即可。
要对比自己的模型，只需要按同样格式录一份 traces——这就是本项目能做「公开基线」的原因。

输入校验用 **pydantic 的声明式模型**，不手写 if/raise：
行格式写错时报错自带字段路径与期望类型，也不用维护一套自己的校验逻辑。
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import adapters
from .metrics import CaseResult, score
from .trajectory import Status, Trajectory


class CaseRow(BaseModel):
    """用例集的一行。字段名与 jsonl 键一致。"""

    model_config = ConfigDict(validate_default=True)

    id: str = ""
    expected_tools: list[Any] | None = None
    question: str = ""
    match: str | None = None
    args_mode: str | None = None
    expect_no_calls: bool = False
    forbid_tools: list[str] = Field(default_factory=list)
    allow_duplicates: bool = False
    category: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("缺少 id")
        return text

    @field_validator("expected_tools")
    @classmethod
    def _expected_required(cls, value: list[Any] | None) -> list[Any]:
        if value is None:
            raise ValueError("缺少 expected_tools")
        return value


class TraceRow(BaseModel):
    """轨迹集的一行：`{id, format, payload, latency_ms?, cost_usd?}`。"""

    model_config = ConfigDict(validate_default=True)

    id: str = ""
    format: str = "calls"
    payload: list[Any] | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None

    @field_validator("id")
    @classmethod
    def _id_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("缺少 id")
        return text

    @field_validator("payload")
    @classmethod
    def _payload_required(cls, value: list[Any] | None) -> list[Any]:
        if value is None:
            raise ValueError("payload 必须是列表")
        return value


@dataclass
class Case:
    """已校验的用例。`match` / `args_mode` 可按用例覆盖全局默认值。"""

    case_id: str
    expected_tools: list[Any]
    question: str = ""
    match: str | None = None
    args_mode: str | None = None
    expect_no_calls: bool = False
    forbid_tools: list[str] = field(default_factory=list)
    allow_duplicates: bool = False
    category: str = ""
    tags: list[str] = field(default_factory=list)


def _read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        yield line_no, json.loads(raw)


def load_cases(path: str | Path) -> list[Case]:
    """读 jsonl 用例集；格式错误由 pydantic 报出字段路径。"""
    cases: list[Case] = []
    for line_no, row in _read_jsonl(path):
        try:
            parsed = CaseRow.model_validate(row)
        except ValidationError as error:
            raise ValueError(f"{path}:{line_no} 用例格式错误：{error}") from None
        cases.append(
            Case(
                case_id=parsed.id,
                expected_tools=list(parsed.expected_tools or []),
                question=parsed.question,
                match=parsed.match,
                args_mode=parsed.args_mode,
                expect_no_calls=parsed.expect_no_calls,
                forbid_tools=list(parsed.forbid_tools),
                allow_duplicates=parsed.allow_duplicates,
                category=parsed.category,
                tags=list(parsed.tags),
            )
        )
    return cases


def load_traces(path: str | Path) -> dict[str, Trajectory]:
    """读 jsonl 轨迹集。"""
    traces: dict[str, Trajectory] = {}
    for line_no, row in _read_jsonl(path):
        try:
            parsed = TraceRow.model_validate(row)
        except ValidationError as error:
            raise ValueError(f"{path}:{line_no} 轨迹格式错误：{error}") from None
        trajectory = adapters.parse(parsed.payload or [], parsed.format)
        trajectory.latency_ms = parsed.latency_ms
        trajectory.cost_usd = parsed.cost_usd
        traces[parsed.id] = trajectory
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
                fail_on_duplicates=fail_on_duplicates and not case.allow_duplicates,
                expect_no_calls=case.expect_no_calls,
                forbid_tools=case.forbid_tools,
            )
        )
    return results


__all__ = ["Case", "CaseRow", "TraceRow", "load_cases", "load_traces", "run"]
