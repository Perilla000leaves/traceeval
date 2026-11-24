"""打分：把「期望」与「轨迹」比成一个三态结果，并给出可归因的原因。

三条与其他评测工具不同、也是本项目主张的地方：

1. **假成功优先于参数错误**。模型发出了 tool_call 但没拿到返回，比"没调对工具"更危险——
   它看起来成功了，后续再基于这个空结果继续推理。所以先查假成功。
2. **未判定不等于失败**。没写期望工具的用例说明"没测"，把它算成失败会让通过率失去意义；
   把它算成通过更糟。这里单列成 `UNDETERMINED`，报告里同时给通过率和未判定率。
3. **归因要能落到一句话**。失败原因按「假成功 / 缺工具 / 参数缺 key / 重复调用」分类，
   而不是笼统给一个 0 分。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .trajectory import (
    Call,
    ExpectedCall,
    Status,
    Trajectory,
    as_expected,
    duplicates,
    match_names,
)

#: 失败原因分类（报告按这个聚合）
REASON_GHOST = "ghost_call"
REASON_MISSING = "missing_tool"
REASON_ORDER = "order_mismatch"
REASON_ARGS = "args_mismatch"
REASON_DUPLICATE = "duplicate_call"
REASON_ERROR = "run_error"


@dataclass
class CaseResult:
    """一条用例的判定结果。`metrics` 是给报告用的原始数字。"""

    case_id: str
    status: Status
    reasons: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS


def _args_problems(
    actual: Sequence[Call],
    expected: Sequence[ExpectedCall],
    args_mode: str,
) -> list[str]:
    """期望声明了参数的调用，实际是否满足。

    默认 `keys` 模式只看「期望提到的 key 在不在」，值会漂（日期、ID）不参与比较；
    `full` 模式才比较值。
    """
    problems: list[str] = []
    for want in expected:
        if not want.args:
            continue
        # 取第一个同名调用（重复调用本身由 duplicates 单独报告）
        candidates = [call for call in actual if call.name == want.name]
        if not candidates:
            continue  # 缺工具已由 missing 报告，不重复计
        given = candidates[0].args or {}
        missing_keys = sorted(set(want.args) - set(given))
        if missing_keys:
            problems.append(f"{want.name} 缺参数 {', '.join(missing_keys)}")
            continue
        if args_mode == "full":
            differing = sorted(
                key for key, value in want.args.items() if given.get(key) != value
            )
            if differing:
                problems.append(f"{want.name} 参数值与期望不符：{', '.join(differing)}")
    return problems


def score(
    case_id: str,
    expected_tools: Sequence[Any],
    actual: Trajectory,
    *,
    match: str = "any_order",
    args_mode: str = "keys",
    require_results: bool = True,
    fail_on_duplicates: bool = True,
    error: str = "",
) -> CaseResult:
    """给一条用例打分。

    `require_results=True` 时，没有返回的调用会被判成假成功；
    把它设为 False 可以评测"只看发了什么"的场景（例如只采到请求侧日志）。
    """
    expected = as_expected(expected_tools)
    calls = actual.calls
    names = actual.names()

    metrics: dict[str, Any] = {
        "expected": [item.name for item in expected],
        "actual": names,
        "latency_ms": actual.latency_ms,
        "cost_usd": actual.cost_usd,
    }

    if error:
        return CaseResult(
            case_id=case_id,
            status=Status.FAIL,
            reasons=[f"运行报错：{error[:120]}"],
            codes=[REASON_ERROR],
            metrics=metrics,
        )

    if not expected:
        # 没写期望 = 没测，不是没通过
        return CaseResult(
            case_id=case_id,
            status=Status.UNDETERMINED,
            reasons=["未提供期望工具，跳过判定"],
            metrics=metrics,
        )

    ghosts = actual.ghosts()
    repeated = duplicates(names)
    missing = [item.name for item in expected if item.name not in names]
    args_problems = _args_problems(calls, expected, args_mode)
    matched = match_names(names, [item.name for item in expected], match)

    metrics.update(
        {
            "matched": matched,
            "missing": missing,
            "ghosts": [call.name for call in ghosts],
            "duplicates": repeated,
            "args_problems": args_problems,
        }
    )

    reasons: list[str] = []
    codes: list[str] = []

    # 1) 假成功优先：看起来调了，其实是空的
    if require_results and ghosts:
        reasons.append(
            f"假成功：{len(ghosts)} 个调用没有返回（{', '.join(call.name for call in ghosts)}）"
        )
        codes.append(REASON_GHOST)
    # 2) 工具没调到（区分「缺工具」与「顺序/数量不符」——两者的修复方向不同）
    if not matched:
        if missing:
            reasons.append(f"未满足匹配（{match}）：缺 {', '.join(missing)}")
            codes.append(REASON_MISSING)
        else:
            reasons.append(f"未满足匹配（{match}）：工具都调了，但顺序或数量与期望不符")
            codes.append(REASON_ORDER)
    # 3) 参数不满足
    if args_problems:
        reasons.extend(args_problems)
        codes.append(REASON_ARGS)
    # 4) 重复调用
    if fail_on_duplicates and repeated:
        reasons.append(f"重复调用：{', '.join(repeated)}")
        codes.append(REASON_DUPLICATE)

    status = Status.FAIL if reasons else Status.PASS
    return CaseResult(case_id=case_id, status=status, reasons=reasons, codes=codes, metrics=metrics)


def score_many(
    cases: Sequence[tuple[str, Sequence[Any], Trajectory]],
    **kwargs: Any,
) -> list[CaseResult]:
    """批量打分（`cases` 是 `(case_id, expected_tools, trajectory)` 三元组）。"""
    return [score(case_id, expected, actual, **kwargs) for case_id, expected, actual in cases]


__all__ = [
    "REASON_ARGS",
    "REASON_DUPLICATE",
    "REASON_ERROR",
    "REASON_GHOST",
    "REASON_MISSING",
    "REASON_ORDER",
    "CaseResult",
    "score",
    "score_many",
]
