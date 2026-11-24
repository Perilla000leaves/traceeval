"""在测试里断言轨迹：把评测器当断言库用。

agent 的单测通常写「调了没调、顺序对不对」，手写比较既啰嗦又容易漏掉假成功。
这里直接复用打分逻辑，失败时报出可读原因。

用法::

    from traceeval.testing import assert_tools

    def test_agent_calls_lookup_first():
        result = run_agent("查一下这个工单")
        assert_tools(result.trajectory(), ["lookup_ticket"], match="in_order")
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .metrics import CaseResult, score
from .trajectory import Status, Trajectory


class TrajectoryAssertionError(AssertionError):
    """轨迹断言失败，消息里带实际调用与原因。"""


def assert_tools(
    actual: Trajectory,
    expected: Sequence[Any],
    *,
    match: str = "any_order",
    args_mode: str = "keys",
    require_results: bool = True,
    fail_on_duplicates: bool = True,
    case_id: str = "assertion",
) -> CaseResult:
    """断言轨迹满足期望；不满足则抛 `TrajectoryAssertionError`。

    与 `score()` 的区别只有一个：这里把失败变成异常，方便在 pytest 里直接当断言用。
    """
    result = score(
        case_id,
        expected,
        actual,
        match=match,
        args_mode=args_mode,
        require_results=require_results,
        fail_on_duplicates=fail_on_duplicates,
    )
    if result.status is Status.UNDETERMINED:
        raise TrajectoryAssertionError(
            "期望工具为空，无法断言：" + (result.reasons[0] if result.reasons else "")
        )
    if result.status is Status.FAIL:
        actual_names = " -> ".join(actual.names()) or "（没有调用任何工具）"
        raise TrajectoryAssertionError(
            "轨迹断言失败：\n"
            f"- 期望：{list(expected)}\n"
            f"- 实际：{actual_names}\n"
            f"- 原因：{'；'.join(result.reasons)}"
        )
    return result


__all__ = ["TrajectoryAssertionError", "assert_tools"]
