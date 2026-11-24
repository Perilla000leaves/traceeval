"""委托 agentevals：LangChain 官方的轨迹匹配器，不自己实现第二套规则。"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("agentevals", reason="没装 agentevals，跳过委托测试")

from traceeval.integrations import AGENTEVALS_AVAILABLE, evaluate_trajectory  # noqa: E402
from traceeval.trajectory import Status  # noqa: E402


def _messages(*calls: tuple[str, dict]) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"c{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
                for index, (name, args) in enumerate(calls)
            ],
        }
    ]


def test_agentevals_is_available():
    assert AGENTEVALS_AVAILABLE is True


def test_unordered_match_passes_regardless_of_order():
    result = evaluate_trajectory(
        _messages(("b", {}), ("a", {})),
        _messages(("a", {}), ("b", {})),
        match="any_order",
    )
    assert result.status is Status.PASS
    assert result.metrics["engine"] == "agentevals"


def test_missing_tool_fails():
    result = evaluate_trajectory(
        _messages(("a", {})),
        _messages(("a", {}), ("b", {})),
        match="any_order",
    )
    assert result.status is Status.FAIL
    assert "agentevals" in result.reasons[0]


def test_keys_mode_allows_extra_actual_args():
    """keys 模式 → agentevals 的 superset：实际参数多出来不算错。"""
    result = evaluate_trajectory(
        _messages(("t", {"date": "2026-11-18", "limit": 10})),
        _messages(("t", {"date": "2026-11-18"})),
        match="exact",
        args_mode="keys",
    )
    assert result.status is Status.PASS


def test_full_mode_requires_exact_args():
    result = evaluate_trajectory(
        _messages(("t", {"date": "2026-11-18", "limit": 10})),
        _messages(("t", {"date": "2026-11-18"})),
        match="exact",
        args_mode="full",
    )
    assert result.status is Status.FAIL


def test_in_order_falls_back_to_builtin_since_agentevals_lacks_it():
    """agentevals 没有「子序列」语义，这一种退回本项目实现。"""
    result = evaluate_trajectory(
        _messages(("x", {}), ("a", {}), ("y", {}), ("b", {})),
        _messages(("a", {}), ("b", {})),
        match="in_order",
    )
    assert result.status is Status.PASS
    assert result.metrics["engine"] == "builtin"
