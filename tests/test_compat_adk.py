"""可选兼容测试：装了 google-adk 时，验证真实工具对象的回调协议。

平时跳过——使用者不需要装 ADK；CI 里单独一个 job 装上跑这组，
防止 ADK 改回调签名后我们的录制器悄悄失效。
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("google.adk", reason="没装 google-adk，跳过兼容测试")

from google.adk.tools import FunctionTool  # noqa: E402

from traceeval.integrations import AdkTraceRecorder, trajectory_from_adk_events  # noqa: E402


def _search(query: str) -> dict:
    """一个最小可用的 ADK 工具实现。"""
    return {"ok": True, "query": query}


def test_recorder_reads_name_from_real_function_tool():
    """ADK 传的是 BaseTool 实例，录制器只取 .name —— 用真实对象验证这一点。"""
    tool = FunctionTool(_search)
    recorder = AdkTraceRecorder()

    recorder.before_tool_callback(tool, {"query": "x"})
    recorder.after_tool_callback(tool, result={"ok": True})

    trajectory = recorder.trajectory()
    assert trajectory.names() == [tool.name]
    assert trajectory.calls[0].args == {"query": "x"}
    assert trajectory.calls[0].result_ok is True


def test_callback_signatures_accept_adk_ordering():
    """ADK 按位置传 (tool, args, tool_context)；关键字传 result / error。"""
    recorder = AdkTraceRecorder()

    before = inspect.signature(recorder.before_tool_callback)
    assert list(before.parameters)[:3] == ["tool", "args", "tool_context"]

    after = inspect.signature(recorder.after_tool_callback)
    assert "result" in after.parameters

    error = inspect.signature(recorder.on_tool_error_callback)
    assert "error" in error.parameters


def test_events_parser_handles_function_tool_name():
    """事件流里的 function_call.name 与真实工具名一致。"""
    tool = FunctionTool(_search)
    events = [
        {"content": {"parts": [{"function_call": {"id": "c1", "name": tool.name, "args": {"query": "x"}}}]}},
        {
            "content": {
                "parts": [{"function_response": {"id": "c1", "name": tool.name, "response": {"ok": True}}}]
            }
        },
    ]
    trajectory = trajectory_from_adk_events(events)
    assert trajectory.names() == [tool.name]
    assert trajectory.calls[0].result_ok is True
