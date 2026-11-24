"""ADK 接入 + LangGraph 共用回调协议的验证。"""
from __future__ import annotations

from types import SimpleNamespace

from traceeval.integrations import (
    AdkTraceRecorder,
    TraceCallbackHandler,
    trajectory_from_adk_events,
)


def _tool(name: str):
    """模拟 ADK 的 BaseTool：只用到 .name 一个属性。"""
    return SimpleNamespace(name=name)


# ---------------------------------------------------------------- ADK 运行期


def test_adk_recorder_pairs_callback_by_tool_name():
    recorder = AdkTraceRecorder()
    recorder.before_tool_callback(_tool("lookup_ticket"), {"ticket_id": 8842})
    recorder.after_tool_callback(_tool("lookup_ticket"), result={"ok": 1})

    trajectory = recorder.trajectory()
    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].args == {"ticket_id": 8842}
    assert trajectory.calls[0].result_ok is True


def test_adk_callbacks_return_none_to_avoid_short_circuit():
    """ADK 里回调返回 dict 会短路并当结果用——录制器必须返回 None。"""
    recorder = AdkTraceRecorder()
    assert recorder.before_tool_callback(_tool("t"), {}) is None
    assert recorder.after_tool_callback(_tool("t"), result={"ok": 1}) is None
    assert recorder.on_tool_error_callback(_tool("t"), error=RuntimeError("boom")) is None


def test_adk_error_marks_failure_and_keeps_call():
    recorder = AdkTraceRecorder()
    recorder.before_tool_callback(_tool("rebuild"), {})
    recorder.on_tool_error_callback(_tool("rebuild"), error=RuntimeError("boom"))

    trajectory = recorder.trajectory()
    assert trajectory.names() == ["rebuild"]
    assert trajectory.calls[0].result_ok is False


def test_adk_empty_result_counts_as_no_result():
    """ADK 的失败常常是返回空 dict —— 不能当成拿到了结果。"""
    recorder = AdkTraceRecorder()
    recorder.before_tool_callback(_tool("t"), {})
    recorder.after_tool_callback(_tool("t"), result={})
    assert recorder.trajectory().calls[0].result_ok is False


def test_adk_same_tool_twice_pairs_fifo():
    recorder = AdkTraceRecorder()
    recorder.before_tool_callback(_tool("t"), {"i": 1})
    recorder.after_tool_callback(_tool("t"), result={"ok": 1})
    recorder.before_tool_callback(_tool("t"), {"i": 2})
    # 第二次没有 end → 应记为假成功，而不是覆盖第一次
    trajectory = recorder.trajectory()

    assert [call.result_ok for call in trajectory.calls] == [True, None]


def test_adk_end_without_start_is_ignored():
    recorder = AdkTraceRecorder()
    recorder.after_tool_callback(_tool("t"), result={"ok": 1})
    assert recorder.trajectory().calls == []


# ---------------------------------------------------------------- ADK 离线事件


def test_adk_events_pair_by_id():
    events = [
        {
            "content": {
                "parts": [
                    {"function_call": {"id": "c1", "name": "lookup_ticket", "args": {"id": 1}}}
                ]
            }
        },
        {
            "content": {
                "parts": [{"function_response": {"id": "c1", "name": "lookup_ticket", "response": {"ok": 1}}}]
            }
        },
    ]
    trajectory = trajectory_from_adk_events(events)

    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].args == {"id": 1}
    assert trajectory.calls[0].result_ok is True


def test_adk_events_pair_by_name_when_id_missing():
    events = [
        {"content": {"parts": [{"function_call": {"name": "t", "args": {"a": 1}}}]}},
        {"content": {"parts": [{"function_response": {"name": "t", "response": {"ok": 1}}}]}},
    ]
    trajectory = trajectory_from_adk_events(events)
    assert trajectory.calls[0].result_ok is True


def test_adk_events_skip_non_tool_parts():
    events = [
        {"content": {"parts": [{"text": "我在想下一步"}]}},
        {"content": {"parts": [{"function_call": {"name": "t"}}]}},
    ]
    assert trajectory_from_adk_events(events).names() == ["t"]


def test_adk_events_empty_response_is_no_result():
    events = [
        {"content": {"parts": [{"function_call": {"id": "c1", "name": "t"}}]}},
        {"content": {"parts": [{"function_response": {"id": "c1", "name": "t", "response": {}}}]}},
    ]
    assert trajectory_from_adk_events(events).calls[0].result_ok is False


# ---------------------------------------------------------------- LangGraph


def test_langgraph_uses_same_callback_protocol():
    """LangGraph 复用 LangChain 的回调协议，同一个 handler 即可录制。"""
    handler = TraceCallbackHandler()
    # LangGraph 会额外传 parent_run_id / tags / metadata，实现必须能吃下未知参数
    handler.on_tool_start(
        {"name": "lookup_ticket"},
        None,
        run_id="lg-1",
        parent_run_id="node-1",
        tags=["graph:step"],
        metadata={"langgraph_node": "tools"},
        inputs={"ticket_id": 8842},
    )
    handler.on_tool_end("ok", run_id="lg-1", parent_run_id="node-1")

    trajectory = handler.trajectory()
    assert trajectory.calls[0].name == "lookup_ticket"
    assert trajectory.calls[0].result_ok is True
