"""框架接入：LangChain 回调、OTel span、断言助手。"""
from __future__ import annotations

import pytest

from traceeval.integrations import TraceCallbackHandler, trajectory_from_otel_spans
from traceeval.testing import TrajectoryAssertionError, assert_tools
from traceeval.trajectory import Call, Trajectory

# ---------------------------------------------------------------- LangChain 回调


def test_handler_pairs_start_and_end_by_run_id():
    handler = TraceCallbackHandler()
    handler.on_tool_start({"name": "lookup_ticket"}, None, run_id="r1", inputs={"id": 1})
    handler.on_tool_end("ok", run_id="r1")

    trajectory = handler.trajectory()
    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].args == {"id": 1}
    assert trajectory.calls[0].result_ok is True


def test_handler_marks_error_as_failed_but_keeps_call():
    handler = TraceCallbackHandler()
    handler.on_tool_start({"name": "lookup_ticket"}, None, run_id="r1")
    handler.on_tool_error(RuntimeError("boom"), run_id="r1")

    trajectory = handler.trajectory()
    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].result_ok is False


def test_handler_treats_empty_output_as_no_result():
    """很多框架把异常吞成空串；空输出不能算拿到结果。"""
    handler = TraceCallbackHandler()
    handler.on_tool_start({"name": "t"}, None, run_id="r1")
    handler.on_tool_end("   ", run_id="r1")
    assert handler.trajectory().calls[0].result_ok is False


def test_handler_ignores_end_without_start():
    handler = TraceCallbackHandler()
    handler.on_tool_end("ok", run_id="ghost")
    assert handler.trajectory().calls == []


def test_handler_reset_clears_state():
    handler = TraceCallbackHandler()
    handler.on_tool_start({"name": "t"}, None, run_id="r1")
    handler.reset()
    assert handler.trajectory().calls == []


def test_handler_works_with_serialized_id_when_name_missing():
    """有的框架只给 serialized['id']，不一定会给 name。"""
    handler = TraceCallbackHandler()
    handler.on_tool_start({"id": "tool:lookup"}, None, run_id="r1")
    handler.on_tool_end("ok", run_id="r1")
    assert handler.trajectory().names() == ["tool:lookup"]


# ---------------------------------------------------------------- OTel


def test_otel_spans_are_read_with_genai_attributes():
    spans = [
        {
            "name": "execute_tool lookup_ticket",
            "attributes": {
                "gen_ai.tool.name": "lookup_ticket",
                "gen_ai.tool.call.arguments": '{"ticket_id": 8842}',
            },
            "status": {"code": "STATUS_CODE_OK"},
        }
    ]
    trajectory = trajectory_from_otel_spans(spans)

    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].args == {"ticket_id": 8842}
    assert trajectory.calls[0].result_ok is True


def test_otel_error_status_marks_failure():
    spans = [
        {
            "attributes": {"gen_ai.tool.name": "rebuild"},
            "status": {"code": "STATUS_CODE_ERROR"},
        }
    ]
    assert trajectory_from_otel_spans(spans).calls[0].result_ok is False


def test_otel_skips_spans_without_tool_name():
    """LLM span 与 chain span 没有 gen_ai.tool.name，必须跳过而不是记成 unknown。"""
    spans = [
        {"attributes": {"gen_ai.operation.name": "chat"}},
        {"attributes": {"gen_ai.tool.name": "t"}},
    ]
    assert trajectory_from_otel_spans(spans).names() == ["t"]


# ---------------------------------------------------------------- 断言助手


def test_assert_tools_passes():
    trajectory = Trajectory(calls=[Call("a", result_ok=True)])
    assert_tools(trajectory, ["a"])


def test_assert_tools_reports_expected_and_actual():
    trajectory = Trajectory(calls=[Call("b", result_ok=True)])
    with pytest.raises(TrajectoryAssertionError) as box:
        assert_tools(trajectory, ["a"])

    message = str(box.value)
    assert "期望" in message and "实际" in message and "原因" in message
    assert "b" in message


def test_assert_tools_rejects_ghost_call():
    """断言里同样要抓假成功，否则单测会放过空结果。"""
    trajectory = Trajectory(calls=[Call("a", result_ok=None)])
    with pytest.raises(TrajectoryAssertionError, match="假成功"):
        assert_tools(trajectory, ["a"])


def test_assert_tools_rejects_empty_expectation():
    with pytest.raises(TrajectoryAssertionError, match="无法断言"):
        assert_tools(Trajectory(calls=[]), [])
