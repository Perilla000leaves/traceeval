"""适配器：三种消息形态都要能正确按 id 配对，假成功不能丢。"""
from __future__ import annotations

import pytest

from traceeval.adapters import from_anthropic, from_calls, from_events, from_openai, parse


def test_openai_pairs_tool_call_with_result():
    messages = [
        {"role": "user", "content": "查一下"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "lookup_ticket", "arguments": '{"id": 1}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    trajectory = from_openai(messages)

    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].result_ok is True
    assert trajectory.ghosts() == []


def test_openai_keeps_ghost_when_no_tool_message():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "lookup_ticket", "arguments": "{}"}}],
        }
    ]
    trajectory = from_openai(messages)

    assert trajectory.names() == ["lookup_ticket"]
    assert [call.name for call in trajectory.ghosts()] == ["lookup_ticket"]


def test_openai_empty_content_counts_as_no_result():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "t"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "   "},
    ]
    trajectory = from_openai(messages)
    assert trajectory.calls[0].result_ok is False


def test_anthropic_pairs_tool_use_and_tool_result():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "u1", "name": "lookup_ticket", "input": {"id": 1}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "ok"}]},
    ]
    trajectory = from_anthropic(messages)

    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].args == {"id": 1}
    assert trajectory.calls[0].result_ok is True


def test_anthropic_error_result_is_not_ok():
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "t"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "u1", "is_error": True, "content": "boom"}],
        },
    ]
    trajectory = from_anthropic(messages)
    assert trajectory.calls[0].result_ok is False


def test_events_pair_by_id():
    events = [
        {"type": "tool_call", "id": "e1", "name": "t", "args": {"a": 1}},
        {"type": "tool_result", "id": "e1", "ok": True},
        {"type": "tool_call", "id": "e2", "name": "t2"},
    ]
    trajectory = from_events(events)

    assert trajectory.names() == ["t", "t2"]
    assert trajectory.calls[0].result_ok is True
    assert trajectory.calls[1].result_ok is None


def test_calls_format_reads_result_flag():
    trajectory = from_calls([{"name": "t", "args": {"a": 1}, "result_ok": False}])
    assert trajectory.calls[0].result_ok is False


def test_parse_rejects_unknown_format():
    with pytest.raises(ValueError, match="未知轨迹格式"):
        parse([], "langsmith")


def test_parse_rejects_non_list_payload():
    with pytest.raises(ValueError, match="需要一个消息或事件列表"):
        parse({"a": 1}, "openai")
