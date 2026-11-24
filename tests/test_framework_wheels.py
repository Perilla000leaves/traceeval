"""用框架自带的类型与校验，而不是手写解析/校验。

- ADK：真实的 `google.genai.types.Content / Part / FunctionCall` 对象
- OTel：真实的 `opentelemetry.sdk.trace.ReadableSpan`
- pydantic：用例/轨迹文件的声明式校验
"""
from __future__ import annotations

import json

import pytest

from traceeval.suite import CaseRow, TraceRow

# ---------------------------------------------------------------- ADK 真实对象


def test_adk_events_accept_real_types_objects():
    """不是 dict，而是 google.genai 的 Content / Part / FunctionCall 对象。"""
    types = pytest.importorskip("google.genai.types")
    from traceeval.integrations import trajectory_from_adk_events

    call = types.Part(
        function_call=types.FunctionCall(id="c1", name="lookup_ticket", args={"ticket_id": 8842})
    )
    response = types.Part(
        function_response=types.FunctionResponse(
            id="c1", name="lookup_ticket", response={"ok": True}
        )
    )
    events = [
        {"content": types.Content(role="model", parts=[call])},
        {"content": types.Content(role="user", parts=[response])},
    ]

    trajectory = trajectory_from_adk_events(events)
    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].args == {"ticket_id": 8842}
    assert trajectory.calls[0].result_ok is True


# ---------------------------------------------------------------- OTel 真实 span


def test_otel_reads_real_readable_span():
    """用真实的 TracerProvider 造一个 span，走 `.attributes` / `.status.status_code`。"""
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("execute_tool lookup_ticket") as span:
        span.set_attribute("gen_ai.tool.name", "lookup_ticket")
        span.set_attribute("gen_ai.tool.call.arguments", json.dumps({"ticket_id": 8842}))

    from traceeval.integrations import trajectory_from_otel_spans

    spans = exporter.get_finished_spans()
    trajectory = trajectory_from_otel_spans(spans)

    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].args == {"ticket_id": 8842}
    assert trajectory.calls[0].result_ok is True
    assert otel_trace is not None


def test_otel_reads_error_status_from_real_span():
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import Status as OtelStatus
    from opentelemetry.trace import StatusCode

    exporter = InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("execute_tool rebuild") as span:
        span.set_attribute("gen_ai.tool.name", "rebuild")
        span.set_status(OtelStatus(StatusCode.ERROR))

    from traceeval.integrations import trajectory_from_otel_spans

    trajectory = trajectory_from_otel_spans(exporter.get_finished_spans())
    assert trajectory.calls[0].result_ok is False


# ---------------------------------------------------------------- pydantic 校验


def test_case_row_rejects_missing_id():
    with pytest.raises(ValueError, match="缺少 id"):
        CaseRow.model_validate({"expected_tools": ["a"]})


def test_case_row_rejects_missing_expected_tools():
    with pytest.raises(ValueError, match="缺少 expected_tools"):
        CaseRow.model_validate({"id": "c1"})


def test_case_row_accepts_full_row():
    row = CaseRow.model_validate(
        {"id": " c1 ", "expected_tools": ["a"], "match": "exact", "tags": ["x"]}
    )
    assert row.id == "c1"
    assert row.match == "exact"


def test_trace_row_rejects_non_list_payload():
    with pytest.raises(ValueError):
        TraceRow.model_validate({"id": "c1", "payload": {"a": 1}})
