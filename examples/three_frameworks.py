"""三个框架接入同一个评测器的对照 demo。

同一份期望（`lookup_ticket` → `lookup_customer`），三条来源不同的轨迹：

1. LangChain：`TraceCallbackHandler` 挂在 callbacks 上；
2. LangGraph：同一套回调协议（额外带 parent_run_id / tags / metadata）；
3. ADK：`AdkTraceRecorder` 挂在 before/after_tool_callback 上。

运行：`python examples/three_frameworks.py`
"""
from __future__ import annotations

from types import SimpleNamespace

from traceeval import AdkTraceRecorder, TraceCallbackHandler, score

EXPECTED = ["lookup_ticket", "lookup_customer"]


def langchain_run() -> TraceCallbackHandler:
    """模拟一次 LangChain agent 调用（这里直接触发回调，不需要装 langchain）。"""
    handler = TraceCallbackHandler()
    handler.on_tool_start({"name": "lookup_ticket"}, None, run_id="lc-1", inputs={"ticket_id": 8842})
    handler.on_tool_end({"ticket_id": 8842}, run_id="lc-1")
    handler.on_tool_start({"name": "lookup_customer"}, None, run_id="lc-2", inputs={"id": 501})
    handler.on_tool_end({"name": "example-corp"}, run_id="lc-2")
    return handler


def langgraph_run() -> TraceCallbackHandler:
    """LangGraph 复用同一回调协议，会多传 parent_run_id / tags / metadata。"""
    handler = TraceCallbackHandler()
    handler.on_tool_start(
        {"name": "lookup_ticket"},
        None,
        run_id="lg-1",
        parent_run_id="node-tools",
        tags=["graph:step:2"],
        metadata={"langgraph_node": "tools"},
        inputs={"ticket_id": 8842},
    )
    handler.on_tool_end({"ticket_id": 8842}, run_id="lg-1", parent_run_id="node-tools")
    handler.on_tool_start(
        {"name": "lookup_customer"},
        None,
        run_id="lg-2",
        parent_run_id="node-tools",
        tags=["graph:step:2"],
        inputs={"id": 501},
    )
    handler.on_tool_end({"name": "example-corp"}, run_id="lg-2", parent_run_id="node-tools")
    return handler


def adk_run() -> AdkTraceRecorder:
    """ADK 走 before/after_tool_callback；这里用 SimpleNamespace 模拟 BaseTool。"""
    recorder = AdkTraceRecorder()
    ticket = SimpleNamespace(name="lookup_ticket")
    customer = SimpleNamespace(name="lookup_customer")

    recorder.before_tool_callback(ticket, {"ticket_id": 8842})
    recorder.after_tool_callback(ticket, result={"ticket_id": 8842})
    recorder.before_tool_callback(customer, {"id": 501})
    # 故意不调 after：模拟工具被中断 → 应该被判成假成功
    return recorder


def main() -> None:
    rows = [
        ("LangChain", langchain_run().trajectory()),
        ("LangGraph", langgraph_run().trajectory()),
        ("ADK（第二个工具中断）", adk_run().trajectory()),
    ]
    print(f"{'框架':<22}{'结果':<14}原因")
    print("-" * 72)
    for label, trajectory in rows:
        result = score(label, EXPECTED, trajectory, match="in_order")
        reason = "；".join(result.reasons) or "-"
        print(f"{label:<22}{result.status.value:<14}{reason}")


if __name__ == "__main__":
    main()
