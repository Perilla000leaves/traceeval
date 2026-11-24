"""真实 LangChain 工具跑一次，用 handler 录轨迹并评测。

需要 `pip install traceeval[langchain]`（即 langchain-core）。

运行：`python examples/langchain_tool.py`
"""
from __future__ import annotations

from traceeval import TraceCallbackHandler, report, score

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    raise SystemExit("需要 langchain-core：pip install traceeval[langchain]") from None


@tool
def lookup_ticket(ticket_id: int) -> dict:
    """按工单号查询工单基本信息。"""
    return {"ticket_id": ticket_id, "customer_id": 501}


@tool
def lookup_customer(customer_id: int) -> dict:
    """按客户号查询客户名称。"""
    return {"customer_id": customer_id, "name": "example-corp"}


def main() -> None:
    handler = TraceCallbackHandler()

    # 真实调用：LangChain 自己触发 on_tool_start / on_tool_end
    config = {"callbacks": [handler]}
    lookup_ticket.invoke({"ticket_id": 8842}, config=config)
    lookup_customer.invoke({"customer_id": 501}, config=config)

    trajectory = handler.trajectory()
    result = score("real-langchain", ["lookup_ticket", "lookup_customer"], trajectory, match="in_order")

    print("录到的轨迹：", trajectory.names())
    print("参数：", [call.args for call in trajectory.calls])
    print("判定：", result.status.value, result.reasons or "-")
    print()
    print(report.to_markdown([result], title="真实 LangChain 工具的轨迹评测"))


if __name__ == "__main__":
    main()
