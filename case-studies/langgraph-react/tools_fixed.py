"""修复后的工具层：结构化返回 + 参数校验 + 明确的错误与提示。

和 `tools_baseline.py` 的差异只有三点，但正好对住基线暴露的三类失败：

1. **永远返回结构化结果**（`{"ok": true, ...}` / `{"ok": false, "error": ...}`），
   模型不会再把"空"误当"成功"；
2. **参数严格校验**，错在哪直接说，模型不用猜格式；
3. **查不到时给出可用值提示**，模型可以自己纠正一次就成功，而不是反复瞎试。
"""
from __future__ import annotations

from typing import Any

from data import (
    CUSTOMERS,
    NOTIFICATIONS,
    REPORTS,
    TICKETS,
    is_valid_date,
    tickets_of_customer,
    tickets_on,
)
from langchain_core.tools import tool


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _error(code: str, message: str, hint: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "error": code, "message": message}
    if hint:
        result["hint"] = hint
    return result


@tool
def lookup_ticket(ticket_id: int) -> dict:
    """按工单号查询工单详情（状态、客户、创建时间）。工单不存在时返回 ok=false。"""
    ticket = TICKETS.get(int(ticket_id))
    if ticket is None:
        return _error(
            "ticket_not_found",
            f"没有工单 {ticket_id}",
            hint="这是确定性错误，重试同样的调用不会成功；请告知用户该工单号不存在。",
        )
    return _ok(ticket=ticket)


@tool
def lookup_customer(customer_id: int) -> dict:
    """按客户号查询客户名称与套餐。客户不存在时返回 ok=false。"""
    customer = CUSTOMERS.get(int(customer_id))
    if customer is None:
        return _error(
            "customer_not_found",
            f"没有客户 {customer_id}",
            hint="这是确定性错误，重试同样的调用不会成功；请告知用户该客户号不存在。",
        )
    return _ok(customer=customer)


@tool
def list_tickets(date: str) -> dict:
    """按创建日期（YYYY-MM-DD）查询工单。日期格式错误时返回 ok=false。"""
    if not is_valid_date(str(date)):
        return _error(
            "invalid_date",
            f"日期格式不正确：{date}",
            hint="这是确定性错误；请让用户确认日期（格式 YYYY-MM-DD），不要自己猜一个日期去试。",
        )
    found = tickets_on(str(date))
    return _ok(date=date, count=len(found), tickets=found)


@tool
def list_tickets_by_customer(customer_id: int) -> dict:
    """查询某个客户名下的全部工单。客户不存在时返回 ok=false。"""
    if int(customer_id) not in CUSTOMERS:
        return _error(
            "customer_not_found",
            f"没有客户 {customer_id}",
            hint="这是确定性错误，重试同样的调用不会成功；请告知用户该客户号不存在。",
        )
    found = tickets_of_customer(int(customer_id))
    return _ok(customer_id=customer_id, count=len(found), tickets=found)


@tool
def close_ticket(ticket_id: int) -> dict:
    """关闭工单。工单不存在时返回 ok=false，不会静默成功。"""
    ticket = TICKETS.get(int(ticket_id))
    if ticket is None:
        return _error("ticket_not_found", f"没有工单 {ticket_id}", hint="这是确定性错误；请告知用户该工单号不存在。")
    ticket["status"] = "closed"
    return _ok(ticket_id=ticket_id, status="closed")


@tool
def notify_customer(ticket_id: int, message: str) -> dict:
    """给客户发送关于某个工单的通知。工单不存在或内容为空时返回 ok=false。"""
    if int(ticket_id) not in TICKETS:
        return _error("ticket_not_found", f"没有工单 {ticket_id}")
    if not str(message).strip():
        return _error("empty_message", "通知内容为空", hint="请提供要发送的正文")
    NOTIFICATIONS.append({"ticket_id": int(ticket_id), "message": str(message)})
    return _ok(ticket_id=ticket_id, sent=True)


@tool
def export_report(month: str) -> dict:
    """导出月度报表（YYYY-MM）。月份无数据时返回 ok=false。"""
    report = REPORTS.get(str(month))
    if report is None:
        return _error(
            "report_not_found",
            f"{month} 没有报表",
            hint="这是确定性错误，重试同样的调用不会成功；请告知用户该月没有报表。",
        )
    return _ok(report=report)


TOOLS = [
    lookup_ticket,
    lookup_customer,
    list_tickets,
    list_tickets_by_customer,
    close_ticket,
    notify_customer,
    export_report,
]
