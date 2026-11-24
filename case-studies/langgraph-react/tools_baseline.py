"""基线工具层：很常见的一种写法——返回字符串、异常吞掉返回空串、参数不校验。

这些写法在手工测试时都没问题，问题只在**模型拿到空串之后的行为**：
它分不清"查到但为空"和"失败了"，于是一遍遍重试、或者换个参数瞎试。
这就是 traceeval 要抓的东西。
"""
from __future__ import annotations

import json

from data import CUSTOMERS, REPORTS, TICKETS, is_valid_date, tickets_of_customer, tickets_on
from langchain_core.tools import tool


@tool
def lookup_ticket(ticket_id: int) -> str:
    """查询工单信息。"""
    try:
        return json.dumps(TICKETS[ticket_id], ensure_ascii=False)
    except Exception:
        return ""


@tool
def lookup_customer(customer_id: int) -> str:
    """查询客户信息。"""
    try:
        return json.dumps(CUSTOMERS[customer_id], ensure_ascii=False)
    except Exception:
        return ""


@tool
def list_tickets(date: str) -> str:
    """按日期查询工单。"""
    try:
        return json.dumps(tickets_on(date), ensure_ascii=False)
    except Exception:
        return ""


@tool
def list_tickets_by_customer(customer_id: int) -> str:
    """查询某个客户名下的工单。"""
    try:
        return json.dumps(tickets_of_customer(customer_id), ensure_ascii=False)
    except Exception:
        return ""


@tool
def close_ticket(ticket_id: int) -> str:
    """关闭工单。"""
    try:
        TICKETS[ticket_id]["status"] = "closed"
        return "ok"
    except Exception:
        return ""


@tool
def notify_customer(ticket_id: int, message: str) -> str:
    """给客户发通知。"""
    try:
        return "sent"
    except Exception:
        return ""


@tool
def export_report(month: str) -> str:
    """导出月度报表。"""
    try:
        return json.dumps(REPORTS[month], ensure_ascii=False)
    except Exception:
        return ""


TOOLS = [
    lookup_ticket,
    lookup_customer,
    list_tickets,
    list_tickets_by_customer,
    close_ticket,
    notify_customer,
    export_report,
]

#: 参数格式说明（基线版：没有校验，只写一句宽松描述）
NOTES = {"date": "日期", "month": "月份，如 2026-10"}
_ = is_valid_date  # 基线版没有用到严格校验
