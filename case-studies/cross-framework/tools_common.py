"""跨框架评测：同一批任务、同一模型、同一套工具，只换 agent 框架。

工具用**普通函数**定义一次，三个框架各自包装：

- LangGraph / LangChain：`langchain_core.tools.tool` 包成 StructuredTool；
- ADK：`google.adk.tools.FunctionTool` 包成 BaseTool。

这样"工具语义"完全一致，差异只可能来自框架本身。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 数据集与任务集在兄弟目录（langgraph-react），挂进 sys.path
_SIBLING = Path(__file__).resolve().parent.parent / "langgraph-react"
if str(_SIBLING) not in sys.path:
    sys.path.insert(0, str(_SIBLING))

# ---------------------------------------------------------------- 工具本体
# 与 tools_fixed.py 语义一致：结构化返回 ok/error，异常不外泄。


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _error(code: str, message: str, hint: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "error": code, "message": message}
    if hint:
        result["hint"] = hint
    return result


def lookup_ticket(ticket_id: int) -> dict:
    """按工单号查询工单详情（状态、客户、创建时间）。工单不存在时返回 ok=false。"""
    from data import TICKETS

    ticket = TICKETS.get(int(ticket_id))
    if ticket is None:
        return _error("ticket_not_found", f"没有工单 {ticket_id}", "请告知用户该工单号不存在。")
    return _ok(ticket=dict(ticket))


def lookup_customer(customer_id: int) -> dict:
    """按客户号查询客户名称与套餐。客户不存在时返回 ok=false。"""
    from data import CUSTOMERS

    customer = CUSTOMERS.get(int(customer_id))
    if customer is None:
        return _error("customer_not_found", f"没有客户 {customer_id}", "请告知用户该客户号不存在。")
    return _ok(customer=dict(customer))


def list_tickets(date: str) -> dict:
    """按创建日期（YYYY-MM-DD）查询工单。日期格式错误时返回 ok=false。"""
    from data import is_valid_date, tickets_on

    if not is_valid_date(str(date)):
        return _error(
            "invalid_date",
            f"日期格式不正确：{date}",
            "请让用户确认日期（格式 YYYY-MM-DD），不要自己猜一个日期去试。",
        )
    found = [dict(t) for t in tickets_on(str(date))]
    return _ok(date=date, count=len(found), tickets=found)


def list_tickets_by_customer(customer_id: int) -> dict:
    """查询某个客户名下的全部工单。客户不存在时返回 ok=false。"""
    from data import CUSTOMERS, tickets_of_customer

    if int(customer_id) not in CUSTOMERS:
        return _error("customer_not_found", f"没有客户 {customer_id}", "请告知用户该客户号不存在。")
    found = [dict(t) for t in tickets_of_customer(int(customer_id))]
    return _ok(customer_id=customer_id, count=len(found), tickets=found)


def close_ticket(ticket_id: int) -> dict:
    """关闭工单。工单不存在时返回 ok=false，不会静默成功。"""
    from data import TICKETS

    ticket = TICKETS.get(int(ticket_id))
    if ticket is None:
        return _error("ticket_not_found", f"没有工单 {ticket_id}", "请告知用户该工单号不存在。")
    ticket["status"] = "closed"
    return _ok(ticket_id=ticket_id, status="closed")


def notify_customer(ticket_id: int, message: str) -> dict:
    """给客户发送关于某个工单的通知。工单不存在或内容为空时返回 ok=false。"""
    from data import NOTIFICATIONS, TICKETS

    if int(ticket_id) not in TICKETS:
        return _error("ticket_not_found", f"没有工单 {ticket_id}")
    if not str(message).strip():
        return _error("empty_message", "通知内容为空", "请先向用户要正文")
    NOTIFICATIONS.append({"ticket_id": int(ticket_id), "message": str(message)})
    return _ok(ticket_id=ticket_id, sent=True)


def export_report(month: str) -> dict:
    """导出月度报表（YYYY-MM）。月份无数据时返回 ok=false。"""
    from data import REPORTS

    report = REPORTS.get(str(month))
    if report is None:
        return _error("report_not_found", f"{month} 没有报表", "请告知用户该月没有报表。")
    return _ok(report=dict(report))


FUNCS = [
    lookup_ticket,
    lookup_customer,
    list_tickets,
    list_tickets_by_customer,
    close_ticket,
    notify_customer,
    export_report,
]

#: 三个框架共用的系统提示（改进版：日期格式 + 授权即执行 + 空通知例外）
SYSTEM_PROMPT = (
    "你是客服支持助手，只能依据工具返回的数据回答，不要编造。\n"
    "如果工具返回里带 ok=false 或 error，说明是确定性失败：不要重复同样的调用，"
    "也不要自己猜一个替代参数去试，把错误原因告诉用户、请他确认。\n"
    "今天是 2026-11-18。日期参数一律用 YYYY-MM-DD 传入；"
    "用户只说「11 月 18 日」这类不带年份的说法时，补上今天的年份再传。\n"
    "只有当用户既没说通知内容、也没有可推断的意图时，才向他索要正文；"
    "像「通知客户已处理完」这种已经把意思说清的，直接把这句话当作正文发出，不要反问。\n"
    "唯一例外：用户明确要求发送「空通知」或正文确实是空的，这时不要调用通知工具。"
)


def load_tasks(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else _SIBLING / "tasks.jsonl"
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
