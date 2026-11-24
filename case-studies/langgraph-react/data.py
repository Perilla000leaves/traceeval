"""case study 用的虚构数据集：客服工单。

刻意做成小、确定、可复现——case study 要比较的是**工具契约层**的差异，
不是数据规模。
"""
from __future__ import annotations

from typing import Any

TICKETS: dict[int, dict[str, Any]] = {
    8842: {
        "ticket_id": 8842,
        "customer_id": 501,
        "status": "open",
        "subject": "cannot log in",
        "created": "2026-11-17",
    },
    8843: {
        "ticket_id": 8843,
        "customer_id": 502,
        "status": "pending",
        "subject": "refund request",
        "created": "2026-11-18",
    },
    8844: {
        "ticket_id": 8844,
        "customer_id": 501,
        "status": "closed",
        "subject": "invoice copy",
        "created": "2026-11-18",
    },
}

CUSTOMERS: dict[int, dict[str, Any]] = {
    501: {"customer_id": 501, "name": "example-corp", "plan": "pro"},
    502: {"customer_id": 502, "name": "acme-ltd", "plan": "basic"},
}

REPORTS: dict[str, dict[str, Any]] = {
    "2026-10": {"month": "2026-10", "rows": 1200, "url": "https://oss.example.com/report-2026-10.csv"},
    "2026-11": {"month": "2026-11", "rows": 340, "url": "https://oss.example.com/report-2026-11.csv"},
}


def tickets_on(date: str) -> list[dict[str, Any]]:
    return [ticket for ticket in TICKETS.values() if ticket["created"] == date]


def tickets_of_customer(customer_id: int) -> list[dict[str, Any]]:
    return [ticket for ticket in TICKETS.values() if ticket["customer_id"] == customer_id]


def is_valid_date(value: str) -> bool:
    """严格校验 `YYYY-MM-DD`；不做宽松解析——宽松解析正是要测的坑之一。"""
    from datetime import date

    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True
