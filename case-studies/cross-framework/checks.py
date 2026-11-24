"""答复与副作用核验（与 langgraph-react/run.py 同口径，抽出来给跨框架共用）。"""
from __future__ import annotations

from typing import Any


def check_side_effects(task: dict[str, Any], snap: dict[str, Any], known_tickets: set[int]) -> tuple[bool, str]:
    rule = task.get("side_effects") or {}
    missing_closed = [tid for tid in (rule.get("expect_closed") or []) if tid not in snap["closed"]]
    if missing_closed:
        return False, f"应该关闭但没有关闭：{missing_closed}"
    missing_notified = [tid for tid in (rule.get("expect_notified") or []) if tid not in snap["notified"]]
    if missing_notified:
        return False, f"应该通知但没有通知：{missing_notified}"
    if rule.get("forbid_notify_missing_ticket"):
        bad = [tid for tid in snap["notified"] if tid not in known_tickets]
        if bad:
            return False, f"给不存在的工单发了通知：{bad}"
    return True, ""


def check_answer(task: dict[str, Any], text: str) -> tuple[bool, str]:
    rule = task.get("answer_rules") or {}
    required = rule.get("require_any") or []
    if required and not any(phrase in text for phrase in required):
        return False, f"答复没说明真实原因（应含 {'/'.join(required)}）"
    forbidden = [phrase for phrase in (rule.get("forbid_any") or []) if phrase in text]
    if forbidden:
        return False, f"答复给出了错误结论（出现 {'/'.join(forbidden)}）"
    return True, ""


__all__ = ["check_answer", "check_side_effects"]
