"""适配器：把不同来源的轨迹归一成 `Trajectory`。

三种主流形态都能读：

- **OpenAI**：`assistant.tool_calls` + `role="tool"` 的返回消息；
- **Anthropic**：`tool_use` / `tool_result` 内容块；
- **通用事件流**：`{"type": "tool_call" | "tool_result"}`，MCP、自建 runtime、
  OpenTelemetry span 都能映射成这个形态。

**为什么必须按 id 配对**：模型「发出 tool_call」和「拿到 tool_result」是两件事。
只看 assistant 消息有没有 tool_call，会把**假成功**（发了但没返回）判成通过；
而这类失败在日志里看起来完全正常，是 agent 评测里最容易漏掉的一类。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .trajectory import Call, Trajectory


def _content_ok(content: Any) -> bool:
    """返回内容是否算「拿到了结果」。空串与 None 视为没拿到。"""
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, (list, tuple)):
        return bool(content)
    return True


def from_openai(messages: Iterable[Mapping[str, Any]]) -> Trajectory:
    """OpenAI 形态：assistant.tool_calls → Call，role="tool" 消息补返回。"""
    calls: list[Call] = []
    pending: dict[str, int] = {}  # tool_call_id -> calls 下标
    pending_calls: dict[str, Call] = {}

    for message in messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            for item in message.get("tool_calls") or ():
                function = item.get("function") or {}
                name = str(function.get("name") or "")
                if not name:
                    continue
                call = Call(name=name, args=None, result_ok=None, raw=dict(item))
                calls.append(call)
                call_id = str(item.get("id") or "")
                if call_id:
                    pending[call_id] = len(calls) - 1
                    pending_calls[call_id] = call
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            index = pending.pop(call_id, None)
            if index is not None:
                original = calls[index]
                calls[index] = Call(
                    name=original.name,
                    args=original.args,
                    result_ok=_content_ok(message.get("content")),
                    raw=original.raw,
                )
    return Trajectory(calls=calls)


def from_anthropic(messages: Iterable[Mapping[str, Any]]) -> Trajectory:
    """Anthropic 形态：`tool_use` 与 `tool_result` 成对出现。"""
    calls: list[Call] = []
    pending: dict[str, int] = {}

    for message in messages:
        blocks = message.get("content")
        if not isinstance(blocks, (list, tuple)):
            continue
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            kind = str(block.get("type") or "")
            if kind == "tool_use":
                name = str(block.get("name") or "")
                if not name:
                    continue
                args = block.get("input")
                calls.append(
                    Call(
                        name=name,
                        args=args if isinstance(args, dict) else None,
                        result_ok=None,
                        raw=dict(block),
                    )
                )
                use_id = str(block.get("id") or "")
                if use_id:
                    pending[use_id] = len(calls) - 1
            elif kind == "tool_result":
                index = pending.pop(str(block.get("tool_use_id") or ""), None)
                if index is None:
                    continue
                original = calls[index]
                is_error = bool(block.get("is_error"))
                ok = (not is_error) and _content_ok(block.get("content"))
                calls[index] = Call(
                    name=original.name, args=original.args, result_ok=ok, raw=original.raw
                )
    return Trajectory(calls=calls)


def from_events(events: Iterable[Mapping[str, Any]]) -> Trajectory:
    """通用事件流：`tool_call` / `tool_result` 按 id 配对。"""
    calls: list[Call] = []
    pending: dict[str, int] = {}

    for event in events:
        kind = str(event.get("type") or event.get("event") or "")
        if kind in ("tool_call", "tool_use"):
            name = str(event.get("name") or "")
            if not name:
                continue
            args = event.get("args", event.get("arguments", event.get("input")))
            calls.append(
                Call(
                    name=name,
                    args=args if isinstance(args, dict) else None,
                    result_ok=None,
                    raw=dict(event),
                )
            )
            call_id = str(event.get("id") or event.get("call_id") or "")
            if call_id:
                pending[call_id] = len(calls) - 1
        elif kind in ("tool_result", "tool_response"):
            index = pending.pop(str(event.get("id") or event.get("call_id") or ""), None)
            if index is None:
                continue
            original = calls[index]
            ok = bool(event.get("ok", not event.get("error"))) and _content_ok(
                event.get("content", event.get("result", True))
            )
            calls[index] = Call(name=original.name, args=original.args, result_ok=ok, raw=original.raw)
    return Trajectory(calls=calls)


def from_calls(calls: Sequence[Mapping[str, Any]]) -> Trajectory:
    """已经是 `{name, args, result_ok}` 的形态，直接读（用于回放与单测）。"""
    return Trajectory(
        calls=[
            Call(
                name=str(item.get("name") or ""),
                args=item.get("args") if isinstance(item.get("args"), dict) else None,
                result_ok=item.get("result_ok"),
                raw=dict(item),
            )
            for item in calls
            if item.get("name")
        ]
    )


#: 支持的输入格式 → 解析函数
ADAPTERS = {
    "openai": from_openai,
    "anthropic": from_anthropic,
    "events": from_events,
    "calls": from_calls,
}


def parse(payload: Any, fmt: str) -> Trajectory:
    """按 `fmt` 解析；未知格式抛 `ValueError`（宁可显式失败，不要静默返回空轨迹）。"""
    adapter = ADAPTERS.get(fmt)
    if adapter is None:
        raise ValueError(f"未知轨迹格式 {fmt!r}，可选：{', '.join(sorted(ADAPTERS))}")
    if not isinstance(payload, (list, tuple)):
        raise ValueError(f"{fmt} 格式需要一个消息或事件列表")
    return adapter(payload)


__all__ = [
    "ADAPTERS",
    "from_anthropic",
    "from_calls",
    "from_events",
    "from_openai",
    "parse",
]
