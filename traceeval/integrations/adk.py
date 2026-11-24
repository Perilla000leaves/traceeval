"""ADK 接入：用回调录制轨迹，或从事件流离线还原。

ADK 的机制和 LangChain 完全不同，所以单独一个适配器：

- **运行期**：把 `AdkTraceRecorder` 挂到 agent 的 `before_tool_callback` /
  `after_tool_callback` / `on_tool_error_callback` 上。三个回调都必须返回 `None`——
  ADK 里返回 dict 表示"短路并用它作为工具结果"，返回非 None 会改变控制流。
- **离线**：`trajectory_from_adk_events` 直接读 ADK 的 Event 列表
  （`content.parts[*].function_call` / `function_response`），不需要跑 agent。

装上 `google-adk` 时用真实的 `BaseTool` 做类型判断；没装时退化成属性读取，
所以核心测试不依赖 ADK。

安装：``pip install traceeval[adk]``
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from typing import Any

from ..trajectory import Call, Trajectory

try:  # 装了就按官方类型判断，没装就退化
    from google.adk.tools import BaseTool as _AdkBaseTool

    ADK_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于环境
    _AdkBaseTool = None  # type: ignore[assignment]
    ADK_AVAILABLE = False

__all__ = ["ADK_AVAILABLE", "AdkTraceRecorder", "trajectory_from_adk_events"]


def _tool_name(tool: Any) -> str:
    """ADK 传的是 BaseTool 实例；名字在 `.name`，取不到就退化用字符串。"""
    if _AdkBaseTool is not None and isinstance(tool, _AdkBaseTool):
        return str(tool.name)
    name = getattr(tool, "name", None)
    return str(name) if name else str(tool)


def _args_of(args: Any) -> dict[str, Any] | None:
    if isinstance(args, Mapping):
        return dict(args)
    if isinstance(args, str) and args.strip():
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None
    return None


def _ok(result: Any) -> bool:
    """工具返回是否算「拿到了结果」。ADK 的失败常常是返回一个空 dict。"""
    if result is None:
        return False
    if isinstance(result, (str, list, tuple, dict, set)):
        return bool(result)
    return True


class AdkTraceRecorder:
    """ADK 回调录制器。

    用法::

        recorder = AdkTraceRecorder()
        agent = LlmAgent(
            ...,
            before_tool_callback=recorder.before_tool_callback,
            after_tool_callback=recorder.after_tool_callback,
            on_tool_error_callback=recorder.on_tool_error_callback,
        )
        await runner.run_async(...)
        result = traceeval.score("case-1", ["lookup_ticket"], recorder.trajectory())

    **为什么按工具名排队而不是用 run_id**：ADK 的回调签名里没有每次调用的唯一 id，
    同一工具并发时只能靠 FIFO 关联"先开始先结束"。ADK 的单个 agent 内工具是顺序执行的，
    这个假设成立；如果你的运行时是并发的，改用 `trajectory_from_adk_events`（事件里有 id）。
    """

    def __init__(self) -> None:
        self._calls: list[Call] = []
        self._pending: dict[str, deque[int]] = defaultdict(deque)

    def before_tool_callback(self, tool: Any, args: Any, tool_context: Any = None) -> None:
        name = _tool_name(tool)
        self._pending[name].append(len(self._calls))
        self._calls.append(Call(name=name, args=_args_of(args), result_ok=None))
        return None

    def after_tool_callback(
        self,
        tool: Any,
        args: Any = None,
        tool_context: Any = None,
        result: Any = None,
        **_: Any,
    ) -> None:
        self._finish(_tool_name(tool), ok=_ok(result))
        return None

    def on_tool_error_callback(
        self,
        tool: Any,
        args: Any = None,
        tool_context: Any = None,
        error: Any = None,
        **_: Any,
    ) -> None:
        """工具抛错：标记失败，调用本身仍留在轨迹里（它确实发生过）。"""
        self._finish(_tool_name(tool), ok=False)
        return None

    def trajectory(self) -> Trajectory:
        return Trajectory(calls=list(self._calls))

    def reset(self) -> None:
        self._calls.clear()
        self._pending.clear()

    def _finish(self, name: str, *, ok: bool) -> None:
        queue = self._pending.get(name)
        if not queue:
            return
        index = queue.popleft()
        original = self._calls[index]
        self._calls[index] = Call(
            name=original.name, args=original.args, result_ok=ok, raw=original.raw
        )


def trajectory_from_adk_events(events: Iterable[Mapping[str, Any]]) -> Trajectory:
    """从 ADK 的事件流离线还原轨迹。

    识别 `content.parts` 里的 `function_call`（模型发起调用）与 `function_response`
    （工具返回）。两者都带 `id` 时按 id 配对；老版本没有 id 时按工具名 FIFO 配对。
    """
    calls: list[Call] = []
    by_id: dict[str, int] = {}
    pending_by_name: dict[str, deque[int]] = defaultdict(deque)

    for event in events:
        content = event.get("content") or {}
        if not isinstance(content, Mapping):
            continue
        parts = content.get("parts") or ()
        if not isinstance(parts, (list, tuple)):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            call_info = part.get("function_call")
            if isinstance(call_info, Mapping):
                name = str(call_info.get("name") or "")
                if not name:
                    continue
                calls.append(
                    Call(
                        name=name,
                        args=_args_of(call_info.get("args")),
                        result_ok=None,
                        raw=dict(call_info),
                    )
                )
                index = len(calls) - 1
                call_id = str(call_info.get("id") or "")
                if call_id:
                    by_id[call_id] = index
                pending_by_name[name].append(index)
                continue

            response = part.get("function_response")
            if isinstance(response, Mapping):
                name = str(response.get("name") or "")
                call_id = str(response.get("id") or "")
                index = by_id.pop(call_id, None) if call_id else None
                if index is None and name and pending_by_name[name]:
                    index = pending_by_name[name].popleft()
                if index is None:
                    continue
                original = calls[index]
                calls[index] = Call(
                    name=original.name,
                    args=original.args,
                    result_ok=_ok(response.get("response")),
                    raw=original.raw,
                )
    return Trajectory(calls=calls)
