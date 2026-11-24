"""框架接入层：把各框架的运行时事件读成 `Trajectory`。

设计原则：**内核不 import 任何框架，接入层用鸭子类型**。

- `TraceCallbackHandler` 实现 LangChain / LangGraph 的回调协议（`on_tool_start` /
  `on_tool_end` / `on_tool_error`），但**不继承也不导入** `BaseCallbackHandler`。
  LangChain 只按方法名回调，鸭子类型完全够用，这样装上框架的人不用为评测多装东西，
  没装框架的人也能跑测试（测试里用普通 dict 模拟 serialized）。
- OTel 走 GenAI 语义约定，接上它就等于接上所有能导出 OTel 的追踪后端。

判定逻辑一行都不在这里——这里只做「把事件收集成 Call 列表」。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..trajectory import Call, Status, Trajectory
from .adk import AdkTraceRecorder, trajectory_from_adk_events

__all__ = [
    "AdkTraceRecorder",
    "TraceCallbackHandler",
    "trajectory_from_adk_events",
    "trajectory_from_otel_spans",
]


class TraceCallbackHandler:
    """LangChain / LangGraph 回调处理器：录下工具调用与返回。

    用法（不依赖 traceeval 装 langchain）：:

        handler = TraceCallbackHandler()
        agent.invoke({...}, config={"callbacks": [handler]})
        result = traceeval.score("case-1", ["lookup_ticket"], handler.trajectory())

    **为什么要按 run_id 配对**：LangChain 的 `on_tool_start` 与 `on_tool_end` 通过
    `run_id` 关联。只记 start 不记 end，就分不清「调了并拿到结果」和「调了但抛错/中断」——
    后者在日志里看起来也是"调用过"。

    **为什么必须有下面三个类属性**：LangChain 的 `handle_event` 直接读
    `handler.ignore_agent` / `handler.raise_error`（没有默认值，也不是 `getattr(handler, x, False)`）。
    鸭子类型缺了它们会抛 `AttributeError`，而外层 `try/except` 只打一条 warning 就跳过——
    表现为「装上 traceeval 后轨迹永远是空的」。这三个属性是协议的一部分，不是可选项。
    """

    #: 协议要求：是否让回调异常向上抛（我们不抛，评测不该影响被测系统）
    raise_error = False
    #: 协议要求：是否忽略 agent 事件（我们只关心工具事件，但不忽略）
    ignore_agent = False
    #: 协议要求：是否在调用线程内同步执行（默认 False 即可）
    run_inline = False

    def __init__(self) -> None:
        self._calls: list[Call] = []
        self._index: dict[str, int] = {}

    # -- LangChain 回调协议（方法名即协议，无需继承） ------------------------- #

    def on_tool_start(
        self,
        serialized: Mapping[str, Any] | None = None,
        input_str: str | None = None,
        *,
        run_id: Any = None,
        inputs: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> None:
        """工具开始执行。`serialized` 里带 name；`inputs` 是结构化参数。"""
        name = ""
        if isinstance(serialized, Mapping):
            name = str(serialized.get("name") or serialized.get("id") or "")
        name = name or "unknown"
        args = dict(inputs) if isinstance(inputs, Mapping) else None
        key = str(run_id) if run_id is not None else f"auto-{len(self._calls)}"
        self._index[key] = len(self._calls)
        self._calls.append(Call(name=name, args=args, result_ok=None))

    def on_tool_end(self, output: Any = None, *, run_id: Any = None, **_: Any) -> None:
        """工具正常结束：标记拿到返回。空输出视为没拿到（很多框架把异常吞成空串）。"""
        self._finish(run_id, ok=_output_ok(output))

    def on_tool_error(self, error: Any = None, *, run_id: Any = None, **_: Any) -> None:
        """工具抛错：标记失败，但调用本身仍记在轨迹里。"""
        self._finish(run_id, ok=False)

    # -- 取结果 --------------------------------------------------------------- #

    def trajectory(self) -> Trajectory:
        """当前录到的轨迹（可在一次运行结束后多次调用）。"""
        return Trajectory(calls=list(self._calls))

    def reset(self) -> None:
        self._calls.clear()
        self._index.clear()

    def _finish(self, run_id: Any, *, ok: bool) -> None:
        key = str(run_id) if run_id is not None else ""
        index = self._index.pop(key, None)
        if index is None:
            return
        original = self._calls[index]
        self._calls[index] = Call(
            name=original.name, args=original.args, result_ok=ok, raw=original.raw
        )


def _output_ok(output: Any) -> bool:
    """工具输出是否算「拿到了结果」。"""
    if output is None:
        return False
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, (list, tuple, dict)):
        return bool(output)
    return True


def trajectory_from_otel_spans(spans: Iterable[Mapping[str, Any]]) -> Trajectory:
    """从 OTel GenAI 语义约定的 span 列表还原轨迹。

    读取的属性键（按 GenAI 语义约定）：

    - `gen_ai.tool.name`：工具名，缺失的 span 直接跳过；
    - `gen_ai.tool.call.id`：配对用（同一工具的并发调用靠它区分）；
    - `gen_ai.tool.call.arguments`：参数（JSON 字符串，解析失败就当没写）；
    - span 的 `status.code == "STATUS_CODE_ERROR"` 视为工具报错。

    接上这个等于接上所有能导出 OTel 的后端（Phoenix / Langfuse / LangSmith 均可导出）。
    """
    import json

    calls: list[Call] = []
    for span in spans:
        attributes = span.get("attributes") or {}
        if not isinstance(attributes, Mapping):
            continue
        name = str(attributes.get("gen_ai.tool.name") or "")
        if not name:
            continue
        raw_args = attributes.get("gen_ai.tool.call.arguments")
        args: dict[str, Any] | None = None
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
                args = parsed if isinstance(parsed, dict) else None
            except ValueError:
                args = None
        elif isinstance(raw_args, Mapping):
            args = dict(raw_args)

        status = span.get("status") or {}
        code = str((status or {}).get("code") or "").upper()
        result_ok = False if "ERROR" in code else True
        calls.append(Call(name=name, args=args, result_ok=result_ok, raw=dict(span)))
    return Trajectory(calls=calls)


def summarize_handler(handler: TraceCallbackHandler) -> dict[str, Any]:
    """给调试用的轻量摘要（不依赖 report 模块，避免循环导入）。"""
    trajectory = handler.trajectory()
    return {
        "calls": trajectory.names(),
        "ghosts": [call.name for call in trajectory.ghosts()],
        "status": Status.PASS.value,
    }
