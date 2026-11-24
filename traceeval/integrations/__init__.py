"""框架接入层：每个框架一个模块，缺依赖时优雅降级。

分层原则：

- **接入模块可以 import 框架**。它本来就是框架专用代码；不 import 只是"照着文档猜协议"。
  LangChain 的 handler 直接继承 `BaseCallbackHandler`，ADK 的录制器按 `BaseTool` 做类型判断。
- **核心判定不 import 框架**。`trajectory` / `metrics` / `report` 对框架一无所知，
  所以测 OpenAI SDK 的人不需要为了评测装 LangChain。

安装对应 extra 即可获得真依赖：

- ``pip install traceeval[langchain]``
- ``pip install traceeval[adk]``
- ``pip install traceeval[compat]``（两个都装，用于验证协议兼容）
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..trajectory import Call, Status, Trajectory
from .adk import ADK_AVAILABLE, AdkTraceRecorder, trajectory_from_adk_events
from .langchain import LANGCHAIN_AVAILABLE, TraceCallbackHandler

__all__ = [
    "ADK_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
    "AdkTraceRecorder",
    "TraceCallbackHandler",
    "trajectory_from_adk_events",
    "trajectory_from_otel_spans",
]


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


def summarize_handler(handler: Any) -> dict[str, Any]:
    """给调试用的轻量摘要（不依赖 report 模块，避免循环导入）。"""
    trajectory = handler.trajectory()
    return {
        "calls": trajectory.names(),
        "ghosts": [call.name for call in trajectory.ghosts()],
        "status": Status.PASS.value,
    }
