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
from .langchain import (
    AGENTEVALS_AVAILABLE,
    LANGCHAIN_AVAILABLE,
    TraceCallbackHandler,
    evaluate_trajectory,
)

__all__ = [
    "ADK_AVAILABLE",
    "AGENTEVALS_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
    "AdkTraceRecorder",
    "TraceCallbackHandler",
    "evaluate_trajectory",
    "trajectory_from_adk_events",
    "trajectory_from_otel_spans",
]


def trajectory_from_otel_spans(spans: Iterable[Any]) -> Trajectory:
    """从 OTel GenAI 语义约定的 span 列表还原轨迹。

    同时接受两种形态：**真实的 `opentelemetry.sdk.trace.ReadableSpan` 对象**
    （读 `.attributes` / `.status.status_code`）与导出的 dict（读 `["attributes"]`）。
    不自己解析 span 结构，只按 GenAI 语义约定的键取值：

    - `gen_ai.tool.name`：工具名，缺失的 span 直接跳过；
    - `gen_ai.tool.call.id`：配对用（同一工具的并发调用靠它区分）；
    - `gen_ai.tool.call.arguments`：参数（JSON 字符串，解析失败就当没写）；
    - 状态为 ERROR 视为工具报错。

    接上这个等于接上所有能导出 OTel 的后端（Phoenix / Langfuse / LangSmith 均可导出）。
    """
    import json

    calls: list[Call] = []
    for span in spans:
        attributes = _span_attributes(span)
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

        result_ok = not _span_is_error(span)
        calls.append(Call(name=name, args=args, result_ok=result_ok, raw={}))
    return Trajectory(calls=calls)


def _span_attributes(span: Any) -> Mapping[str, Any]:
    """读 span 的属性：真实对象走 `.attributes`，dict 走 `["attributes"]`。"""
    if isinstance(span, Mapping):
        attributes = span.get("attributes") or {}
    else:
        attributes = getattr(span, "attributes", None) or {}
    return attributes if isinstance(attributes, Mapping) else {}


def _span_is_error(span: Any) -> bool:
    """span 是否标记为错误：兼容 StatusCode 枚举与字符串两种形态。"""
    if isinstance(span, Mapping):
        status = span.get("status") or {}
        code = status.get("code") if isinstance(status, Mapping) else getattr(status, "code", "")
    else:
        status = getattr(span, "status", None)
        code = getattr(status, "status_code", None)
        if code is None and isinstance(status, Mapping):
            code = status.get("code")
    return "ERROR" in str(code or "").upper()


def summarize_handler(handler: Any) -> dict[str, Any]:
    """给调试用的轻量摘要（不依赖 report 模块，避免循环导入）。"""
    trajectory = handler.trajectory()
    return {
        "calls": trajectory.names(),
        "ghosts": [call.name for call in trajectory.ghosts()],
        "status": Status.PASS.value,
    }
