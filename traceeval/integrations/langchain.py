"""LangChain / LangGraph 接入：**真继承官方回调基类**。

装上 `langchain-core` 时，`TraceCallbackHandler` 就是 `BaseCallbackHandler` 的子类——
类型检查、协议属性（`ignore_agent` 是基类的 property）、未来新增的回调都能自动对齐。
没装时退化成同名接口，核心判定逻辑照常可用（使用者可能只测 OpenAI SDK）。

为什么这样分层：

- **接入模块可以 import 框架**——它本来就是框架专用代码，不 import 才是猜协议；
- **核心判定不 import 框架**——否则测 OpenAI 的人被迫装 LangChain。

安装：``pip install traceeval[langchain]``
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:  # 装了就真继承，没装就退化（不影响核心功能）
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于环境

    class _BaseCallbackHandler:  # type: ignore[no-redef]
        """缺 langchain 时的占位基类，保持同名接口。"""

        raise_error = False
        ignore_agent = False
        run_inline = False

    LANGCHAIN_AVAILABLE = False

from ..trajectory import Call, Trajectory

__all__ = ["LANGCHAIN_AVAILABLE", "TraceCallbackHandler"]


def _output_ok(output: Any) -> bool:
    """工具输出是否算「拿到了结果」。空串/空容器视为没拿到（框架常把异常吞成空串）。"""
    if output is None:
        return False
    if isinstance(output, str):
        return bool(output.strip())
    if isinstance(output, (list, tuple, dict, set)):
        return bool(output)
    return True


class TraceCallbackHandler(_BaseCallbackHandler):  # type: ignore[misc]
    """录下 LangChain / LangGraph 的工具调用与返回。

    用法::

        handler = TraceCallbackHandler()
        agent.invoke({...}, config={"callbacks": [handler]})
        result = traceeval.score("case-1", ["lookup_ticket"], handler.trajectory())

    按 `run_id` 配对 start / end：只记 start 不记 end，就分不清「调了并拿到结果」和
    「调了但抛错/中断」——后者在日志里看起来也是"调用过"，却是假成功。
    """

    #: 评测不该影响被测系统：回调异常不向上抛
    raise_error = False
    #: 只关心工具事件，但显式声明（基类里是 property，覆盖成常量更直观）
    ignore_agent = False
    #: 在调用线程内同步执行
    run_inline = False

    def __init__(self) -> None:
        super().__init__()
        self._calls: list[Call] = []
        self._index: dict[str, int] = {}

    # -- LangChain 回调协议 --------------------------------------------------- #

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
        """工具正常结束：标记拿到返回。"""
        self._finish(run_id, ok=_output_ok(output))

    def on_tool_error(self, error: Any = None, *, run_id: Any = None, **_: Any) -> None:
        """工具抛错：标记失败，调用本身仍留在轨迹里。"""
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
