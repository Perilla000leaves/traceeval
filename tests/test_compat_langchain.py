"""可选兼容测试：真的装了这个框架时，验证鸭子类型能对上。

平时 `pytest.importorskip` 直接跳过——**使用者不需要装框架**；
CI 里单独一个 job 装上 langchain-core 跑这组，用来防止"框架改了协议而我们的鸭子类型悄悄失效"。

为什么不做成硬依赖：装上 traceeval 就顺带升级 LangChain，这种体验没人想要。
"""
from __future__ import annotations

import inspect
import uuid

import pytest

pytest.importorskip("langchain_core", reason="没装 langchain-core，跳过兼容测试")

from langchain_core.callbacks import BaseCallbackHandler, CallbackManager  # noqa: E402

from traceeval.integrations import TraceCallbackHandler  # noqa: E402


def test_handler_accepts_the_kwargs_langchain_actually_passes():
    """用官方基类的签名做对照：它传什么关键字，我们的 handler 必须全都吃下。"""
    ours = {name for name in inspect.signature(TraceCallbackHandler.on_tool_start).parameters}
    theirs = {name for name in inspect.signature(BaseCallbackHandler.on_tool_start).parameters}
    # 我们至少要覆盖官方传的位置/关键字参数（额外接受 **kwargs 兜底）
    missing = {"serialized", "input_str", "run_id"} - ours
    assert not missing, f"handler 缺少官方回调参数：{missing}"
    assert "run_id" in theirs


def test_handler_records_events_fired_by_real_callback_manager():
    """用 LangChain 的 CallbackManager 真实触发回调，而不是手写 dict 模拟。"""
    handler = TraceCallbackHandler()
    manager = CallbackManager(handlers=[handler])

    run_id = uuid.uuid4()
    run_manager = manager.on_tool_start(
        {"name": "lookup_ticket", "id": ["lookup_ticket"]}, "input", run_id=run_id
    )
    run_manager.on_tool_end("ok")

    trajectory = handler.trajectory()
    assert trajectory.names() == ["lookup_ticket"]
    assert trajectory.calls[0].result_ok is True


def test_handler_marks_error_fired_by_manager():
    handler = TraceCallbackHandler()
    manager = CallbackManager(handlers=[handler])

    run_manager = manager.on_tool_start({"name": "rebuild"}, "input")
    run_manager.on_tool_error(RuntimeError("boom"))

    assert handler.trajectory().calls[0].result_ok is False


def test_tool_run_manager_path(tmp_path):
    """真实流程：on_tool_start 返回 run manager，结束时调它的 on_tool_end。"""
    handler = TraceCallbackHandler()
    manager = CallbackManager(handlers=[handler])

    run_manager = manager.on_tool_start({"name": "search"}, "input")
    run_manager.on_tool_end("ok")

    assert handler.trajectory().names() == ["search"]
    assert handler.trajectory().calls[0].result_ok is True


def test_handler_survives_a_raising_sibling_callback():
    """兄弟回调抛错不应该让我们漏记：LangChain 会遍历所有 handler。"""
    class Boom(BaseCallbackHandler):
        def on_tool_start(self, *args, **kwargs):
            raise RuntimeError("sibling boom")

    handler = TraceCallbackHandler()
    manager = CallbackManager(handlers=[Boom(), handler])

    run_manager = manager.on_tool_start({"name": "search"}, "input")
    run_manager.on_tool_end("ok")

    assert handler.trajectory().names() == ["search"]


def test_handler_declares_required_protocol_attributes():
    """这三个属性是 LangChain handle_event 直接读的，缺了会静默丢事件。"""
    handler = TraceCallbackHandler()
    for name in ("raise_error", "ignore_agent", "run_inline"):
        assert hasattr(handler, name), f"缺少协议属性 {name}"
    assert handler.raise_error is False
