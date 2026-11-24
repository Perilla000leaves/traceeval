"""三个框架的 agent 构造与单次运行。

同一模型（DeepSeek）、同一批工具函数、同一段系统提示，只有框架不同：

- `lg`  ：LangGraph 的 `create_react_agent`
- `lc`  ：LangChain 1.x 的 `agents.create_agent`
- `adk` ：Google ADK 的 `LlmAgent` + 自写 `DeepSeekLlm`

轨迹采集也各用各的适配器：LC/LG 走 LangChain 回调，ADK 走 before/after_tool_callback。
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import adk_model
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tools_common import FUNCS, SYSTEM_PROMPT

from traceeval import AdkTraceRecorder, TraceCallbackHandler
from traceeval.trajectory import Trajectory


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=adk_model._api_key(),
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def _langchain_tools() -> list[Any]:
    return [tool(func) for func in FUNCS]


# ---------------------------------------------------------------- 三个框架


def _build_lg() -> Any:
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(_model(), _langchain_tools(), prompt=SYSTEM_PROMPT)


def _build_lc() -> Any:
    from langchain.agents import create_agent

    return create_agent(_model(), _langchain_tools(), system_prompt=SYSTEM_PROMPT)


def _build_adk() -> Any:
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool

    return LlmAgent(
        name="support",
        model=adk_model.DeepSeekLlm(),
        instruction=SYSTEM_PROMPT,
        tools=[FunctionTool(func) for func in FUNCS],
    )


BUILDERS = {"lg": _build_lg, "lc": _build_lc, "adk": _build_adk}


# ---------------------------------------------------------------- 单次运行


#: 复用同一个 event loop：ADK 的 runner 与 httpx 客户端会把资源绑在 loop 上，
#: 每条任务新建 loop 会报 "Event loop is closed"
_LOOP = asyncio.new_event_loop()


def run_task(agent: Any, framework: str, question: str) -> tuple[Trajectory, str, str]:
    """跑一次，返回 (轨迹, 最终答复, 错误)。"""
    if framework == "adk":
        return _LOOP.run_until_complete(_run_adk(agent, question))
    return _run_langchain(agent, question)


def _run_langchain(agent: Any, question: str) -> tuple[Trajectory, str, str]:
    handler = TraceCallbackHandler()
    started = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [("user", question)]},
            config={"callbacks": [handler], "recursion_limit": 16},
        )
        elapsed = (time.perf_counter() - started) * 1000
        trajectory = handler.trajectory()
        trajectory.latency_ms = round(elapsed, 1)
        return trajectory, str(result["messages"][-1].content), ""
    except Exception as error:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        trajectory = handler.trajectory()
        trajectory.latency_ms = round(elapsed, 1)
        return trajectory, "", f"{type(error).__name__}: {error}"


async def _run_adk(agent: Any, question: str) -> tuple[Trajectory, str, str]:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    recorder = AdkTraceRecorder()
    agent.before_tool_callback = recorder.before_tool_callback
    agent.after_tool_callback = recorder.after_tool_callback
    agent.on_tool_error_callback = recorder.on_tool_error_callback

    runner = InMemoryRunner(agent=agent, app_name="case")
    await runner.session_service.create_session(app_name="case", user_id="u", session_id="s")

    started = time.perf_counter()
    texts: list[str] = []
    try:
        async for event in runner.run_async(
            user_id="u",
            session_id="s",
            new_message=types.Content(role="user", parts=[types.Part(text=question)]),
        ):
            content = getattr(event, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "text", None) and getattr(event, "author", "") == "support":
                    texts.append(part.text)
        elapsed = (time.perf_counter() - started) * 1000
        trajectory = recorder.trajectory()
        trajectory.latency_ms = round(elapsed, 1)
        return trajectory, (texts[-1] if texts else ""), ""
    except Exception as error:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        trajectory = recorder.trajectory()
        trajectory.latency_ms = round(elapsed, 1)
        return trajectory, "", f"{type(error).__name__}: {error}"


__all__ = ["BUILDERS", "run_task"]
