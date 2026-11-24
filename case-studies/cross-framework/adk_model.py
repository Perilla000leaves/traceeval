"""ADK 的 DeepSeek 适配层：一个最小的 `BaseLlm` 实现。

ADK 官方只内置 Gemini，以及一个依赖 litellm 的 `LiteLlm`（依赖链很重）。
DeepSeek 是 OpenAI 兼容接口，所以直接写一个 `BaseLlm` 子类就够了——**约 80 行，零额外依赖**：

- 把 ADK 的 `LlmRequest`（contents / system_instruction / function_declarations）
  翻译成 OpenAI 的 messages + tools；
- 把 OpenAI 的返回（文本或 tool_calls）翻译回 ADK 的 `LlmResponse`。

这也是"框架之间差异有多大"的一部分：ADK 的模型接口是显式的 `BaseLlm`，
LangChain 是 `BaseChatModel`，两者都要写适配——只是 ADK 这边 ADK 负责 orchestration。
"""
from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from langchain_openai import ChatOpenAI


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    env_file = Path.home() / ".config" / "traceeval" / "case.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("缺少 DEEPSEEK_API_KEY")


def _schema_to_json(declaration: Any) -> dict[str, Any]:
    """取工具的 JSON Schema。

    ADK 2.15 把参数 schema 放在 `parameters_json_schema`，但**属性访问不一定拿得到**
    （lazy 字段），必须走 `model_dump()` 才稳定。取错会得到空 properties 的 schema，
    模型因此不知道要传什么参数——表现为"第一次调用不带参数、失败后再补"。
    """
    try:
        data = declaration.model_dump(exclude_none=True)
    except AttributeError:
        data = {}
    raw = data.get("parameters_json_schema")
    if isinstance(raw, dict) and raw:
        return dict(raw)
    legacy = data.get("parameters")
    if isinstance(legacy, dict) and legacy:
        legacy.setdefault("type", "object")
        legacy.setdefault("properties", {})
        return legacy
    fallback = getattr(declaration, "parameters_json_schema", None)
    if isinstance(fallback, dict) and fallback:
        return dict(fallback)
    return {"type": "object", "properties": {}}


def _tools_to_openai(llm_request: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    config = getattr(llm_request, "config", None)
    for tool in getattr(config, "tools", None) or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": _schema_to_json(declaration),
                    },
                }
            )
    return tools


def _contents_to_messages(llm_request: Any) -> list[dict[str, Any]]:
    """ADK 的 contents → OpenAI messages（含 tool_calls 与 tool 结果）。"""
    messages: list[dict[str, Any]] = []
    pending_names: list[str] = []

    for content in getattr(llm_request, "contents", None) or []:
        role = getattr(content, "role", "user") or "user"
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for part in getattr(content, "parts", None) or []:
            if getattr(part, "function_call", None) is not None:
                call = part.function_call
                call_id = getattr(call, "id", None) or f"call_{len(pending_names)}"
                pending_names.append(call.name)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(dict(call.args or {}), ensure_ascii=False),
                        },
                    }
                )
            elif getattr(part, "function_response", None) is not None:
                response = part.function_response
                call_id = getattr(response, "id", None)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id or f"call_{len(tool_results)}",
                        "content": json.dumps(response.response, ensure_ascii=False, default=str),
                    }
                )
            elif getattr(part, "text", None):
                text_parts.append(part.text)

        if role == "model":
            message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)
        else:
            messages.extend(tool_results)
            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})

    return messages


class DeepSeekLlm(BaseLlm):
    """把 OpenAI 兼容接口包成 ADK 的模型。"""

    def __init__(self, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com", **kwargs: Any):
        super().__init__(model=model, **kwargs)
        self._client = ChatOpenAI(
            model=model, base_url=base_url, api_key=_api_key(), temperature=0, timeout=60, max_retries=2
        )

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False  # noqa: ARG002
    ) -> AsyncGenerator[LlmResponse, None]:
        messages = _contents_to_messages(llm_request)
        system = getattr(getattr(llm_request, "config", None), "system_instruction", None)
        if system:
            messages = [{"role": "system", "content": system}] + messages

        tools = _tools_to_openai(llm_request)
        client = self._client.bind_tools(tools) if tools else self._client
        reply = await client.ainvoke(messages)

        parts: list[types.Part] = []
        content_text = reply.content if isinstance(reply.content, str) else ""
        if content_text:
            parts.append(types.Part(text=content_text))
        for call in getattr(reply, "tool_calls", None) or []:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call.get("id"),
                        name=call["name"],
                        args=call.get("args") or {},
                    )
                )
            )
        if not parts:
            parts.append(types.Part(text=""))

        yield LlmResponse(content=types.Content(role="model", parts=parts), turn_complete=True)
