"""用 LangGraph 官方 prebuilt ReAct 组装 agent，工具层可切换（baseline / fixed）。

模型三种来源，优先级从高到低：

1. `TRACEEVAL_MODEL=fake`：脚本化模型，只用于验证管线（不需要 key）；
2. `DEEPSEEK_API_KEY`（或 `~/.config/traceeval/case.env` 里读到的）：真实模型；
3. 都没有则直接报错退出——**不静默降级成 fake**，否则数字会被误读成真模型结果。
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt import create_react_agent

#: 两版工具层共用同一段系统提示——差异只来自工具本身
SYSTEM_PROMPT = (
    "你是客服支持助手，只能依据工具返回的数据回答，不要编造。\n"
    "如果工具返回里带 ok=false 或 error，说明是确定性失败：不要重复同样的调用，"
    "也不要自己猜一个替代参数去试，把错误原因告诉用户、请他确认。"
)


def load_tools(variant: str) -> list[Any]:
    module = importlib.import_module(f"tools_{variant}")
    return list(module.TOOLS)


def load_api_key() -> str | None:
    """优先读环境变量，其次读 ~/.config/traceeval/case.env（不进仓库）。"""
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    env_file = Path.home() / ".config" / "traceeval" / "case.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "DEEPSEEK_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


def build_model() -> BaseChatModel:
    kind = os.environ.get("TRACEEVAL_MODEL", "deepseek").lower()
    if kind == "fake":
        return ScriptedChatModel()

    key = load_api_key()
    if not key:
        raise SystemExit(
            "没有找到 DEEPSEEK_API_KEY。\n"
            "把它写进 ~/.config/traceeval/case.env（一行 DEEPSEEK_API_KEY=...），"
            "或设 TRACEEVAL_MODEL=fake 只验证管线。"
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("TRACEEVAL_MODEL_NAME", "deepseek-chat"),
        base_url=os.environ.get("TRACEEVAL_BASE_URL", "https://api.deepseek.com"),
        api_key=key,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


#: improved：只针对诊断出的三条稳定失败做提示层修正，工具层与 fixed 相同
PROMPT_IMPROVED = (
    SYSTEM_PROMPT
    + "\n今天是 2026-11-18。日期参数一律用 YYYY-MM-DD 传入；"
    "用户只说「11 月 18 日」这类不带年份的说法时，补上今天的年份再传。"
    "\n只有当用户既没说通知内容、也没有可推断的意图时，才向他索要正文；"
    "像「通知客户已处理完」这种已经把意思说清的，直接把这句话当作正文发出，不要反问。"
    "\n唯一例外：用户明确要求发送「空通知」或正文确实是空的，这时不要调用通知工具。"
    "\n用户已经明确授权（例如「我已经确认过」「直接关掉」）时，不要再反问，按他说的执行。"
)

PROMPTS = {"baseline": SYSTEM_PROMPT, "fixed": SYSTEM_PROMPT, "improved": PROMPT_IMPROVED}
TOOL_VARIANT = {"baseline": "baseline", "fixed": "fixed", "improved": "fixed"}


def build_agent(variant: str, model: BaseChatModel | None = None):
    return create_react_agent(
        model or build_model(),
        load_tools(TOOL_VARIANT[variant]),
        prompt=PROMPTS[variant],
    )


class ScriptedChatModel(BaseChatModel):
    """脚本化模型：按固定顺序吐三条消息，用于在没有 key 时验证管线。

    不是"假装成真实模型"——输出里会标注 `scripted`，报告里也会写明。
    """

    scripted: bool = True

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:  # noqa: ARG002
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        # 数一下已经发过几次工具调用，决定下一条消息
        tool_messages = sum(1 for message in messages if message.type == "tool")
        if tool_messages == 0:
            message: BaseMessage = AIMessage(
                content="",
                tool_calls=[{"name": "lookup_ticket", "args": {"ticket_id": 8842}, "id": "s1"}],
            )
        elif tool_messages == 1:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "lookup_customer", "args": {"customer_id": 501}, "id": "s2"}],
            )
        else:
            message = AIMessage(content="客户是 example-corp。")
        return ChatResult(generations=[ChatGeneration(message=message)])
