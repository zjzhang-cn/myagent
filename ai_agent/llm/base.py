"""
LLM 抽象层
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class StreamEvent:
    """流式响应事件"""
    type: str  # "thinking" | "token" | "done"
    content: str = ""  # token/thinking 文本 或 完整内容（done 时）
    thinking: str = ""  # 完整推理内容（done 时）
    tool_calls: list[dict] = field(default_factory=list)  # done 时解析的工具调用
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


class BaseLLM(ABC):
    """LLM 基类，定义统一的接口"""

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """
        发送聊天请求，返回 {role, content, tool_calls?}

        Args:
            messages: 消息列表 [{"role": "...", "content": "..."}]
            tools: 可选工具定义列表

        Returns:
            {"role": "assistant", "content": "...", "tool_calls": [...]}
        """
        ...

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """
        流式聊天请求，逐 token 返回

        Args:
            messages: 消息列表
            tools: 可选工具定义列表

        Yields:
            StreamEvent: type="token" 时逐字输出，type="done" 时包含完整结果
        """
        # 默认实现：一次性返回（子类可覆盖）
        response = self.chat(messages, tools)
        yield StreamEvent(
            type="done",
            content=response.content,
            tool_calls=response.tool_calls,
            usage=response.usage,
        )

    @abstractmethod
    def list_models(self) -> list[str]:
        """返回可用模型列表"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前使用的模型名"""
        ...


class LLMResponse:
    """标准化的 LLM 响应"""

    def __init__(
        self,
        content: str = "",
        thinking: str = "",
        thinking_signature: str = "",
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
        usage: dict | None = None,
    ):
        self.content = content
        self.thinking = thinking
        self.thinking_signature = thinking_signature
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __repr__(self):
        return (
            f"LLMResponse(content={self.content[:80]!r}, "
            f"thinking={self.thinking[:40]!r}, "
            f"tool_calls={len(self.tool_calls)}, "
            f"finish={self.finish_reason})"
        )
