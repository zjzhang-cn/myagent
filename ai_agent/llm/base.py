"""
LLM 抽象层
"""

from abc import ABC, abstractmethod
from typing import Any


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
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
        usage: dict | None = None,
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __repr__(self):
        return (
            f"LLMResponse(content={self.content[:80]!r}, "
            f"tool_calls={len(self.tool_calls)}, "
            f"finish={self.finish_reason})"
        )
