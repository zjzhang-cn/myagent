"""
LLM 抽象层 — 统一的大语言模型接口

定义了所有 LLM 后端的公共接口和数据结构：

    BaseLLM     — 抽象基类，定义了 chat / chat_stream / list_models 三个核心方法
    LLMResponse — 标准化的非流式响应（content + thinking + tool_calls + usage）
    StreamEvent — 流式响应事件（支持 token / thinking / done 三种事件类型）

新增 LLM 后端只需继承 BaseLLM 并实现 chat() 和 list_models() 方法即可。
chat_stream() 有默认实现（非流式转流式），但推荐覆盖以获得真正的流式体验。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class StreamEvent:
    """流式响应事件

    LLM 流式输出时逐 token 产生的事件。Agent 根据 type 字段分发处理：
        • "thinking" — 模型的内部推理过程（如 DeepSeek-R1 的 reasoning_content）
        • "token"    — 普通输出文本 token
        • "done"     — 流式结束，包含完整响应信息（content/thinking/tool_calls/usage）

    Attributes:
        type: 事件类型（"thinking" | "token" | "done"）
        content: token 文本（type=token 时）或完整内容（type=done 时）
        thinking: 完整推理内容（type=done 时）
        thinking_signature: 推理签名（Anthropic 扩展思考要求原样传回）
        tool_calls: 解析后的工具调用列表（type=done 时）
        finish_reason: 结束原因（"stop" / "tool_calls" / "length" 等）
        usage: token 用量统计 {"prompt_tokens", "completion_tokens", "total_tokens"}
    """
    type: str  # 事件类型："thinking" | "token" | "done"
    content: str = ""  # token/thinking 文本（done 时包含完整内容）
    thinking: str = ""  # 完整推理内容（done 时）
    thinking_signature: str = ""  # 推理签名（Anthropic 扩展思考要求回传）
    tool_calls: list[dict] = field(default_factory=list)  # done 时解析的工具调用
    finish_reason: str = "stop"  # 结束原因
    usage: dict = field(default_factory=dict)  # token 用量统计


class LLMResponse:
    """标准化的 LLM 非流式响应

    统一 OpenAI 和 Anthropic 的响应格式，方便 Agent 层统一处理。

    Attributes:
        content: 模型输出的文本内容
        thinking: 模型的内部推理过程（如 DeepSeek-R1 的 reasoning_content）
        thinking_signature: 推理签名（Anthropic 扩展思考专用，要求原样传回）
        tool_calls: 解析后的工具调用列表 [{"name": "...", "arguments": {...}}, ...]
        finish_reason: 结束原因（"stop" / "tool_calls" / "length" / "end_turn"）
        usage: token 用量统计
    """

    def __init__(
        self,
        content: str = "",
        thinking: str = "",
        thinking_signature: str = "",
        tool_calls: list[dict] | None = None,
        finish_reason: str = "stop",
        usage: dict | None = None,
    ):
        self.content = content                  # 模型输出的文本
        self.thinking = thinking                # 推理过程（DeepSeek-R1 等）
        self.thinking_signature = thinking_signature  # Anthropic 推理签名
        self.tool_calls = tool_calls or []      # 工具调用列表
        self.finish_reason = finish_reason      # 结束原因
        self.usage = usage or {}                # token 用量

    def has_tool_calls(self) -> bool:
        """是否包含工具调用"""
        return len(self.tool_calls) > 0

    def __repr__(self):
        return (
            f"LLMResponse(content={self.content[:80]!r}, "
            f"thinking={self.thinking[:40]!r}, "
            f"tool_calls={len(self.tool_calls)}, "
            f"finish={self.finish_reason})"
        )


class BaseLLM(ABC):
    """LLM 抽象基类，定义统一的调用接口

    所有 LLM 后端（OpenAI、Anthropic 等）都继承此类，实现统一的 chat / chat_stream 接口。

    核心方法：
        chat()        — 发送非流式聊天请求，返回 LLMResponse
        chat_stream() — 发送流式聊天请求，逐 token 产生 StreamEvent
        list_models() — 获取可用模型列表
        model_name()  — 当前使用的模型名（property）
    """

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        """
        发送非流式聊天请求，等待完整响应后返回。

        Args:
            messages: 消息列表，格式为 [{"role": "...", "content": "..."}]
            tools: 可选工具定义列表（OpenAI function schema 格式）

        Returns:
            LLMResponse: 标准化响应，包含 content / thinking / tool_calls / usage
        """
        ...

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """
        流式聊天请求，逐 token 返回。

        默认实现是非流式的（一次性返回所有内容），子类应覆盖此方法以实现真正的流式输出。

        Args:
            messages: 消息列表
            tools: 可选工具定义列表

        Yields:
            StreamEvent: type="done" 时包含完整结果
        """
        # 默认实现：调用非流式 chat() 并包装为 done 事件
        response = self.chat(messages, tools)
        yield StreamEvent(
            type="done",
            content=response.content,
            tool_calls=response.tool_calls,
            usage=response.usage,
        )

    @abstractmethod
    def list_models(self) -> list[str]:
        """获取可用模型列表（从 API 获取，失败则返回当前模型）"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前使用的模型名"""
        ...
