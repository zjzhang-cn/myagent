"""
LLM 抽象层 — 统一的大语言模型接口

子模块说明：
    base.py       — BaseLLM 抽象基类、LLMResponse、StreamEvent
    openai.py     — OpenAI SDK 集成（支持 12+ 兼容平台，自动推断 Base URL）
    anthropic.py  — Anthropic Claude SDK 集成（扩展思考 support）
"""

# ---------- 基础抽象 ----------
from ai_agent.llm.base import BaseLLM, LLMResponse, StreamEvent

# ---------- OpenAI 后端（可选依赖） ----------
try:
    from ai_agent.llm.openai import OpenAILLM
except ImportError:
    OpenAILLM = None  # type: ignore[assignment]

# ---------- Anthropic 后端（可选依赖） ----------
try:
    from ai_agent.llm.anthropic import AnthropicLLM
except ImportError:
    AnthropicLLM = None  # type: ignore[assignment]

__all__ = ["BaseLLM", "LLMResponse", "StreamEvent", "OpenAILLM", "AnthropicLLM"]
