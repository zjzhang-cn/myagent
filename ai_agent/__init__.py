"""
AI Agent - 基于 OpenAI API 的工具调用型智能 Agent

支持：
- 工具调用 (Function Calling)
- 任务规划与分解
- 短期/工作/长期记忆系统
- OpenAI API 及兼容 API
"""

from ai_agent.config import AgentConfig
from ai_agent.core.agent import Agent, AgentResult, AgentState
from ai_agent.core.memory import LongTermMemory, ShortTermMemory, WorkingMemory
from ai_agent.core.planner import Plan, Planner, PlanStep, StepStatus

try:
    from ai_agent.llm.openai import OpenAILLM
except ImportError:
    OpenAILLM = None  # type: ignore[assignment]

from ai_agent.tools.base import tool
from ai_agent.tools.registry import ToolRegistry, get_registry

__version__ = "1.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentState",
    "LongTermMemory",
    "ShortTermMemory",
    "WorkingMemory",
    "Plan",
    "Planner",
    "PlanStep",
    "StepStatus",
    "OpenAILLM",
    "tool",
    "ToolRegistry",
    "get_registry",
]
