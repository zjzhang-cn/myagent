"""
AI Agent - 基于 Ollama 的工具调用型智能 Agent

支持：
- 工具调用 (Function Calling)
- 任务规划与分解
- 短期/工作/长期记忆系统
- 本地 Ollama 模型
"""

from ai_agent.config import AgentConfig
from ai_agent.core.agent import Agent, AgentResult, AgentState
from ai_agent.core.memory import LongTermMemory, ShortTermMemory, WorkingMemory
from ai_agent.core.planner import Plan, Planner, PlanStep, StepStatus
from ai_agent.llm.ollama import OllamaLLM
from ai_agent.tools.base import tool
from ai_agent.tools.registry import ToolRegistry, get_registry

__version__ = "1.0.0"

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
    "OllamaLLM",
    "tool",
    "ToolRegistry",
    "get_registry",
]
