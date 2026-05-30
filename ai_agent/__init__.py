"""
AI Agent — 基于 OpenAI/Anthropic API 的工具调用型智能 Agent 框架

===== 核心能力 =====
• 工具调用 (Function Calling) — 支持 LLM 原生调用 + 文本解析多级回退
• 任务规划与分解 — 自动评估复杂度，将复杂任务拆解为有序步骤
• 三层记忆系统 — 短期记忆(滑动窗口) + 工作记忆(任务状态) + 长期记忆(SQLite+语义搜索)
• 双 LLM 后端 — OpenAI API(含12+兼容平台) + Anthropic Claude(原生扩展思考)
• 技能系统 — 遵循 Claude Code Skills 规范，渐进式披露
• 安全沙箱 — 路径验证 + 命令白名单双重保护

===== 快速开始 =====
    from ai_agent import Agent, AgentConfig

    agent = Agent(AgentConfig(model="deepseek-v4-flash"))
    result = agent.run("你好，请帮我搜索今天的科技新闻")
    print(result.answer)

===== 公共 API 导出 =====
"""

# ---------- 配置 ----------
from ai_agent.config import AgentConfig

# ---------- 核心引擎 ----------
from ai_agent.core.agent import Agent, AgentResult, AgentState

# ---------- 记忆系统 ----------
from ai_agent.core.memory import LongTermMemory, ShortTermMemory, WorkingMemory

# ---------- 规划系统 ----------
from ai_agent.core.planner import Plan, Planner, PlanStep, StepStatus

# ---------- 技能系统 ----------
from ai_agent.core.skills import Skill, SkillRegistry, SkillStep

# ---------- 提示词配置 ----------
from ai_agent.prompts import PromptsConfig

# ---------- LLM 后端（可选依赖，缺失时设为 None） ----------
try:
    from ai_agent.llm.openai import OpenAILLM
except ImportError:
    OpenAILLM = None  # type: ignore[assignment]

try:
    from ai_agent.llm.anthropic import AnthropicLLM
except ImportError:
    AnthropicLLM = None  # type: ignore[assignment]

# ---------- 工具系统 ----------
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
    "Skill",
    "SkillRegistry",
    "SkillStep",
    "OpenAILLM",
    "AnthropicLLM",
    "PromptsConfig",
    "tool",
    "ToolRegistry",
    "get_registry",
]
