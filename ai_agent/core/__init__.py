"""
核心模块 — Agent 引擎、记忆系统、规划器和技能系统

子模块说明：
    agent.py    — Agent 主类，实现 ReAct 循环（Think→Act→Observe→Reflect）
    memory.py   — 三层记忆系统（短期/工作/长期）
    planner.py  — 任务复杂度评估与步骤分解
    skills.py   — Claude Code Skills 规范实现，渐进式披露
"""

# ---------- Agent 引擎 ----------
from ai_agent.core.agent import Agent, AgentResult, AgentState

# ---------- 记忆系统 ----------
from ai_agent.core.memory import (
    LongTermMemory,    # 长期记忆：SQLite 持久化 + 语义搜索
    ShortTermMemory,   # 短期记忆：滑动窗口对话历史
    WorkingMemory,     # 工作记忆：当前任务状态与中间结果
)

# ---------- 规划系统 ----------
from ai_agent.core.planner import Plan, Planner, PlanStep, StepStatus

# ---------- 技能系统 ----------
from ai_agent.core.skills import Skill, SkillRegistry, SkillStep

__all__ = [
    "Agent",
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
]
