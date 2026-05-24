"""
规划与任务分解模块

负责：
1. 分析任务复杂度，决定是否需要规划
2. 将复杂任务分解为可执行的步骤（含依赖关系）
3. 管理执行进度，支持动态重新规划
4. 检测可并行执行的步骤
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

from ai_agent.prompts import PromptsConfig

if TYPE_CHECKING:
    from ai_agent.core.skills import Skill, SkillRegistry

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """执行计划中的单步"""
    id: int
    description: str
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    dependencies: list[int] = field(default_factory=list)  # 依赖的步骤 id
    tool_hint: str = ""  # 建议使用的工具名

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "dependencies": self.dependencies,
            "tool_hint": self.tool_hint,
        }

    def can_execute(self, completed_ids: set[int]) -> bool:
        """检查是否所有依赖步骤都已完成"""
        return all(dep in completed_ids for dep in self.dependencies)


@dataclass
class Plan:
    """执行计划"""
    task: str
    steps: list[PlanStep]
    created_at: str = ""
    current_step_index: int = 0

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.FAILED)

    @property
    def progress(self) -> float:
        """完成百分比 0.0 - 1.0"""
        if not self.steps:
            return 0.0
        return self.completed_steps / len(self.steps)

    def get_next_step(self) -> PlanStep | None:
        """获取下一个待执行的步骤（考虑依赖关系）"""
        completed_ids = {
            s.id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        for step in self.steps:
            if step.status == StepStatus.PENDING and step.can_execute(completed_ids):
                return step
        return None

    def get_executable_steps(self) -> list[PlanStep]:
        """获取所有当前可执行的步骤（依赖已满足且未执行）"""
        completed_ids = {
            s.id for s in self.steps if s.status == StepStatus.COMPLETED
        }
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING and s.can_execute(completed_ids)
        ]

    def mark_step(self, step_id: int, status: StepStatus, result: str = "") -> None:
        """标记步骤状态"""
        for step in self.steps:
            if step.id == step_id:
                step.status = status
                step.result = result
                return

    def format_for_prompt(self, prompts: PromptsConfig | None = None) -> str:
        """生成计划描述文本，用于注入 prompt"""
        if prompts is None:
            prompts = PromptsConfig()
        lines = [prompts.plan_title.format(task=self.task)]
        for step in self.steps:
            status_icon = prompts.plan_step_status_icons.get(step.status.value, "❓")

            deps = ""
            if step.dependencies:
                deps = prompts.plan_deps_format.format(
                    deps=", ".join(f"Step {d}" for d in step.dependencies)
                )

            lines.append(prompts.plan_step_format.format(
                icon=status_icon, step_id=step.id, desc=step.description, deps=deps
            ))
            if step.result:
                lines.append(prompts.plan_step_result.format(result=step.result[:200]))

        # 显示当前可并行执行的步骤
        executable = self.get_executable_steps()
        if len(executable) > 1:
            lines.append(
                prompts.plan_parallel_hint.format(
                    count=len(executable),
                    descs=", ".join(f"Step {s.id} ({s.description})" for s in executable)
                )
            )
        elif executable:
            lines.append(
                prompts.plan_current_step.format(
                    step_id=executable[0].id, desc=executable[0].description
                )
            )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "current_step_index": self.current_step_index,
        }

    @staticmethod
    def from_dict(data: dict) -> "Plan":
        """从序列化字典恢复 Plan 对象"""
        steps = []
        for s in data.get("steps", []):
            status_str = s.get("status", "pending")
            try:
                status = StepStatus(status_str)
            except ValueError:
                status = StepStatus.PENDING
            steps.append(PlanStep(
                id=s["id"],
                description=s["description"],
                status=status,
                result=s.get("result", ""),
                dependencies=s.get("dependencies", []),
                tool_hint=s.get("tool_hint", ""),
            ))
        return Plan(
            task=data.get("task", ""),
            steps=steps,
            created_at=data.get("created_at", ""),
            current_step_index=data.get("current_step_index", 0),
        )


# ============================================================
# 任务复杂度分析
# ============================================================

# 连接词：表示多步任务
_MULTI_TASK_KEYWORDS = [
    "然后", "接着", "之后", "再", "并且", "同时", "另外",
    "第一步", "第二步", "首先", "其次", "最后", "最终",
]

# 复杂操作关键词
_COMPLEX_KEYWORDS = [
    "分析", "比较", "总结", "整理", "生成报告", "搜索并",
    "查找并", "下载", "安装", "配置", "部署", "调试",
    "写代码", "创建项目", "重构",
]

# 并行关键词（表示无依赖的并行步骤）
_PARALLEL_KEYWORDS = ["同时", "并且", "另外", "与此同时", "此外"]


def estimate_complexity(user_input: str) -> int:
    """
    估算任务复杂度 (1-10)
    - 简单问答: 1-2
    - 单步工具调用: 3-4
    - 多步任务: 5-7
    - 复杂多阶段任务: 8-10
    """
    score = 1
    text = user_input.lower()

    # 字数越多越复杂
    if len(user_input) > 200:
        score += 2
    elif len(user_input) > 100:
        score += 1

    # 多任务关键词
    score += sum(1 for kw in _MULTI_TASK_KEYWORDS if kw in text)

    # 复杂操作关键词
    score += sum(1 for kw in _COMPLEX_KEYWORDS if kw in text)

    # 文件批量操作
    if re.search(r'(所有|批量|每个|全部).*文件', text):
        score += 2

    # URL + 操作模式（如"抓取"、"下载"）
    if re.search(r'https?://', text):
        score += 1

    return min(score, 10)


def _infer_dependencies(parts: list[str]) -> list[list[int]]:
    """根据步骤描述推断依赖关系。

    Args:
        parts: 各步骤的描述文本列表

    Returns:
        dependencies 列表，deps[i] = step i+1 依赖的 step id 列表
    """
    n = len(parts)
    if n <= 1:
        return [[]]

    deps: list[list[int]] = [[] for _ in range(n)]

    # 默认依赖：按照顺序，每个步骤依赖前一步（除非是并行步骤）
    # 并行步骤：当前步骤和上一步之间用"同时"、"另外"等连接
    for i in range(1, n):
        prev = parts[i - 1]
        curr = parts[i]
        # 检查当前步骤是否与上一步是并行关系
        is_parallel = any(kw in curr[:10] for kw in _PARALLEL_KEYWORDS)
        if not is_parallel:
            deps[i].append(i)  # 依赖前一步（Step id = i）

    return deps


def _split_task(task: str) -> list[str]:
    """将任务文本拆分为多个步骤描述。

    按分隔符拆分，同时保留连接词作为步骤前缀以便后续推断依赖。
    """
    # 按分隔符拆分，保留连接词
    parts = re.split(r'[，。,\.;；\n]+', task)
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    return parts


# ============================================================
# 规划器
# ============================================================

class Planner:
    """任务规划器"""

    def __init__(
        self,
        llm_chat: Callable,
        tools_description: str = "",
        skill_registry: SkillRegistry | None = None,
        prompts: PromptsConfig | None = None,
    ):
        """
        Args:
            llm_chat: LLM 聊天函数，签名: (messages: list[dict]) -> str
            tools_description: 可用工具的描述文本
            skill_registry: 技能注册表（可选），提供后启用技能匹配快速路径
            prompts: 提示词配置（可选，默认使用 PromptsConfig()）
        """
        self._chat = llm_chat
        self._tools_desc = tools_description
        self._skill_registry = skill_registry
        self.prompts = prompts or PromptsConfig()

    def should_plan(self, user_input: str, threshold: int = 3) -> bool:
        """判断是否需要启动规划

        两步策略：
        1. 关键词启发式评分
        2. 边界值时用 LLM 二次验证（减少误触）
        """
        complexity = estimate_complexity(user_input)
        logger.info(f"任务复杂度评估: {complexity} (阈值: {threshold})")

        if complexity >= threshold + 2:
            # 远超过阈值，直接启动规划
            return True

        if complexity >= threshold - 1:
            # 边界值（threshold-1 到 threshold+1），LLM 二次验证
            logger.info("复杂度处于边界，启动 LLM 二次验证...")
            return self._llm_verify_complexity(user_input)

        return False

    def _llm_verify_complexity(self, user_input: str) -> bool:
        """用 LLM 判断任务是否复杂到需要规划"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.prompts.verify_complexity_system,
                },
                {
                    "role": "user",
                    "content": self.prompts.verify_complexity_user.format(user_input=user_input),
                },
            ]
            response = self._chat(messages)

            # 判断 LLM 是否认为需要规划
            positive = False
            for word in ["是", "yes", "true", "需要", "复杂"]:
                if word in response.lower():
                    positive = True
                    break
            # "否" 比 "是" 优先级高
            for word in ["否", "no", "不需要", "简单"]:
                if word in response.lower():
                    positive = False
                    break

            logger.info(f"LLM 二次验证结果: {'需要规划' if positive else '不需要规划'} ({response[:100]})")
            return positive
        except Exception as e:
            logger.warning(f"LLM 二次验证失败: {e}，回退到关键词判断")
            return estimate_complexity(user_input) >= 4

    def plan_with_skill(self, task: str, skill_name: str) -> Plan | None:
        """使用指定名称的技能来生成计划。

        Args:
            task: 任务描述
            skill_name: 技能名称（支持部分匹配）

        Returns:
            匹配成功时返回 Plan，否则返回 None
        """
        if not self._skill_registry:
            return None
        skill = self._skill_registry.find_by_name(skill_name)
        if skill:
            return self._plan_from_skill(task, skill)
        return None

    def create_plan(self, task: str) -> Plan:
        """使用 LLM 将任务分解为执行步骤（含依赖关系）。

        技能元数据会注入到提示词中作为参考（渐进式披露第一层）。
        LLM 可以通过返回特殊的 skill 字段来请求加载完整技能内容。
        """
        # 构建技能元数据部分
        skills_section = ""
        if self._skill_registry:
            skills_section = self._skill_registry.describe_for_prompt()
            if skills_section:
                skills_section = f"\n{skills_section}\n"

        messages = [
            {
                "role": "system",
                "content": self.prompts.plan_system.format(
                    tools_description=f"可用工具：{self._tools_desc}",
                    skills_section=skills_section,
                ),
            },
            {"role": "user", "content": self.prompts.plan_user.format(task=task)},
        ]

        try:
            response = self._chat(messages)
            return self._parse_plan(task, response)
        except Exception as e:
            logger.warning(f"LLM 规划失败: {e}，使用简单分解")
            return self._simple_decompose(task)

    def _plan_from_skill(self, task: str, skill: Skill) -> Plan:
        """使用技能模板生成执行计划"""
        logger.info(f"使用技能生成计划: '{skill.name}' ({skill.description})")
        return skill.to_plan(task)

    def _parse_plan(self, task: str, response: str) -> Plan:
        """从 LLM 响应中解析执行计划"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*"steps"[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                steps = []
                for i, s in enumerate(data.get("steps", [])):
                    deps = s.get("dependencies", [])
                    if not isinstance(deps, list):
                        deps = []
                    steps.append(PlanStep(
                        id=s.get("id", i + 1),
                        description=s.get("description", str(s)),
                        tool_hint=s.get("tool_hint", ""),
                        dependencies=deps,
                    ))
                if steps:
                    return Plan(
                        task=task,
                        steps=steps,
                        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
            except json.JSONDecodeError:
                pass

        return self._simple_decompose(task)

    def _simple_decompose(self, task: str) -> Plan:
        """增强的关键词分解（不依赖 LLM）

        改进：
        - 更精确的步骤拆分
        - 自动推断依赖关系
        - 检测并行步骤
        """
        parts = _split_task(task)

        if len(parts) <= 1:
            # 单步任务
            return Plan(
                task=task,
                steps=[PlanStep(id=1, description=task, tool_hint="")],
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        # 多步任务：创建步骤并推断依赖
        steps = []
        for i, part in enumerate(parts):
            steps.append(PlanStep(
                id=i + 1,
                description=part,
                tool_hint="",
            ))

        # 推断依赖关系
        deps = _infer_dependencies(parts)
        for i, dep_list in enumerate(deps):
            if dep_list:
                steps[i].dependencies = dep_list

        return Plan(
            task=task,
            steps=steps,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def replan(
        self,
        original_plan: Plan,
        feedback: str,
    ) -> Plan:
        """根据执行反馈重新规划"""
        messages = [
            {
                "role": "system",
                "content": self.prompts.replan_system,
            },
            {
                "role": "user",
                "content": self.prompts.replan_user.format(
                    original_plan=original_plan.format_for_prompt(self.prompts),
                    feedback=feedback,
                ),
            },
        ]

        try:
            response = self._chat(messages)
            return self._parse_plan(original_plan.task, response)
        except Exception as e:
            logger.error(f"重新规划失败: {e}")
            return original_plan
