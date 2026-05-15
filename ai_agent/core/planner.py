"""
规划与任务分解模块

负责：
1. 分析任务复杂度，决定是否需要规划
2. 将复杂任务分解为可执行的步骤
3. 管理执行进度，支持动态重新规划
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

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

    def mark_step(self, step_id: int, status: StepStatus, result: str = "") -> None:
        """标记步骤状态"""
        for step in self.steps:
            if step.id == step_id:
                step.status = status
                step.result = result
                return

    def format_for_prompt(self) -> str:
        """生成计划描述文本，用于注入 prompt"""
        lines = [f"执行计划（任务: {self.task}）:"]
        for step in self.steps:
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️",
            }.get(step.status, "❓")

            lines.append(f"  {status_icon} Step {step.id}: {step.description}")
            if step.result:
                lines.append(f"      结果: {step.result[:200]}")

        next_step = self.get_next_step()
        if next_step:
            lines.append(f"\n当前应执行: Step {next_step.id} - {next_step.description}")

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
        plan = Plan(
            task=data.get("task", ""),
            steps=steps,
            created_at=data.get("created_at", ""),
            current_step_index=data.get("current_step_index", 0),
        )
        return plan


# ============================================================
# 任务复杂度分析
# ============================================================

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
    multi_task_keywords = [
        "然后", "接着", "之后", "再", "并且", "同时", "另外",
        "第一步", "第二步", "首先", "其次", "最后", "最终",
    ]
    score += sum(1 for kw in multi_task_keywords if kw in text)

    # 复杂操作关键词
    complex_keywords = [
        "分析", "比较", "总结", "整理", "生成报告", "搜索并",
        "查找并", "下载", "安装", "配置", "部署", "调试",
        "写代码", "创建项目", "重构",
    ]
    score += sum(1 for kw in complex_keywords if kw in text)

    # 文件批量操作
    if re.search(r'(所有|批量|每个|全部).*文件', text):
        score += 2

    return min(score, 10)


# ============================================================
# 规划器
# ============================================================

class Planner:
    """任务规划器"""

    def __init__(
        self,
        llm_chat: Callable,
        tools_description: str = "",
    ):
        """
        Args:
            llm_chat: LLM 聊天函数，签名: (messages: list[dict]) -> str
            tools_description: 可用工具的描述文本
        """
        self._chat = llm_chat
        self._tools_desc = tools_description

    def should_plan(self, user_input: str, threshold: int = 3) -> bool:
        """判断是否需要启动规划"""
        complexity = estimate_complexity(user_input)
        logger.info(f"任务复杂度评估: {complexity} (阈值: {threshold})")
        return complexity >= threshold

    def create_plan(self, task: str) -> Plan:
        """使用 LLM 将任务分解为执行步骤"""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个任务规划专家。请将用户的请求分解为具体的执行步骤。\n"
                    f"可用工具：{self._tools_desc}\n\n"
                    "请输出 JSON 格式的计划，每个步骤包含 id, description, tool_hint 字段。\n"
                    '格式：{"steps": [{"id": 1, "description": "...", "tool_hint": "tool_name"}, ...]}\n'
                    "要求：\n"
                    "1. 步骤应该具体、可执行，每个步骤完成一件事\n"
                    "2. 步骤总数不超过 10 个\n"
                    "3. tool_hint 是可选的，如果该步骤明显需要某个工具则给出提示\n"
                    "4. 步骤之间应保持逻辑顺序\n"
                    "5. 仅输出 JSON，不要有其他内容"
                ),
            },
            {"role": "user", "content": f"请为以下任务制定执行计划：\n{task}"},
        ]

        try:
            response = self._chat(messages)
            return self._parse_plan(task, response)
        except Exception as e:
            logger.warning(f"LLM 规划失败: {e}，使用简单分解")
            return self._simple_decompose(task)

    def _parse_plan(self, task: str, response: str) -> Plan:
        """从 LLM 响应中解析执行计划"""
        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*"steps"[\s\S]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                steps = []
                for i, s in enumerate(data.get("steps", [])):
                    steps.append(PlanStep(
                        id=s.get("id", i + 1),
                        description=s.get("description", str(s)),
                        tool_hint=s.get("tool_hint", ""),
                    ))
                if steps:
                    import time
                    return Plan(
                        task=task,
                        steps=steps,
                        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    )
            except json.JSONDecodeError:
                pass

        return self._simple_decompose(task)

    def _simple_decompose(self, task: str) -> Plan:
        """简单关键词分解（不依赖 LLM）"""
        steps = []
        step_id = 1

        # 根据分隔符和连接词拆分
        parts = re.split(r'[，。,\.;；\n]|然后|接着|之后|再|并且|同时', task)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            # 单步任务
            steps.append(PlanStep(
                id=1,
                description=task,
                tool_hint="",
            ))
        else:
            for i, part in enumerate(parts):
                if len(part) > 2:  # 跳过太短的片段
                    steps.append(PlanStep(
                        id=step_id,
                        description=part,
                        tool_hint="",
                    ))
                    step_id += 1

        import time
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
                "content": (
                    "你是一个任务规划专家。当前计划执行遇到问题，请根据反馈调整计划。\n"
                    "请输出调整后的 JSON 格式计划，仅输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始计划：\n{original_plan.format_for_prompt()}\n\n"
                    f"执行反馈：{feedback}\n\n"
                    "请输出调整后的计划："
                ),
            },
        ]

        try:
            response = self._chat(messages)
            return self._parse_plan(original_plan.task, response)
        except Exception as e:
            logger.error(f"重新规划失败: {e}")
            return original_plan
