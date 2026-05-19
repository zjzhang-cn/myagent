"""
技能系统模块

遵循 Claude Code Skills 规范 (agentskills.io):
- 每个 Skill 是一个目录，其中包含 Skill.md 文件
- Skill.md 使用 YAML frontmatter 定义元数据（name, description, dependencies）
- Markdown 正文包含技能指令（渐进式披露）
- description 字段是 LLM 决定何时调用技能的关键依据
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from ai_agent.core.planner import Plan, PlanStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YAML frontmatter 解析（轻量级，不依赖 PyYAML）
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析 Skill.md 中的 YAML frontmatter。

    Returns:
        (metadata_dict, body_text)
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    raw_meta = m.group(1)
    body = m.group(2).strip()
    meta: dict = {}

    for line in raw_meta.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            meta[key] = val

    return meta, body


# ---------------------------------------------------------------------------
# SkillStep — 技能中的单步模板
# ---------------------------------------------------------------------------


@dataclass
class SkillStep:
    """技能中的单步模板（不含运行时状态）。

    这是对 PlanStep 的轻量封装，用于在技能中预定义执行步骤。
    dependencies 中的值为 1-based 步骤序号。
    """

    description: str
    tool_hint: str = ""
    dependencies: list[int] = field(default_factory=list)

    def to_plan_step(self, step_id: int) -> PlanStep:
        return PlanStep(
            id=step_id,
            description=self.description,
            tool_hint=self.tool_hint,
            dependencies=self.dependencies.copy(),
        )

    @staticmethod
    def from_dict(data: dict) -> "SkillStep":
        return SkillStep(
            description=data.get("description", ""),
            tool_hint=data.get("tool_hint", ""),
            dependencies=data.get("dependencies", []),
        )


# ---------------------------------------------------------------------------
# Skill — 遵循 Claude Code Skills 规范
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """一个 Claude Code 兼容的技能。

    规范字段（来自 YAML frontmatter）：
    - name: 技能名称（最多 64 字符）
    - description: 技能描述（最多 200 字符），LLM 据此决定何时调用
    - dependencies: 可选，如 "python>=3.8, pandas>=1.5.0"

    扩展字段：
    - body: Skill.md 的 Markdown 正文（渐进式披露的第二层）
    - path: 技能目录的路径
    - steps: 可选的预定义执行步骤
    """

    name: str
    description: str
    body: str = ""
    dependencies: str = ""
    path: str = ""
    steps: list[SkillStep] = field(default_factory=list)

    def __post_init__(self):
        # 规范约束：name 最多 64 字符
        if len(self.name) > 64:
            logger.warning(f"技能名称超过 64 字符，将被截断: {self.name!r}")
            self.name = self.name[:64]
        # 规范约束：description 最多 200 字符
        if len(self.description) > 200:
            logger.warning(f"技能描述超过 200 字符，将被截断: {self.description[:50]!r}...")
            self.description = self.description[:200]

    def to_plan(self, task: str) -> Plan:
        """将技能的步骤模板转换为执行计划。"""
        if not self.steps:
            # 无预定义步骤时，以技能 body 作为单个指导步骤
            return Plan(
                task=task,
                steps=[
                    PlanStep(
                        id=1,
                        description=f"按照技能 '{self.name}' 的指导完成任务",
                        tool_hint="",
                    )
                ],
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

        plan_steps = [s.to_plan_step(step_id=i + 1) for i, s in enumerate(self.steps)]
        return Plan(
            task=task,
            steps=plan_steps,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def format_for_prompt(self) -> str:
        """生成技能的 prompt 描述文本（用于注入系统提示词）。"""
        lines = [f"### {self.name}", f"描述: {self.description}"]
        if self.dependencies:
            lines.append(f"依赖: {self.dependencies}")
        if self.body:
            lines.append(f"\n{self.body}")
        return "\n".join(lines)

    @staticmethod
    def from_dict(data: dict) -> "Skill":
        """从字典反序列化技能。"""
        steps_data = data.get("steps", [])
        steps = [SkillStep.from_dict(s) for s in steps_data]
        return Skill(
            name=data.get("name", ""),
            description=data.get("description", ""),
            body=data.get("body", ""),
            dependencies=data.get("dependencies", ""),
            path=data.get("path", ""),
            steps=steps,
        )


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------


class SkillRegistry:
    """技能注册表，管理所有已加载的技能。

    支持：
    - 按名称注册/注销/查询技能
    - 生成技能元数据摘要（用于注入系统提示词，渐进式披露第一层）
    - 按名称获取完整技能内容（渐进式披露第二层）
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill
        logger.debug(f"技能已注册: {skill.name}")

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)
        logger.debug(f"技能已注销: {name}")

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_all(self) -> list[Skill]:
        return list(self._skills.values())

    def describe_for_prompt(self) -> str:
        """生成技能元数据摘要（第一层渐进式披露）。

        LLM 根据这些描述判断是否需要加载完整技能内容。
        仅包含 name + description，不含 body。
        """
        if not self._skills:
            return ""

        lines = ["## 可用技能"]
        for skill in self._skills.values():
            deps = f" (依赖: {skill.dependencies})" if skill.dependencies else ""
            lines.append(f"- **{skill.name}**: {skill.description}{deps}")
        lines.append(
            "\n当任务匹配某个技能描述时，你可以调用 `use_skill` 工具来加载该技能的完整指导内容。"
        )
        return "\n".join(lines)

    def find_by_name(self, name: str) -> Skill | None:
        """按名称查找技能（大小写不敏感，部分匹配）。"""
        name_lower = name.lower()
        # 精确匹配
        if name_lower in self._skills:
            return self._skills[name_lower]
        # 部分匹配
        for skill in self._skills.values():
            if name_lower in skill.name.lower():
                return skill
        return None


# ---------------------------------------------------------------------------
# 从目录加载技能
# ---------------------------------------------------------------------------


def _load_skill_from_dir(skill_dir: str) -> Skill | None:
    """从技能目录加载单个 Skill。

    目录中必须包含 Skill.md 文件。
    """
    skill_md_path = os.path.join(skill_dir, "Skill.md")
    if not os.path.isfile(skill_md_path):
        logger.warning(f"技能目录缺少 Skill.md: {skill_dir}")
        return None

    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        logger.warning(f"读取 Skill.md 失败: {skill_md_path}: {e}")
        return None

    meta, body = _parse_frontmatter(content)

    name = meta.get("name", "")
    if not name:
        # 回退：使用目录名作为技能名
        name = os.path.basename(os.path.normpath(skill_dir))
        logger.warning(f"Skill.md 缺少 name 字段，使用目录名: {name}")

    description = meta.get("description", "")

    skill = Skill(
        name=name,
        description=description,
        body=body,
        dependencies=meta.get("dependencies", ""),
        path=os.path.abspath(skill_dir),
    )
    logger.debug(f"从目录加载技能: {skill_dir} -> {skill.name}")
    return skill


def load_skills_from_directory(base_dir: str) -> list[Skill]:
    """从基础目录中加载所有技能。

    扫描 base_dir 下的每个子目录，对其中的 Skill.md 进行解析。
    同时兼容旧格式：base_dir 下直接放置的 .json 文件。

    目录结构示例:
        ~/.ai_agent/skills/
        ├── my-skill/
        │   ├── Skill.md
        │   └── resources/
        └── another-skill/
            └── Skill.md
    """
    skills: list[Skill] = []
    base_dir = os.path.expanduser(base_dir)

    if not os.path.isdir(base_dir):
        logger.warning(f"技能基础目录不存在: {base_dir}")
        return skills

    for entry in sorted(os.listdir(base_dir)):
        skill_dir = os.path.join(base_dir, entry)

        # 新格式：子目录包含 Skill.md
        if os.path.isdir(skill_dir):
            skill = _load_skill_from_dir(skill_dir)
            if skill:
                skills.append(skill)

        # 兼容旧格式：.json 文件
        elif entry.endswith(".json"):
            _load_legacy_json(skill_dir, skills)

    return skills


def _load_legacy_json(filepath: str, skills: list[Skill]) -> None:
    """兼容旧版 JSON 格式的技能定义。"""
    import json

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        skill = Skill.from_dict(data)
        if skill.name:
            skills.append(skill)
            logger.debug(f"从 JSON 加载技能（旧格式）: {filepath} -> {skill.name}")
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"技能 JSON 解析失败: {filepath}: {e}")
    except Exception as e:
        logger.warning(f"加载技能文件异常: {filepath}: {e}")
