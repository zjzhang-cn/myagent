"""
AI Agent 全局配置模块

AgentConfig 是一个 dataclass，集中管理所有配置项，支持：
    - 环境变量回退（API_KEY、BASE_URL、MODEL 等）
    - ~ 路径自动展开
    - 合理的默认值（开箱即用）

使用方式：
    from ai_agent.config import AgentConfig

    # 使用默认配置
    config = AgentConfig()

    # 自定义配置
    config = AgentConfig(
        model="gpt-4o",
        temperature=0.3,
        enable_planning=False,
        max_context_tokens=32768,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ai_agent.prompts import PromptsConfig


@dataclass
class AgentConfig:
    """Agent 全局配置（dataclass，所有字段都有默认值）

    配置项分为几个大类：
        • LLM 配置 — 模型选择、API 密钥、温度等
        • Agent 行为 — 循环限制、工具并发、自动批准等
        • 规划配置 — 任务分解与重新规划控制
        • 上下文管理 — Token 窗口与工具结果截断
        • 记忆配置 — 三层记忆系统的参数
        • 工具与技能 — 自定义工具目录、技能启用等
        • 安全配置 — 路径沙箱与命令白名单
        • 提示词配置 — 系统提示词与 LLM 交互模板
    """

    # ============================================================
    # LLM 配置
    # ============================================================

    model: str = "deepseek-v4-flash"  # 模型名称（如 gpt-4o / deepseek-chat / claude-sonnet-4-20250514）
    llm_type: str = "openai"  # LLM 后端类型："openai" 或 "anthropic"
    api_key: str | None = None  # API 密钥（留空则从 LLM_API_KEY 环境变量读取）
    base_url: str | None = None  # API 基础地址（留空则根据模型名自动推断，或从 LLM_BASE_URL 环境变量读取）
    temperature: float = 0.7      # 生成温度（0.0=确定性, 2.0=高随机性）
    max_tokens: int = 4096        # 单次生成最大 token 数

    # ============================================================
    # Agent 行为配置
    # ============================================================

    max_iterations: int = 0            # ReAct 循环最大轮次，0=不限制（防止无限循环）
    max_tool_rounds: int = 0           # 工具调用最大轮数，0=不限制
    max_tool_calls_per_iteration: int = 3  # 单轮迭代中最多可以调用几个工具
    max_parallel_tools: int = 5        # 并发工具执行的最大线程数
    parallel_tool_execution: bool = True  # 是否启用并发工具执行（互不依赖的工具同时运行）
    auto_approve: bool = False         # 是否自动批准所有工具操作（跳过用户确认，慎用）

    # ============================================================
    # 日志与输出
    # ============================================================

    verbose: bool = True               # 是否打印详细日志到控制台

    # ============================================================
    # 任务规划配置
    # ============================================================

    enable_planning: bool = True       # 是否启用任务规划模块
    plan_threshold_complexity: int = 3 # 任务复杂度 >= 此值时自动启动规划（范围 1-10）
    max_replan_attempts: int = 2       # 执行失败时最多重新规划次数

    # ============================================================
    # 上下文窗口管理
    # ============================================================

    max_context_tokens: int = 65536    # 最大上下文 token 数（超出自动裁剪旧消息），默认 64K
    max_tool_result_chars: int = 32768 # 单条工具结果最大字符数（超出截断），默认 32K

    # ============================================================
    # 记忆系统配置
    # ============================================================

    short_term_memory_size: int = 20   # 短期记忆滑动窗口大小（最近 N 轮对话）
    long_term_memory_path: str = "~/.ai_agent/long_term_memory.db"  # 长期记忆 SQLite 路径
    state_dir: str = "~/.ai_agent/sessions"  # 会话状态文件存储目录

    # ============================================================
    # 状态持久化配置
    # ============================================================

    auto_snapshot_interval: int = 0    # 每 N 轮迭代自动保存会话快照，0=关闭
    auto_resume: bool = True           # 交互模式启动时是否自动恢复上次会话
    state_compress_threshold: int = 500 # 字符串超过此长度时启用 zlib 压缩存储

    # ============================================================
    # 工具与技能配置
    # ============================================================

    tools_dir: str = "~/.ai_agent/tools"
    """自定义工具目录。启动时自动扫描并加载其中所有 @tool 装饰的 Python 文件"""

    skills: list[dict] = field(default_factory=list)
    """内联技能定义列表，每项为 Skill 的字典表示。
    示例: [{"name": "code-review", "description": "代码审查技能", "body": "..."}]"""

    skills_dir: str | None = None
    """技能定义文件目录。留空时默认加载 ~/.ai_agent/skills 和 ./.ai_agent/skills。
    显式设置后仅加载指定目录。支持 : 分隔多个路径"""

    enable_skills: bool = True  # 是否启用技能系统

    # ============================================================
    # 安全配置（路径沙箱 + 命令白名单）
    # ============================================================

    workspace_directories: list[str] = field(default_factory=lambda: ["."])
    """文件操作允许的目录列表（默认仅当前工作目录）。
    示例: ['.', '~/projects', '/tmp']"""

    shell_allowed_commands: set[str] = field(default_factory=lambda: {
        # 文件查看
        "ls", "cat", "echo", "pwd",
        # 文本处理
        "find", "grep", "head", "tail", "wc", "sort", "uniq", "cut", "tr",
        # 系统信息
        "date", "whoami", "which", "uname", "uptime", "df", "du", "env",
        # 进程管理
        "ps", "pgrep",
        # 文件操作
        "mkdir", "cp", "mv", "touch", "rm", "chmod",
        # 压缩/归档
        "tar", "gzip", "gunzip", "zip", "unzip",
        # 开发工具
        "python", "python3", "pip", "pip3",
        "git", "node", "npm", "npx", "make",
        # 文本流处理
        "awk", "sed", "xargs",
        # 网络请求
        "curl", "wget",
    })
    """Shell 命令白名单。不在白名单中的命令会被拦截"""

    shell_allow_all_commands: bool = False
    """设为 True 允许所有 Shell 命令（不推荐，有安全风险）"""

    # ============================================================
    # 系统提示词（工具调用格式和计划指令由 Agent._build_messages() 动态注入）
    # ============================================================

    system_prompt: str = field(default_factory=lambda: (
        "你是一个智能 AI Agent，能够通过工具调用和规划来完成复杂任务。\n"
        "请用中文回复用户。\n"
        "完成所有步骤后，基于工具执行结果生成完整、有帮助的总结。"
    ))
    """系统提示词。Agent 启动时写入短期记忆。工具调用格式和计划进度会在运行时动态追加"""

    # ============================================================
    # 提示词配置（可自定义覆盖所有 LLM 交互模板）
    # ============================================================

    prompts: "PromptsConfig | None" = None
    """自定义提示词模板。设置后将覆盖默认的 PromptsConfig"""

    def __post_init__(self):
        """初始化后处理：展开配置中的 ~ 路径为绝对路径"""
        self.long_term_memory_path = os.path.expanduser(self.long_term_memory_path)
        self.state_dir = os.path.expanduser(self.state_dir)
