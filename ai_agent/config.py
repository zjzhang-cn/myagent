"""
AI Agent 配置文件
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ai_agent.prompts import PromptsConfig


@dataclass
class AgentConfig:
    """Agent 全局配置"""

    # LLM 配置
    model: str = "deepseek-v4-flash"  # 模型名
    llm_type: str = "openai"  # LLM 后端类型：openai / anthropic
    api_key: str | None = None  # API 密钥（不提供则从 LLM_API_KEY 环境变量读取）
    base_url: str | None = None  # API 基础地址（不提供则根据模型名自动推断 或 使用 SDK 默认）
    temperature: float = 0.7
    max_tokens: int = 4096

    # Agent 行为
    max_iterations: int = 0            # ReAct 循环最大轮次（0=不限制）
    max_tool_rounds: int = 0           # 工具调用最大轮数（0=不限制）
    max_tool_calls_per_iteration: int = 3
    max_parallel_tools: int = 5       # 并发工具执行的最大线程数
    parallel_tool_execution: bool = True  # 是否启用并发工具执行
    auto_approve: bool = False  # 是否自动批准所有工具操作（跳过确认）
    verbose: bool = True              # 打印详细日志

    # 规划配置
    enable_planning: bool = True      # 是否启用规划模块
    plan_threshold_complexity: int = 3  # 复杂度 >= 此值时启动规划
    max_replan_attempts: int = 2      # 执行失败时最多重新规划次数

    # 上下文窗口管理
    max_context_tokens: int = 65536  # 最大上下文 token 数，超出自动裁剪（64K）
    max_tool_result_chars: int = 32768  # 单条工具结果最大字符数，超出截断（32K）

    # 记忆配置
    short_term_memory_size: int = 20  # 最近 N 轮对话
    long_term_memory_path: str = "~/.ai_agent/long_term_memory.db"
    state_dir: str = "~/.ai_agent/sessions"  # 会话状态文件存储目录

    # 状态持久化
    auto_snapshot_interval: int = 0  # 每 N 轮迭代自动快照（0=关闭），保存为 _auto_ 前缀
    auto_resume: bool = True  # 启动交互模式时自动恢复上次会话状态
    state_compress_threshold: int = 500  # 字符串超过此长度时压缩存储（字符数）

    # 工具配置
    tools_dir: str = "~/.ai_agent/tools"
    """自定义工具目录，启动时自动加载其中的 @tool 装饰的 Python 文件"""

    # 技能配置
    skills: list[dict] = field(default_factory=list)
    """内联技能定义列表，每项为 Skill 的字典表示"""
    skills_dir: str | None = None
    """技能定义文件目录，默认自动加载 ~/.ai_agent/skills 和 ./.ai_agent/skills。设置后仅加载指定目录"""
    enable_skills: bool = True
    """是否启用技能系统"""

    # 安全配置
    workspace_directories: list[str] = field(default_factory=lambda: ["."])
    """文件操作允许的目录列表（默认仅当前目录）。添加更多: ['.', '~/projects']"""
    shell_allowed_commands: set[str] = field(default_factory=lambda: {
        "ls", "cat", "echo", "pwd", "find", "grep", "head", "tail",
        "wc", "sort", "uniq", "cut", "tr", "date", "whoami", "which",
        "uname", "uptime", "df", "du", "env", "ps", "pgrep",
        "mkdir", "cp", "mv", "touch", "rm", "chmod",
        "tar", "gzip", "gunzip", "zip", "unzip",
        "python", "python3", "pip", "pip3",
        "git", "node", "npm", "npx", "make",
        "awk", "sed", "xargs",
        "curl", "wget",
    })
    shell_allow_all_commands: bool = False
    """设为 True 允许所有 Shell 命令（不推荐，有安全风险）"""

    # 系统提示词（工具调用格式和计划指令由 Agent._build_messages() 动态注入）
    system_prompt: str = field(default_factory=lambda: (
        "你是一个智能 AI Agent，能够通过工具调用和规划来完成复杂任务。\n"
        "请用中文回复用户。\n"
        "完成所有步骤后，基于工具执行结果生成完整、有帮助的总结。"
    ))

    # 提示词配置（可自定义覆盖所有 LLM 提示词模板）
    prompts: "PromptsConfig | None" = None

    def __post_init__(self):
        self.long_term_memory_path = os.path.expanduser(self.long_term_memory_path)
        self.state_dir = os.path.expanduser(self.state_dir)
