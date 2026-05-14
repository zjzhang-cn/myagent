"""
AI Agent 配置文件
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Agent 全局配置"""

    # LLM 配置
    model: str = "minimax-m2.5:cloud"  # Ollama 模型名
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Agent 行为
    max_iterations: int = 15          # ReAct 循环最大轮次
    max_tool_calls_per_iteration: int = 3
    max_parallel_tools: int = 5       # 并发工具执行的最大线程数
    parallel_tool_execution: bool = True  # 是否启用并发工具执行
    verbose: bool = True              # 打印详细日志

    # 规划配置
    enable_planning: bool = True      # 是否启用规划模块
    plan_threshold_complexity: int = 3  # 复杂度 >= 此值时启动规划
    max_replan_attempts: int = 2      # 执行失败时最多重新规划次数

    # 上下文窗口管理
    max_context_tokens: int = 8192   # 最大上下文 token 数，超出自动裁剪
    max_tool_result_chars: int = 3000  # 单条工具结果最大字符数，超出截断

    # 记忆配置
    short_term_memory_size: int = 20  # 最近 N 轮对话
    long_term_memory_path: str = "~/.ai_agent/long_term_memory.json"

    # 工具配置
    tools_dir: Optional[str] = None   # 自定义工具目录

    # 系统提示词
    system_prompt: str = field(default_factory=lambda: (
        "你是一个智能 AI Agent，能够通过工具调用和规划来完成复杂任务。\n"
        "请用中文回复用户。\n"
        "当你需要调用工具时，请严格使用以下 JSON 格式：\n"
        '{"tool_call": {"name": "工具名", "arguments": {"参数名": "参数值"}}}\n'
        "一次可以调用多个工具，用逗号分隔多个 JSON 对象。\n"
        "如果不需要调用工具，直接回复用户即可。"
    ))

    def __post_init__(self):
        self.long_term_memory_path = os.path.expanduser(self.long_term_memory_path)
