"""
工具注册中心 — 工具全生命周期管理

ToolRegistry 是工具系统的核心，负责：
    • 注册/注销 — register() / unregister() 管理工具定义
    • 查询 — get() / list_tools() / list_definitions() 获取工具信息
    • 执行 — execute() 安全执行工具并捕获异常
    • Schema 生成 — to_openai_schemas() 生成 OpenAI Function Calling 格式
    • 动态加载 — load_tools_from_directory() 从目录自动发现 @tool 函数

使用方式：
    from ai_agent.tools.registry import ToolRegistry
    from ai_agent.tools.base import tool

    @tool(name="hello", description="打招呼", params=[])
    def hello() -> str:
        return "你好！"

    registry = ToolRegistry()
    registry.register_function(hello)
    result = registry.execute("hello", {})
"""

import importlib.util
import json
import logging
import os
import sys
import traceback
from typing import Any, Callable

from ai_agent.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表 — 管理所有可用工具

    核心职责：
        1. 维护工具名称 → ToolDefinition 的映射表
        2. 生成 OpenAI 兼容的 function schema（用于 Function Calling）
        3. 安全执行工具调用（参数校验 + 异常捕获）
        4. 动态从目录加载 @tool 装饰的 Python 文件

    线程安全：使用 dict 作为存储，register/unregister 应在初始化阶段完成。
    执行阶段仅读取，多线程并发安全。
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """注册工具定义（直接传入 ToolDefinition 对象）"""
        self._tools[definition.name] = definition
        logger.debug(f"工具已注册: {definition.name}")

    def register_tool_instance(self, tool: BaseTool) -> None:
        """注册工具实例（从 BaseTool 子类）"""
        self.register(tool.definition())

    def register_function(self, func: Callable) -> None:
        """从被 @tool 装饰的函数注册工具

        要求函数具有 _tool_definition 属性（由 @tool 装饰器设置）。
        这是最常用的注册方式。
        """
        if hasattr(func, "_tool_definition"):
            self.register(func._tool_definition)
        else:
            raise ValueError(f"函数 {func.__name__} 缺少 _tool_definition，请使用 @tool 装饰器")

    def unregister(self, name: str) -> None:
        """注销工具（按名称移除）"""
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        """按名称获取工具定义，不存在返回 None"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名称"""
        return list(self._tools.keys())

    def list_definitions(self) -> list[ToolDefinition]:
        """列出所有已注册的工具定义对象"""
        return list(self._tools.values())

    def to_openai_schemas(self) -> list[dict]:
        """生成 OpenAI tools 参数的完整列表

        Returns:
            列表，每项为 {"type": "function", "function": {...}}
            可直接传给 OpenAI chat.completions.create(tools=...)
        """
        return [td.to_openai_schema() for td in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """执行指定工具并返回字符串结果

        执行流程：
            1. 查找工具定义（不存在返回错误提示）
            2. 检查 handler 是否绑定
            3. 调用 handler(**arguments)
            4. 捕获 TypeError（参数错误）并返回友好提示
            5. 捕获其他异常并记录完整 traceback

        Args:
            name: 工具名称
            arguments: 参数字典（如 {"query": "天气", "max_results": 5}）

        Returns:
            执行结果字符串（成功）或错误信息字符串（失败）
        """
        tool_def = self._tools.get(name)
        if not tool_def:
            return f"错误：未找到工具 '{name}'。可用工具: {', '.join(self.list_tools())}"

        if tool_def.handler is None:
            return f"错误：工具 '{name}' 没有绑定的处理函数"

        try:
            result = tool_def.handler(**arguments)
            # 确保返回字符串
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, indent=2)
            return result
        except TypeError as e:
            return (
                f"参数错误: {e}\n"
                f"工具 '{name}' 期望参数: "
                f"{[(p.name, p.type) for p in tool_def.parameters]}"
            )
        except Exception as e:
            logger.error(f"工具执行异常 {name}: {traceback.format_exc()}")
            return f"工具执行异常: {e}"

    def describe_for_prompt(self) -> str:
        """生成工具描述文本（用于 prompt）"""
        lines = ["可用工具列表："]
        for td in self._tools.values():
            params_desc = ", ".join(
                f"{p.name}: {p.type}" + ("?" if not p.required else "")
                for p in td.parameters
            )
            lines.append(f"- {td.name}({params_desc}): {td.description}")
        return "\n".join(lines)


# 全局单例
_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def load_tools_from_directory(directory: str) -> list[Callable]:
    """从目录中动态加载所有 @tool 装饰的 Python 工具模块。

    扫描 directory 下所有 .py 文件（排除 __init__.py 和以 _ 开头的文件），
    动态导入模块，提取其中所有被 @tool 装饰器标记的函数。

    Args:
        directory: 工具模块目录路径

    Returns:
        所有被 @tool 装饰的可调用函数列表

    使用示例:
        tools = load_tools_from_directory("~/my-tools")
        for func in tools:
            registry.register_function(func)
    """
    tools: list[Callable] = []
    directory = os.path.expanduser(directory)

    if not os.path.isdir(directory):
        logger.warning(f"工具目录不存在: {directory}")
        return tools

    # 确保目录在 sys.path 中（用于相对导入）
    parent_dir = os.path.dirname(directory)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".py") or fname.startswith("_") or fname == "__init__.py":
            continue

        fpath = os.path.join(directory, fname)
        module_name = fname[:-3]  # 去掉 .py

        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(
                f"_dynamic_tool_{module_name}", fpath
            )
            if spec is None or spec.loader is None:
                logger.warning(f"无法加载工具模块: {fpath}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning(f"加载工具模块失败: {fpath}: {e}")
            continue

        # 查找模块中所有被 @tool 装饰的函数
        count = 0
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if callable(obj) and hasattr(obj, "_tool_definition"):
                tools.append(obj)
                logger.info(f"从文件加载工具: {fname} -> {attr_name}")
                count += 1

        if count == 0:
            logger.debug(f"工具模块 {fname} 中未找到 @tool 装饰的函数")

    return tools
