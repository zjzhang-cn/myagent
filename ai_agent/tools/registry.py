"""
工具注册中心
"""

import json
import logging
import traceback
from typing import Any, Callable

from ai_agent.tools.base import BaseTool, ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表，管理所有可用工具"""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """注册工具定义"""
        self._tools[definition.name] = definition
        logger.debug(f"工具已注册: {definition.name}")

    def register_tool_instance(self, tool: BaseTool) -> None:
        """注册工具实例"""
        self.register(tool.definition())

    def register_function(self, func: Callable) -> None:
        """从被 @tool 装饰的函数注册"""
        if hasattr(func, "_tool_definition"):
            self.register(func._tool_definition)
        else:
            raise ValueError(f"函数 {func.__name__} 缺少 _tool_definition，请使用 @tool 装饰器")

    def unregister(self, name: str) -> None:
        """注销工具"""
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        """获取工具定义"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有工具名"""
        return list(self._tools.keys())

    def list_definitions(self) -> list[ToolDefinition]:
        """列出所有工具定义"""
        return list(self._tools.values())

    def to_ollama_schemas(self) -> list[dict]:
        """生成 Ollama tools 参数"""
        return [td.to_ollama_schema() for td in self._tools.values()]

    def to_openai_schemas(self) -> list[dict]:
        """生成 OpenAI tools 参数"""
        return [td.to_openai_schema() for td in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """执行指定工具并返回结果"""
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
