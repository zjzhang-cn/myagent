"""
工具系统基类模块

定义工具的元数据结构和注册方式。核心概念：

    ToolParameter  — 工具参数的描述（名称、类型、是否必填、可选枚举值等）
    ToolDefinition — 完整的工具描述（名称、说明、参数列表 + 执行函数）
    BaseTool       — 工具的抽象基类（推荐使用 @tool 装饰器代替继承）
    @tool 装饰器    — 推荐的工具定义方式，将普通函数快速注册为工具

使用 @tool 装饰器的示例：

    from ai_agent.tools.base import tool

    @tool(
        name="get_weather",
        description="获取指定城市的天气信息",
        params=[
            {"name": "city", "type": "string", "description": "城市名称", "required": True},
        ],
    )
    def get_weather(city: str) -> str:
        return f"{city}: 晴, 25°C"
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParameter:
    """工具参数定义

    描述单个参数的元数据，用于生成 OpenAI 兼容的 function schema。

    Attributes:
        name: 参数名称（需与函数签名一致）
        type: 参数类型（"string", "number", "boolean", "array", "object" 等）
        description: 参数用途描述（用于 LLM 理解如何填参）
        required: 是否为必填参数
        default: 默认值（非必填参数使用）
        enum: 可选枚举值列表（如 ["png", "jpg", "gif"]）
    """
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """工具定义 — 完整的工具元数据 + 执行函数

    在 ToolRegistry 中注册和管理。每个已注册的工具对应一个 ToolDefinition 实例。

    Attributes:
        name: 工具名称（唯一标识，LLM 调用时使用）
        description: 工具功能描述（告知 LLM 何时使用此工具）
        parameters: 参数定义列表
        handler: 实际执行的函数（接受 **kwargs，返回字符串或可序列化对象）
        requires_approval: 是否需要用户确认后才能执行（如写文件、删文件等危险操作）
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] | None = None
    requires_approval: bool = False  # 是否需要用户确认后才能执行

    def to_openai_schema(self) -> dict:
        """生成 OpenAI 兼容的 function schema

        Returns:
            dict: {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
        """
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
            if p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class BaseTool(ABC):
    """工具抽象基类

    推荐使用 @tool 装饰器来定义工具（更简洁）。
    仅在需要复杂状态管理时继承此类。

    子类需要实现两个方法：
        definition() → ToolDefinition  — 返回工具的元数据定义
        execute(**kwargs) → str        — 执行工具逻辑，返回字符串结果
    """

    @abstractmethod
    def definition(self) -> ToolDefinition:
        """返回工具的定义"""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        ...


# ------------------------------------------------------------
# @tool 装饰器 — 推荐的工具定义方式
# ------------------------------------------------------------

def tool(
    name: str,
    description: str,
    params: list[dict] | None = None,
    requires_approval: bool = False,
):
    """装饰器：将普通 Python 函数注册为可被 LLM 调用的工具

    装饰后的函数会自动获得 `_tool_definition` 属性，ToolRegistry 通过此属性识别和注册。

    Args:
        name: 工具名称（LLM 调用时使用，建议使用 snake_case）
        description: 工具功能描述（告知 LLM 何时使用、如何使用此工具）
        params: 参数定义列表，每项为包含 name/type/description/required 等键的字典。
                示例: [
                    {"name": "query", "type": "string", "description": "搜索关键词", "required": True},
                    {"name": "max_results", "type": "number", "description": "最多返回条数", "required": False},
                ]
        requires_approval: 是否需要用户确认后才能执行（危险操作如写文件、删文件建议设为 True）

    Usage:
        @tool(
            name="search_web",
            description="搜索互联网获取最新信息",
            params=[
                {"name": "query", "type": "string", "description": "搜索关键词", "required": True},
                {"name": "max_results", "type": "number", "description": "最多返回条数", "required": False},
            ],
        )
        def search_web(query: str, max_results: int = 5) -> str:
            # 实现搜索逻辑
            return f"搜索结果: {query}"
    """
    params = params or []

    def decorator(func: Callable):
        # 将参数字典列表转换为 ToolParameter 对象列表
        tool_params = [
            ToolParameter(
                name=p["name"],
                type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
                enum=p.get("enum"),
            )
            for p in params
        ]

        # 创建 ToolDefinition 并附加到函数对象上，ToolRegistry 通过此属性识别 @tool 函数
        func._tool_definition = ToolDefinition(
            name=name,
            description=description,
            parameters=tool_params,
            handler=func,
            requires_approval=requires_approval,
        )
        return func

    return decorator
