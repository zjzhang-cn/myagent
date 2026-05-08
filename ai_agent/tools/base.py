"""
工具基类
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """工具定义，用于注册和生成 schema"""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] | None = None

    def to_openai_schema(self) -> dict:
        """生成 OpenAI 兼容的 function schema"""
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

    def to_ollama_schema(self) -> dict:
        """生成 Ollama 兼容的 tool schema"""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {
                "type": p.type,
                "description": p.description,
            }
            if p.enum:
                prop["enum"] = p.enum
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

    推荐使用 tool 装饰器来定义工具，简单场景也可以继承此类。
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
# 简化工具注册方式：使用装饰器
# ------------------------------------------------------------

def tool(
    name: str,
    description: str,
    params: list[dict] | None = None,
):
    """装饰器：将函数注册为工具

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

        func._tool_definition = ToolDefinition(
            name=name,
            description=description,
            parameters=tool_params,
            handler=func,
        )
        return func

    return decorator
