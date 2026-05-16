"""Tests for ai_agent.tools.registry."""

import pytest

from ai_agent.tools.base import tool, ToolDefinition
from ai_agent.tools.registry import ToolRegistry


def _make_func(name="test_tool", description="A test tool"):
    """Create a function with @tool decorator for testing."""
    @tool(name=name, description=description,
          params=[{"name": "p", "type": "string", "description": "p", "required": True}])
    def func(p: str = ""):
        return f"result: {p}"
    return func


class TestToolRegistry:
    def test_register_and_list(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("tool1"))
        assert len(registry.list_tools()) == 1

    def test_register_multiple(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("t1"))
        registry.register_function(_make_func("t2"))
        assert len(registry.list_tools()) == 2

    def test_register_overwrites_duplicate(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("dup"))
        registry.register_function(_make_func("dup"))  # Overwrites
        assert len(registry.list_tools()) == 1

    def test_get(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("my_tool"))
        definition = registry.get("my_tool")
        assert definition is not None
        assert definition.name == "my_tool"

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_execute(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("greet"))
        result = registry.execute("greet", {"p": "world"})
        assert "world" in result

    def test_execute_nonexistent(self):
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert "未找到工具" in result or "错误" in result

    def test_list_definitions(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("tool1"))
        registry.register_function(_make_func("tool2"))
        defs = registry.list_definitions()
        assert len(defs) == 2
        names = [d.name for d in defs]
        assert "tool1" in names
        assert "tool2" in names

    def test_describe_for_prompt(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("search", "Search the web"))
        description = registry.describe_for_prompt()
        assert "search" in description
        assert "Search the web" in description

    def test_to_openai_schemas(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("my_tool"))
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "my_tool"

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register_function(_make_func("remove_me"))
        assert len(registry.list_tools()) == 1
        registry.unregister("remove_me")
        assert len(registry.list_tools()) == 0

    def test_unregister_nonexistent(self):
        registry = ToolRegistry()
        registry.unregister("nothing")  # Should not raise

    def test_register_tool_instance(self):
        registry = ToolRegistry()
        # Create a BaseTool subclass
        from ai_agent.tools.base import BaseTool

        class TestTool(BaseTool):
            def definition(self):
                return ToolDefinition(
                    name="test_inst",
                    description="A test tool instance",
                    parameters=[
                        {"name": "p", "type": "string", "description": "p", "required": True}
                    ],
                )
            def execute(self, **kwargs):
                return str(kwargs)

        registry.register_tool_instance(TestTool())
        assert registry.get("test_inst") is not None
