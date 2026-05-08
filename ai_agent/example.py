#!/usr/bin/env python3
"""
AI Agent 使用示例

展示各种使用场景：
1. 基础使用：创建 Agent 并提问
2. 自定义工具：注册自定义工具
3. 工具调用：让 Agent 调用工具完成任务
4. 规划模式：让 Agent 分解复杂任务
5. 记忆系统：使用长短期记忆
"""

from ai_agent.config import AgentConfig
from ai_agent.core.agent import Agent
from ai_agent.llm.ollama import OllamaLLM
from ai_agent.tools.base import tool
from ai_agent.tools.registry import ToolRegistry


# ============================================================
# 示例 1: 基础使用
# ============================================================

def example_basic():
    """最简单的用法"""
    print("=" * 60)
    print("示例 1: 基础使用")
    print("=" * 60)

    agent = Agent()
    result = agent.run("你好，请自我介绍一下")

    print(f"Agent: {result.answer}")
    print(f"耗时: {result.elapsed_seconds:.2f}s")
    print(f"迭代次数: {result.iterations}")


# ============================================================
# 示例 2: 自定义工具
# ============================================================

@tool(
    name="get_weather",
    description="获取指定城市的天气信息（模拟）",
    params=[
        {"name": "city", "type": "string", "description": "城市名称", "required": True},
        {"name": "unit", "type": "string", "description": "温度单位: celsius 或 fahrenheit", "required": False},
    ],
)
def get_weather(city: str, unit: str = "celsius") -> str:
    """模拟天气查询"""
    import random
    temp = random.randint(-5, 35)
    conditions = ["晴", "多云", "小雨", "阴天"]
    condition = random.choice(conditions)
    symbol = "°C" if unit == "celsius" else "°F"
    return f"{city}天气: {condition}, 温度 {temp}{symbol}, 湿度 {random.randint(30, 90)}%"


@tool(
    name="calculate",
    description="执行数学计算，支持基本运算和函数",
    params=[
        {"name": "expression", "type": "string", "description": "数学表达式，如 '2 + 3 * 4'", "required": True},
    ],
)
def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    import math
    # 安全白名单
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt, "sin": math.sin,
        "cos": math.cos, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def example_custom_tools():
    """注册自定义工具并使用"""
    print("\n" + "=" * 60)
    print("示例 2: 自定义工具")
    print("=" * 60)

    config = AgentConfig(model="minimax-m2.5:cloud")
    llm = OllamaLLM(model="minimax-m2.5:cloud")

    # 创建注册表并添加自定义工具
    registry = ToolRegistry()
    registry.register_function(get_weather)
    registry.register_function(calculate)

    agent = Agent(config=config, llm=llm, tool_registry=registry)

    print(f"已注册 {len(registry.list_tools())} 个工具:")
    for name in registry.list_tools():
        print(f"  • {name}")

    # 测试工具调用
    result = agent.run("北京今天天气怎么样？顺便帮我算一下 3.14 * 256 等于多少")
    print(f"\nAgent: {result.answer}")


# ============================================================
# 示例 3: 文件操作
# ============================================================

def example_file_ops():
    """使用 Agent 进行文件操作"""
    print("\n" + "=" * 60)
    print("示例 3: 文件操作")
    print("=" * 60)

    from ai_agent.tools.builtin import (
        list_directory,
        read_file,
        run_shell_command,
        write_file,
    )

    config = AgentConfig(model="minimax-m2.5:cloud")
    llm = OllamaLLM(model="minimax-m2.5:cloud")

    registry = ToolRegistry()
    registry.register_function(list_directory)
    registry.register_function(read_file)
    registry.register_function(write_file)
    registry.register_function(run_shell_command)

    agent = Agent(config=config, llm=llm, tool_registry=registry)

    result = agent.run("列出当前目录下的所有 Python 文件")
    print(f"Agent: {result.answer}")


# ============================================================
# 示例 4: 记忆系统
# ============================================================

def example_memory():
    """使用长期记忆"""
    print("\n" + "=" * 60)
    print("示例 4: 记忆系统")
    print("=" * 60)

    config = AgentConfig()
    llm = OllamaLLM(model="minimax-m2.5:cloud")
    agent = Agent(config=config, llm=llm)

    # 手动添加长期记忆
    agent.long_term.remember(
        "用户的名字是张三，是一名 Python 开发者",
        importance=4,
    )
    agent.long_term.remember(
        "用户偏好简洁的代码风格，不喜欢过度设计",
        importance=3,
    )

    print("已添加长期记忆:")
    entries = agent.long_term.list_all()
    for e in entries:
        print(f"  [{e.category}] (重要度:{e.importance}) {e.content}")

    # 搜索记忆
    result = agent.long_term.recall("用户偏好")
    print(f"\n搜索 '用户偏好':\n{result}")

    agent.long_term.close()


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    import sys

    examples = {
        "1": example_basic,
        "2": example_custom_tools,
        "3": example_file_ops,
        "4": example_memory,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("可用示例:")
        print("  1 - 基础使用")
        print("  2 - 自定义工具")
        print("  3 - 文件操作")
        print("  4 - 记忆系统")
        print(f"\n运行: python {__file__} <编号>")
