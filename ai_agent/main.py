#!/usr/bin/env python3
"""
AI Agent 主入口

用法：
    # 交互模式
    python -m ai_agent.main

    # 单次查询
    python -m ai_agent.main "帮我搜索今天的新闻"

    # 指定模型
    python -m ai_agent.main --model qwen2.5:14b "分析这个项目结构"

    # 列出可用模型
    python -m ai_agent.main --list-models
"""

import argparse
import logging
import sys
from pathlib import Path

from ai_agent.config import AgentConfig
from ai_agent.core.agent import Agent
from ai_agent.llm.ollama import OllamaLLM
from ai_agent.tools.builtin import (
    delete_file,
    fetch_url,
    list_directory,
    read_file,
    run_shell_command,
    search_web,
    write_file,
)
from ai_agent.tools.registry import ToolRegistry

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai_agent")


def create_agent(
    model: str = "minimax-m2.5:cloud",
    host: str = "http://localhost:11434",
    temperature: float = 0.7,
    verbose: bool = False,
    on_step: callable = None,
    on_token: callable = None,
) -> Agent:
    """创建并配置 Agent"""
    config = AgentConfig(
        model=model,
        ollama_host=host,
        temperature=temperature,
        verbose=verbose,
    )

    llm = OllamaLLM(
        model=model,
        host=host,
        temperature=temperature,
    )

    # 创建工具注册表并注册内置工具
    registry = ToolRegistry()
    for func in [
        read_file,
        write_file,
        list_directory,
        delete_file,
        run_shell_command,
        search_web,
        fetch_url,
    ]:
        registry.register_function(func)

    agent = Agent(
        config=config,
        llm=llm,
        tool_registry=registry,
        on_step=on_step,
        on_token=on_token,
    )

    return agent


def _print_step(event: str, data: dict) -> None:
    """实时打印 Agent 思考过程"""
    if event == "start":
        print(f"\n{'─' * 40}")
    elif event == "planning":
        steps = data.get("steps", [])
        if steps:
            print(f"\n📋 任务规划 ({len(steps)} 步):")
            for s in steps:
                print(f"   {s['id']}. {s['description']}")
            print()
    elif event == "token":
        # 流式 token — 由 on_token 回调处理，这里不做额外输出
        pass
    elif event == "thinking":
        content = data.get("content", "")
        iteration = data.get("iteration", 0)
        # 流式模式下 token 已经实时显示过了，这里只做摘要
        if "tool_call" in content:
            print(f"  🔧 [{iteration}] 决定调用工具...")
        elif len(content) <= 200:
            # 非流式模式：完整显示
            print(f"  💭 [{iteration}] {content}")
        else:
            print(f"  💭 [{iteration}] {content[:200]}...")
    elif event == "acting":
        tool = data.get("tool", "")
        args = data.get("arguments", {})
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        print(f"\n  🔧 调用: {tool}({args_str})")
    elif event == "observing":
        tool = data.get("tool", "")
        result = data.get("result", "")
        summary = result[:120].replace("\n", " ") + ("..." if len(result) > 120 else "")
        print(f"  📊 {tool} → {summary}")
    elif event == "done":
        pass  # 最终答案在下面单独打印


def _make_stream_printer():
    """创建一个带状态的流式打印回调"""
    state = {"started": False, "iteration": 0}

    def on_token(token: str) -> None:
        if not state["started"]:
            print("\n💭 ", end="", flush=True)
            state["started"] = True
        print(token, end="", flush=True)

    return on_token, state


def interactive_mode(agent: Agent) -> None:
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print(f"🤖 AI Agent (模型: {agent.llm.model_name})")
    print(f"📦 可用工具: {len(agent.tool_registry.list_tools())} 个")
    print(f"🧠 规划: {'启用' if agent.config.enable_planning else '关闭'}")
    print("输入 'quit' 或 'exit' 退出, 'reset' 重置, 'tools' 查看工具")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("✅ Agent 已重置")
            continue

        if user_input.lower() == "tools":
            print("\n可用工具：")
            for td in agent.tool_registry.list_definitions():
                params = ", ".join(p.name for p in td.parameters)
                print(f"  • {td.name}({params}): {td.description}")
            print()
            continue

        if user_input.lower() == "memory":
            stats = agent.long_term.stats()
            print(f"\n长期记忆统计: 共 {stats['total']} 条")
            entries = agent.long_term.list_all(limit=5)
            for e in entries:
                print(f"  [{e.category}] {e.content[:100]}")
            print()
            continue

        # 执行（带流式思考过程显示）
        stream_print, stream_state = _make_stream_printer()
        agent.on_token = stream_print

        result = agent.run(user_input)

        # 流式结束后换行
        if stream_state["started"]:
            print()

        print(f"\n{'─' * 40}")
        print(f"✅ Agent: {result.answer}")
        print(f"   (耗时 {result.elapsed_seconds:.1f}s, {result.iterations} 轮迭代)")
        if result.plan:
            print(f"   (计划: {result.plan.completed_steps}/{result.plan.total_steps} 步完成)")
        print()


def list_models(host: str = "http://localhost:11434") -> None:
    """列出可用模型"""
    llm = OllamaLLM(host=host)
    models = llm.list_models()
    print(f"\n可用模型 ({len(models)} 个):")
    for m in models:
        print(f"  • {m}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent - 基于 Ollama 的工具调用型智能 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 交互模式
  %(prog)s "搜索今天的科技新闻"                  # 单次查询
  %(prog)s --model qwen2.5:14b "分析代码"       # 指定模型
  %(prog)s --list-models                       # 列出可用模型
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="要执行的查询（不提供则进入交互模式）",
    )
    parser.add_argument(
        "--model", "-m",
        default="minimax-m2.5:cloud",
        help="Ollama 模型名 (默认: minimax-m2.5:cloud)",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama 服务地址 (默认: http://localhost:11434)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="生成温度 (默认: 0.7)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出可用模型并退出",
    )
    parser.add_argument(
        "--no-planning",
        action="store_true",
        help="禁用任务规划",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger("ai_agent").setLevel(logging.WARNING)

    if args.list_models:
        list_models(args.host)
        return

    # 创建 Agent（交互模式默认流式）
    stream_print, stream_state = _make_stream_printer()
    agent = create_agent(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        verbose=args.verbose,
        on_step=_print_step,
        on_token=stream_print if not args.query else None,  # 单次查询默认不流式
    )

    if args.no_planning:
        agent.config.enable_planning = False

    if args.query:
        # 单次查询模式
        print(f"\n查询: {args.query}")
        print(f"模型: {args.model}")
        print("-" * 40)
        # 单次查询也使用流式
        stream_single, stream_single_state = _make_stream_printer()
        agent.on_token = stream_single
        result = agent.run(args.query)
        if stream_single_state["started"]:
            print()
        print(f"\n{result.answer}")
        print(f"\n(耗时 {result.elapsed_seconds:.1f}s, {result.iterations} 轮)")
    else:
        # 交互模式
        interactive_mode(agent)

    # 清理
    agent.long_term.close()


if __name__ == "__main__":
    main()
