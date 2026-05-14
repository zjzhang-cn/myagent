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

    # 启用日志文件（记录所有 LLM 交互）
    python -m ai_agent.main --log-file agent.log "搜索新闻"
"""

import argparse
import logging
import sys
from pathlib import Path

from ai_agent.config import AgentConfig
from ai_agent.core.agent import Agent
from ai_agent.llm.ollama import OllamaLLM
from ai_agent.llm.openai import OpenAILLM
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

logger = logging.getLogger("ai_agent")


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """配置日志系统

    Args:
        verbose: 设为 True 时启用 DEBUG 级别
        log_file: 日志文件路径，设置后将 LLM 交互记录到文件
    """
    handlers: list[logging.Handler] = []

    # 控制台 handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    handlers.append(console)

    # 文件 handler（记录所有 LLM 交互详情）
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handlers.append(file_handler)
        # 同时输出提示到 stderr
        sys.stderr.write(f"📝 LLM 交互日志: {log_path}\n")

    logging.basicConfig(
        level=logging.DEBUG if verbose or log_file else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def create_agent(
    model: str = "minimax-m2.5:cloud",
    host: str = "http://localhost:11434",
    provider: str = "ollama",
    api_key: str | None = None,
    openai_base_url: str | None = None,
    temperature: float = 0.7,
    verbose: bool = False,
    on_step: callable = None,
    on_token: callable = None,
    on_thinking: callable = None,
    response_log_path: str | None = None,
    enable_thinking: bool = False,
) -> Agent:
    """创建并配置 Agent"""
    config = AgentConfig(
        model=model,
        provider=provider,
        ollama_host=host,
        api_key=api_key,
        openai_base_url=openai_base_url,
        temperature=temperature,
        verbose=verbose,
    )

    if provider == "openai":
        llm = OpenAILLM(
            model=model,
            api_key=api_key,
            base_url=openai_base_url,
            temperature=temperature,
            response_log_path=response_log_path,
            enable_thinking=enable_thinking,
        )
    else:
        llm = OllamaLLM(
            model=model,
            host=host,
            temperature=temperature,
            response_log_path=response_log_path,
            enable_thinking=enable_thinking,
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
        on_thinking=on_thinking,
    )

    return agent


# 模块级流式状态，供 _print_step 跨迭代重置
_stream_state = {"started": False}


def _print_step(event: str, data: dict) -> None:
    """实时打印 Agent 思考过程"""
    if event == "start":
        _stream_state["started"] = False
        print(f"\n{'─' * 40}")
    elif event == "planning":
        _stream_state["started"] = False
        steps = data.get("steps", [])
        if steps:
            print(f"\n📋 任务规划 ({len(steps)} 步):")
            for s in steps:
                print(f"   {s['id']}. {s['description']}")
            print()
    elif event == "thinking_token":
        # 流式推理 token — 由 on_thinking 回调处理
        pass
    elif event == "token":
        pass  # 流式 token 由 on_token 处理
    elif event == "thinking":
        content = data.get("content", "")
        iteration = data.get("iteration", 0)
        tool_calls = data.get("tool_calls", [])

        if tool_calls:
            if content and not content.startswith("{"):
                # 有推理文本（流式已显示），补充工具调用说明
                if _stream_state["started"]:
                    print()
                    _stream_state["started"] = False
                print(f"  💭 [{iteration}] 决定调用: {', '.join(t['name'] for t in tool_calls)}")
            else:
                # 原生 function calling：content 为空或纯 JSON
                if _stream_state["started"]:
                    print()
                    _stream_state["started"] = False
                tool_descs = []
                for t in tool_calls:
                    args = t.get("arguments", {})
                    args_brief = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
                    tool_descs.append(f"{t['name']}({args_brief})" if args_brief else t['name'])
                print(f"  💭 [{iteration}] 分析后决定调用工具: {', '.join(tool_descs)}")
        elif content:
            if _stream_state["started"]:
                print()
                _stream_state["started"] = False
            if len(content) <= 200:
                print(f"  💭 [{iteration}] {content}")
            else:
                print(f"  💭 [{iteration}] {content[:200]}...")
    elif event == "acting":
        _stream_state["started"] = False
        tool = data.get("tool", "")
        args = data.get("arguments", {})
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        print(f"  🔧 调用: {tool}({args_str})")
    elif event == "observing":
        tool = data.get("tool", "")
        result = data.get("result", "")
        summary = result[:120].replace("\n", " ") + ("..." if len(result) > 120 else "")
        print(f"  📊 {tool} → {summary}")
    elif event == "done":
        pass  # 最终答案在下面单独打印


def _make_stream_printer():
    """创建一个带状态的流式打印回调（共享 _stream_state）"""
    def on_token(token: str) -> None:
        if not _stream_state["started"]:
            print("\n💭 ", end="", flush=True)
            _stream_state["started"] = True
        print(token, end="", flush=True)

    return on_token


def _make_thinking_printer():
    """创建一个推理（thinking）流式打印回调"""
    state = {"started": False}

    def on_thinking(token: str) -> None:
        if not state["started"]:
            print("\n🧠 ", end="", flush=True)
            state["started"] = True
        print(token, end="", flush=True)

    return on_thinking, state


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
        _stream_state["started"] = False
        agent.on_token = _make_stream_printer()
        if agent.llm.enable_thinking:
            thinking_p, thinking_s = _make_thinking_printer()
            agent.on_thinking = thinking_p

        result = agent.run(user_input)

        # 流式结束后换行
        if _stream_state["started"]:
            print()
            _stream_state["started"] = False

        print(f"\n{'─' * 40}")
        print(f"✅ Agent: {result.answer}")
        print(f"   (耗时 {result.elapsed_seconds:.1f}s, {result.iterations} 轮迭代)")
        if result.plan:
            print(f"   (计划: {result.plan.completed_steps}/{result.plan.total_steps} 步完成)")
        print()


def list_models(host: str = "http://localhost:11434", provider: str = "ollama",
                api_key: str | None = None, openai_base_url: str | None = None) -> None:
    """列出可用模型"""
    if provider == "openai":
        llm = OpenAILLM(api_key=api_key, base_url=openai_base_url)
    else:
        llm = OllamaLLM(host=host)
    models = llm.list_models()
    provider_label = f" ({provider})"
    print(f"\n可用模型 ({len(models)} 个){provider_label}:")
    for m in models:
        print(f"  • {m}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent - 基于 Ollama / OpenAI 的工具调用型智能 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 交互模式 (Ollama)
  %(prog)s --provider openai --model gpt-4o   # 使用 OpenAI
  %(prog)s "搜索今天的科技新闻"                  # 单次查询
  %(prog)s --model qwen2.5:14b "分析代码"       # 指定模型
  %(prog)s --list-models                       # 列出可用模型
  %(prog)s --list-models --provider openai     # 列出 OpenAI 模型
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
        help="模型名 (默认: minimax-m2.5:cloud)",
    )
    parser.add_argument(
        "--provider", "-p",
        default="ollama",
        choices=["ollama", "openai"],
        help="LLM 提供商 (默认: ollama, 可选: openai)",
    )
    parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama 服务地址 (默认: http://localhost:11434)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API 密钥（也可通过 OPENAI_API_KEY 环境变量设置）",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        metavar="URL",
        help="OpenAI API 基础地址（默认自动推断，也可通过 OPENAI_BASE_URL 环境变量设置）",
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
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="启用模型推理（think），支持推理的模型将在回复前输出思考过程",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        default=False,
        help="禁用模型推理",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="LLM 交互日志文件路径（记录所有请求/响应/工具调用）",
    )

    args = parser.parse_args()

    # 配置日志（必须在其他模块导入之前完成，但这里是最早的调用点）
    setup_logging(verbose=args.verbose, log_file=args.log_file)

    # 原始响应 JSONL 路径（从 --log-file 派生）
    response_log_path = None
    if args.log_file:
        log_path = Path(args.log_file)
        response_log_path = str(log_path.with_suffix(".jsonl"))
        sys.stderr.write(f"📝 原始 LLM 响应: {response_log_path}\n")

    if args.list_models:
        list_models(
            host=args.host,
            provider=args.provider,
            api_key=args.api_key,
            openai_base_url=args.openai_base_url,
        )
        return

    # 确定是否启用推理
    enable_thinking = args.think and not args.no_think

    # 创建 Agent（交互模式默认流式）
    stream_print = _make_stream_printer()
    thinking_print, thinking_state = _make_thinking_printer()
    agent = create_agent(
        model=args.model,
        host=args.host,
        provider=args.provider,
        api_key=args.api_key,
        openai_base_url=args.openai_base_url,
        temperature=args.temperature,
        verbose=args.verbose,
        on_step=_print_step,
        on_token=stream_print if not args.query else None,
        on_thinking=thinking_print if (enable_thinking and not args.query) else None,
        response_log_path=response_log_path,
        enable_thinking=enable_thinking,
    )

    if args.no_planning:
        agent.config.enable_planning = False

    if args.query:
        # 单次查询模式
        print(f"\n查询: {args.query}")
        print(f"模型: {args.model}")
        print("-" * 40)
        # 单次查询也使用流式
        _stream_state["started"] = False
        agent.on_token = _make_stream_printer()
        result = agent.run(args.query)
        if _stream_state["started"]:
            print()
            _stream_state["started"] = False
        print(f"\n{result.answer}")
        print(f"\n(耗时 {result.elapsed_seconds:.1f}s, {result.iterations} 轮)")
    else:
        # 交互模式
        interactive_mode(agent)

    # 清理
    agent.long_term.close()


if __name__ == "__main__":
    main()
