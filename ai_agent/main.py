#!/usr/bin/env python3
"""
AI Agent 主入口

用法：
    # 交互模式
    python -m ai_agent.main

    # 单次查询
    python -m ai_agent.main "帮我搜索今天的新闻"

    # 指定模型
    python -m ai_agent.main --model deepseek-v4-flash "分析这个项目结构"

    # 列出可用模型
    python -m ai_agent.main --list-models

    # 启用日志文件（记录所有 LLM 交互）
    python -m ai_agent.main --log-file agent.log "搜索新闻"
"""

import argparse
import datetime
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from ai_agent.config import AgentConfig
from ai_agent.core.agent import Agent
from ai_agent.llm.openai import OpenAILLM
from ai_agent.tools.builtin import (
    delete_file,
    fetch_url,
    kill_process,
    list_directory,
    list_processes,
    poll_process,
    read_file,
    run_shell_command,
    search_web,
    write_file,
)
from ai_agent.tools.registry import ToolRegistry

logger = logging.getLogger("ai_agent")

# 终端颜色/图标控制
_use_color = True  # 默认值，在 main() 中根据 --color/--no-color 和 isatty 确定


def _icon(emoji: str, fallback: str = "") -> str:
    """根据 _use_color 返回 emoji 或纯文本替代"""
    return emoji if _use_color else fallback


# 命令 Tab 补齐定义（支持带 / 前缀和不带的两种形式）
_COMMAND_COMPLETIONS: dict[str, dict | None] = {
    "/state": {
        "save": None, "load": None, "list": None,
        "delete": None, "prune": None, "help": None,
    },
    "/memory": {
        "stats": None, "bycat": None, "list": None,
        "search": None, "semsearch": None, "ss": None,
        "cat": None, "show": None, "delete": None,
        "forget": None, "reindex": None, "clear": None,
        "add": None, "help": None,
    },
    "/save": None,
    "/tools": None,
    "/skills": None,
    "/reset": None,
    "/quit": None,
    "/exit": None,
    "/help": None,
}


from prompt_toolkit.completion import Completer, Completion, PathCompleter


class _HybridCompleter(Completer):
    """混合补齐器：/ 前缀用命令补齐，# 前缀用文件路径补齐"""

    def __init__(self, command_completer: NestedCompleter):
        self._commands = command_completer
        self._paths = PathCompleter(
            only_directories=False,
            expanduser=True,
        )

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # 查找光标所在单词起点：# 优先级最高（路径中 / 属于文件名一部分）
        hash_pos = text.rfind("#", 0, document.cursor_position)
        if hash_pos >= 0:
            word_start = hash_pos
        else:
            slash_pos = text.rfind("/", 0, document.cursor_position)
            if slash_pos >= 0:
                word_start = slash_pos
            else:
                space_pos = text.rfind(" ", 0, document.cursor_position)
                word_start = space_pos + 1 if space_pos >= 0 else 0
        current_word = text[word_start:document.cursor_position]

        # / 开头 → 命令补齐
        if current_word.startswith("/"):
            yield from self._commands.get_completions(document, complete_event)
        # # 开头 → 文件路径补齐
        elif current_word.startswith("#"):
            # 去掉 # 前缀后做路径补齐
            from prompt_toolkit.document import Document
            prefix_len = word_start + 1  # 跳过 #
            # 构造一个新的 Document，去掉 # 前缀
            new_text = text[:word_start] + text[word_start + 1:]
            new_doc = Document(
                text=new_text,
                cursor_position=document.cursor_position - 1,
            )
            yield from self._paths.get_completions(new_doc, complete_event)
        else:
            # 非特殊前缀，用命令补齐（不匹配则不补）
            yield from self._commands.get_completions(document, complete_event)


def _resolve_file_references(text: str) -> str:
    """解析输入中的 #文件路径 引用，替换为文件内容。

    支持格式: #path/to/file 或 #./relative/path 或 #/absolute/path
    路径以空格或行尾结束。多个引用各自替换。
    """
    import re

    pattern = r"(?:^|\s)#([^\s]+)"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text

    result_parts = []
    last_end = 0
    for m in matches:
        raw_path = m.group(1)
        # 去掉可能的尾部标点
        clean_path = raw_path.rstrip(",;:.!?)]}，。；：！？）】")
        expanded = os.path.expanduser(clean_path)
        if not os.path.isabs(expanded):
            expanded = os.path.abspath(expanded)
        file_path = Path(expanded)

        # 添加匹配前的文本
        result_parts.append(text[last_end:m.start()])

        if not file_path.exists():
            result_parts.append(f"\n[# 文件不存在: {clean_path}]\n")
        elif not file_path.is_file():
            result_parts.append(f"\n[# 不是文件: {clean_path}]\n")
        elif not os.access(file_path, os.R_OK):
            result_parts.append(f"\n[# 无读取权限: {clean_path}]\n")
        else:
            try:
                # 检测是否为二进制文件
                with open(file_path, "rb") as f:
                    head = f.read(8192)
                if b"\x00" in head:
                    stat = file_path.stat()
                    result_parts.append(
                        f"\n[# 二进制文件: {clean_path} "
                        f"(大小: {stat.st_size:,} 字节)]\n"
                    )
                else:
                    content = head.decode("utf-8", errors="replace")
                    # 如果文件较大，读取剩余部分
                    if len(content) >= 8192:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    # 限制注入到对话中的长度
                    max_chars = 20000
                    if len(content) > max_chars:
                        content = content[:max_chars] + (
                            f"\n...(文件过长已截断，共 {len(content):,} 字符，"
                            f"显示前 {max_chars:,} 字符)"
                        )
                    ext = file_path.suffix.lstrip(".") or "text"
                    result_parts.append(
                        f"\n--- 文件: {clean_path} ---\n"
                        f"```{ext}\n{content}\n```\n"
                    )
            except Exception as e:
                result_parts.append(f"\n[# 读取失败: {clean_path} — {e}]\n")

        last_end = m.end()

    result_parts.append(text[last_end:])
    return "".join(result_parts)


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
        sys.stderr.write(f"{_icon('📝', '[LOG]')} LLM 交互日志: {log_path}\n")

    logging.basicConfig(
        level=logging.DEBUG if verbose or log_file else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def create_agent(
    model: str | None = None,
    api_key: str | None = None,
    openai_base_url: str | None = None,
    temperature: float | None = None,
    state_dir: str | None = None,
    memory_path: str | None = None,
    max_context_tokens: int | None = None,
    max_tool_result_chars: int | None = None,
    max_tokens: int | None = None,
    tools_dir: str | None = None,
    skills_dir: str | None = None,
    verbose: bool = False,
    on_step: callable = None,
    on_token: callable = None,
    on_thinking: callable = None,
    response_log_path: str | None = None,
    enable_thinking: bool = False,
) -> Agent:
    """创建并配置 Agent"""
    model = model or os.environ.get("AGENT_MODEL", "deepseek-v4-flash")
    temperature = temperature if temperature is not None else float(os.environ.get("AGENT_TEMPERATURE", "0.7"))
    state_dir = state_dir or os.environ.get("AGENT_STATE_DIR", "~/.ai_agent/sessions")
    memory_path = memory_path or os.environ.get("AGENT_MEMORY_PATH", "~/.ai_agent/long_term_memory.db")
    tools_dir = tools_dir or os.environ.get("AGENT_TOOLS_DIR")
    skills_dir = skills_dir or os.environ.get("AGENT_SKILLS_DIR")
    if max_context_tokens is None:
        val = os.environ.get("AGENT_MAX_CONTEXT_TOKENS", "")
        max_context_tokens = int(val) if val else 65536
    if max_tool_result_chars is None:
        val = os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "")
        max_tool_result_chars = int(val) if val else 32768
    if max_tokens is None:
        val = os.environ.get("AGENT_MAX_TOKENS", "")
        max_tokens = int(val) if val else 4096
    config = AgentConfig(
        model=model,
        api_key=api_key,
        openai_base_url=openai_base_url,
        temperature=temperature,
        state_dir=state_dir,
        long_term_memory_path=memory_path,
        max_context_tokens=max_context_tokens,
        max_tool_result_chars=max_tool_result_chars,
        max_tokens=max_tokens,
        verbose=verbose,
    )
    if tools_dir:
        config.tools_dir = tools_dir
    if skills_dir:
        config.skills_dir = skills_dir

    llm = OpenAILLM(
        model=model,
        api_key=api_key,
        base_url=openai_base_url,
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
        kill_process,
        list_processes,
        poll_process,
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
        sep = "─" * 40 if _use_color else "-" * 40
        print(f"\n{sep}")
    elif event == "planning":
        _stream_state["started"] = False
        steps = data.get("steps", [])
        if steps:
            print(f"\n{_icon('📋', '[Plan]')} 任务规划 ({len(steps)} 步):")
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
                print(f"  {_icon('💭', '[Think]')} [{iteration}] 决定调用: {', '.join(t['name'] for t in tool_calls)}")
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
                print(f"  {_icon('💭', '[Think]')} [{iteration}] 分析后决定调用工具: {', '.join(tool_descs)}")
        elif content:
            if _stream_state["started"]:
                print()
                _stream_state["started"] = False
            if len(content) <= 200:
                print(f"  {_icon('💭', '[Think]')} [{iteration}] {content}")
            else:
                print(f"  {_icon('💭', '[Think]')} [{iteration}] {content[:200]}...")
    elif event == "acting":
        _stream_state["started"] = False
        tool = data.get("tool", "")
        args = data.get("arguments", {})
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        print(f"  {_icon('🔧', '[Tool]')} 调用: {tool}({args_str})")
    elif event == "observing":
        tool = data.get("tool", "")
        result = data.get("result", "")
        summary = result[:120].replace("\n", " ") + ("..." if len(result) > 120 else "")
        print(f"  {_icon('📊', '[Result]')} {tool} → {summary}")
    elif event == "done":
        pass  # 最终答案在下面单独打印
def _make_stream_printer():
    """创建一个带状态的流式打印回调（共享 _stream_state）"""
    def on_token(token: str) -> None:
        if not _stream_state["started"]:
            print(f"\n{_icon('💭', '[Think]')} ", end="", flush=True)
            _stream_state["started"] = True
        print(token, end="", flush=True)

    return on_token


def _make_thinking_printer():
    """创建一个推理（thinking）流式打印回调"""
    state = {"started": False}

    def on_thinking(token: str) -> None:
        if not state["started"]:
            print(f"\n{_icon('🧠', '[Think]')} ", end="", flush=True)
            state["started"] = True
        print(token, end="", flush=True)

    return on_thinking, state


def interactive_mode(agent: Agent) -> None:
    """交互式对话模式"""
    # 自动恢复上次会话
    resumed = None
    if agent.config.auto_resume:
        resumed = agent.resume_last_session()

    sep = "=" * 60 if _use_color else "-" * 60
    print(f"\n{sep}")
    print(f"{_icon('🤖', '[AI Agent]')} AI Agent (模型: {agent.llm.model_name})")
    print(f"{_icon('📦', '[Tools]')} 可用工具: {len(agent.tool_registry.list_tools())} 个")
    print(f"{_icon('🧠', '[Plan]')} 规划: {'启用' if agent.config.enable_planning else '关闭'}")
    if resumed:
        saved_at = resumed.get("saved_at", "")[:19]
        model = resumed.get("model", "")
        plan_info = ""
        if agent.current_plan:
            plan_info = f" | 计划: {agent.current_plan.completed_steps}/{agent.current_plan.total_steps} 步"
        print(f"{_icon('🔄', '[Resumed]')} 已恢复会话: {resumed['name']} ({saved_at}){plan_info}")
    print(f"命令: /quit 退出  /save 保存  /reset 重置  /tools 工具列表  /skills 技能列表  /state 状态管理  /memory 记忆管理")
    print(f"{sep}\n")

    # 设置 prompt_toolkit 会话
    history_path = os.path.expanduser("~/.ai_agent/history")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    session = PromptSession(history=FileHistory(history_path))
    completer = _HybridCompleter(NestedCompleter.from_nested_dict(_COMMAND_COMPLETIONS))
    prompt_style = Style.from_dict({
        "prompt": "#00aa00 bold",
    }) if _use_color else None
    prompt_msg = [("class:prompt", "👤 你: ")] if _use_color else "You: "

    while True:
        try:
            user_input = session.prompt(
                prompt_msg,
                completer=completer,
                style=prompt_style,
            ).strip()
        except KeyboardInterrupt:
            # Ctrl+C — 取消当前输入，不退出
            print(" (取消输入，Ctrl+D 退出)")
            continue
        except EOFError:
            # Ctrl+D — 退出
            print("\n再见！")
            break

        if not user_input:
            continue

        # 以 / 开头的输入，先检查是否为已知命令
        _KNOWN_COMMANDS = {
            "quit", "exit", "q", "reset", "clear", "tools", "skills",
            "save", "help", "h", "state", "memory",
        }
        if user_input.startswith("/"):
            first_word = user_input[1:].split(maxsplit=1)[0].lower()
            if first_word in _KNOWN_COMMANDS:
                user_input = user_input[1:]
                lowered = user_input.lower()
            else:
                # 不以已知命令开头（如 /tmp/path）→ 当作普通输入
                lowered = ""
                user_input = _resolve_file_references(user_input)
        else:
            # 非命令输入直接交给 Agent 处理
            lowered = ""
            # 解析 #文件引用
            user_input = _resolve_file_references(user_input)

        if lowered in ("quit", "exit", "q"):
            print("再见！")
            break

        if lowered in ("reset", "clear"):
            agent.reset()
            print(f"{_icon('✅', '[OK]')} Agent 已重置")
            continue

        if lowered in ("tools",):
            print("\n可用工具：")
            for td in agent.tool_registry.list_definitions():
                params = ", ".join(p.name for p in td.parameters)
                print(f"  {_icon('•', '*')} {td.name}({params}): {td.description}")
            print()
            continue

        if lowered in ("skills",):
            skills = agent.skill_registry.list_all()
            if not skills:
                print("\n(无已加载的技能)\n")
            else:
                print(f"\n已加载的技能 ({len(skills)}):")
                for s in skills:
                    deps = f" [依赖: {s.dependencies}]" if s.dependencies else ""
                    print(f"  {_icon('•', '*')} {s.name}: {s.description}{deps}")
                print()
            continue

        if lowered == "save" or lowered.startswith("save "):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                name = parts[1].strip()
            else:
                import hashlib
                path_hash = hashlib.sha256(os.getcwd().encode()).hexdigest()[:8]
                name = f"manual_{path_hash}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                fpath = agent.save_state(name)
                print(f"\n{_icon('✅', '[OK]')} 会话已保存: {name}\n   {fpath}\n")
            except Exception as e:
                print(f"\n{_icon('❌', '[ERROR]')} 保存失败: {e}\n")
            continue

        if lowered in ("help", "h"):
            print(f"""
可用命令:
  /help, /h             显示此帮助
  /save [名称]           保存当前会话（不填名称则自动生成）
  /tools                列出可用工具
  /skills               列出已加载技能
  /state save <name>    保存当前会话状态
  /state load <name>    加载/切换到指定状态
  /state list           列出所有已保存状态
  /state delete <name>  删除指定状态
  /state prune          清理所有自动快照
  /memory help          记忆管理（关键词搜索/语义搜索/增删查改/分类统计）
  /reset, /clear        重置 Agent 状态（清空对话）
  /quit, /exit, /q      退出
            """.strip())
            print()
            continue

        if lowered.startswith("state"):
            parts = user_input.split(maxsplit=2)
            cmd = parts[1] if len(parts) > 1 else "list"

            if cmd == "save" and len(parts) > 2:
                name = parts[2]
                try:
                    fpath = agent.save_state(name)
                    print(f"\n{_icon('✅', '[OK]')} 状态已保存: {fpath}\n")
                except Exception as e:
                    print(f"\n{_icon('❌', '[ERROR]')} 保存失败: {e}\n")

            elif cmd == "load" and len(parts) > 2:
                name = parts[2]
                success = agent.load_state(name)
                if success:
                    plan_info = ""
                    if agent.current_plan:
                        plan_info = f" (计划: {agent.current_plan.completed_steps}/{agent.current_plan.total_steps} 步)"
                    print(f"\n{_icon('✅', '[OK]')} 状态已加载: {name}{plan_info}\n")
                else:
                    print(f"\n{_icon('❌', '[ERROR]')} 未找到状态: {name}\n")

            elif cmd == "list":
                states = agent.list_states()
                if not states:
                    print("\n暂无已保存的状态。\n")
                else:
                    auto_count = sum(1 for s in states if s.get("auto_snapshot"))
                    header = f"\n已保存的状态 ({len(states)} 个"
                    if auto_count:
                        header += f"，含 {auto_count} 个自动快照"
                    header += ")"
                    print(header)
                    print("-" * 60)
                    for s in states:
                        prefix = "  [A]" if s.get("auto_snapshot") else "  [M]"
                        print(f"{prefix} {s['name']}")
                        if s.get("cwd"):
                            print(f"     cwd: {s['cwd']}")
                        if s.get("saved_at"):
                            print(f"     保存: {s['saved_at'][:19]}")
                        meta_parts = []
                        if s.get("model"):
                            meta_parts.append(s["model"])
                        if s.get("agent_state"):
                            meta_parts.append(s["agent_state"])
                        if s.get("schema_version"):
                            meta_parts.append(f"v{s['schema_version']}")
                        if meta_parts:
                            print(f"     {' | '.join(meta_parts)}")
                        print()
                print()

            elif cmd == "delete" and len(parts) > 2:
                name = parts[2]
                success = agent.delete_state(name)
                if success:
                    print(f"\n{_icon('✅', '[OK]')} 已删除状态: {name}\n")
                else:
                    print(f"\n{_icon('❌', '[ERROR]')} 未找到状态: {name}\n")

            elif cmd == "help":
                print("""
状态管理命令:
  state save <name>     保存当前会话状态
  state load <name>     加载/切换到指定状态
  state list            列出所有已保存状态
  state delete <name>   删除指定状态
  state prune           删除所有自动快照（_auto_ 前缀）
                """)

            elif cmd == "prune":
                states = agent.list_states()
                pruned = 0
                for s in states:
                    if s.get("auto_snapshot"):
                        if agent.delete_state(s["name"]):
                            pruned += 1
                print(f"\n{_icon('🧹', '[Clean]')} 已清理 {pruned} 个自动快照\n")

            else:
                print(f"\n未知子命令: {cmd}，可用命令: save, load, list, delete, help\n")
            continue

        if lowered.startswith("memory"):
            parts = user_input.split(maxsplit=2)
            cmd = parts[1] if len(parts) > 1 else "overview"

            if cmd == "stats" or cmd == "bycat":
                stats = agent.long_term.stats()
                print(f"\n长期记忆统计: 共 {stats['total']} 条")
                for cat, count in stats.get("by_category", {}).items():
                    print(f"  [{cat}] {count} 条")

                if cmd == "bycat" and stats["total"] > 0:
                    print()
                    for cat, count in stats.get("by_category", {}).items():
                        entries = agent.long_term.list_all(category=cat, limit=10)
                        print(f"  [{cat}] ({count} 条):")
                        for e in entries:
                            created = e.created_at[:10] if len(e.created_at) > 10 else e.created_at
                            print(f"    [#{e.id}] *{e.importance}  {created}")
                            short = e.content[:80].replace("\n", " ")
                            print(f"          {short}")
                        if count > 10:
                            print(f"          ... 还有 {count - 10} 条")
                        print()
                print()

            elif cmd == "list":
                limit = int(parts[2]) if len(parts) > 2 else 20
                entries = agent.long_term.list_all(limit=limit)
                if not entries:
                    print("\n暂无记忆。\n")
                else:
                    print(f"\n长期记忆 (最近 {len(entries)} 条):")
                    print("-" * 60)
                    for e in entries:
                        created = e.created_at[:19] if len(e.created_at) > 19 else e.created_at
                        print(f"  [#{e.id}] [{e.category}] *{e.importance}")
                        print(f"        {created}")
                        print(f"        {e.content[:200]}")
                        print()
                print()

            elif cmd == "search" and len(parts) > 2:
                query = parts[2]
                entries = agent.long_term.search(query=query, limit=20)
                if not entries:
                    print(f"\n未找到包含 \"{query}\" 的记忆。\n")
                else:
                    print(f"\n搜索 \"{query}\" (找到 {len(entries)} 条):")
                    print("-" * 60)
                    for e in entries:
                        created = e.created_at[:19] if len(e.created_at) > 19 else e.created_at
                        print(f"  [#{e.id}] [{e.category}] *{e.importance}")
                        print(f"        {created}")
                        print(f"        {e.content[:200]}")
                        print()
                print()

            elif cmd in ("semsearch", "ss") and len(parts) > 2:
                query = parts[2]
                results = agent.long_term.semantic_search(query=query, limit=20, min_score=0.1)
                if not results:
                    print(f"\n语义搜索未找到与 \"{query}\" 相关的记忆。\n")
                else:
                    print(f"\n语义搜索 \"{query}\" (找到 {len(results)} 条):")
                    print("-" * 60)
                    for entry, score in results:
                        created = entry.created_at[:19] if len(entry.created_at) > 19 else entry.created_at
                        print(f"  [#{entry.id}] [{entry.category}] *{entry.importance}  相似度:{score:.3f}")
                        print(f"        {created}")
                        print(f"        {entry.content[:200]}")
                        print()
                print()

            elif cmd == "cat" and len(parts) > 2:
                category = parts[2]
                entries = agent.long_term.list_all(category=category, limit=50)
                if not entries:
                    print(f"\n分类 \"{category}\" 下暂无记忆。\n")
                else:
                    print(f"\n分类 [{category}] ({len(entries)} 条):")
                    print("-" * 60)
                    for e in entries:
                        created = e.created_at[:19] if len(e.created_at) > 19 else e.created_at
                        print(f"  [#{e.id}] *{e.importance}  {created}")
                        print(f"        {e.content[:200]}")
                        print()
                print()

            elif cmd == "show" and len(parts) > 2:
                try:
                    mem_id = int(parts[2])
                    entry = agent.long_term.get(mem_id)
                    if entry:
                        print(f"\n[#{entry.id}] {entry.category}  重要度 {entry.importance}")
                        print(f"  创建: {entry.created_at}")
                        print(f"  更新: {entry.updated_at}")
                        print(f"  内容: {entry.content}")
                    else:
                        print(f"\n未找到编号为 {mem_id} 的记忆。\n")
                except ValueError:
                    print(f"\n无效的编号: {parts[2]}\n")
                print()

            elif cmd == "delete" and len(parts) > 2:
                try:
                    mem_id = int(parts[2])
                    entry = agent.long_term.get(mem_id)
                    if entry:
                        agent.long_term.delete(mem_id)
                        print(f"\n{_icon('✅', '[OK]')} 已删除记忆 #{mem_id}: {entry.content[:80]}...\n")
                    else:
                        print(f"\n未找到编号为 {mem_id} 的记忆。\n")
                except ValueError:
                    print(f"\n无效的编号: {parts[2]}\n")
                print()

            elif cmd == "forget" and len(parts) > 2:
                query = parts[2]
                count = agent.long_term.forget(query)
                if count > 0:
                    print(f"\n{_icon('✅', '[OK]')} 已遗忘 {count} 条包含 \"{query}\" 的记忆\n")
                else:
                    print(f"\n未找到包含 \"{query}\" 的记忆\n")
                print()

            elif cmd == "reindex":
                if not agent.long_term._embedding_fn:
                    print(f"\n{_icon('⚠️', '[WARN]')} 未配置嵌入模型，无法重建索引\n")
                else:
                    print(f"\n{_icon('🔄', '[Busy]')} 正在为所有记忆重新计算嵌入向量...")
                    success, failed = agent.long_term.reindex_all()
                    if failed:
                        print(f"{_icon('✅', '[OK]')} 完成: {success} 成功, {failed} 失败\n")
                    else:
                        print(f"{_icon('✅', '[OK]')} 已为 {success} 条记忆重建嵌入向量\n")

            elif cmd == "clear":
                count = agent.long_term.clear_all()
                print(f"\n{_icon('✅', '[OK]')} 已删除全部 {count} 条长期记忆\n")

            elif cmd == "add" and len(parts) > 2:
                content = parts[2]
                import_cat = parts[3] if len(parts) > 3 else "note"
                agent.long_term.add(content=content, category=import_cat)
                print(f"\n{_icon('✅', '[OK]')} 已添加记忆 [{import_cat}]: {content[:80]}...\n")
                print()

            elif cmd == "help":
                print("""
记忆管理命令:
  memory                 查看概览（统计 + 最近 10 条）
  memory stats           查看统计
  memory list [n]        列出最近 n 条
  memory search <关键词>  关键词搜索记忆
  memory semsearch <关键词>  语义搜索记忆（基于向量相似度）
  memory ss <关键词>      同上，简写
  memory cat <分类>       按分类筛选
  memory show <id>       查看单条完整内容
  memory delete <id>     删除指定编号的记忆
  memory forget <关键词>  模糊搜索并删除
  memory reindex         为所有记忆重新计算嵌入向量
  memory clear           删除全部长期记忆
  memory add <内容> [分类] 手动添加记忆
                """)

            else:
                # 默认 overview：统计 + 最近 10 条
                stats = agent.long_term.stats()
                print(f"\n长期记忆统计: 共 {stats['total']} 条")
                for cat, count in stats.get("by_category", {}).items():
                    print(f"  [{cat}] {count} 条")
                entries = agent.long_term.list_all(limit=10)
                if entries:
                    print(f"\n最近 {len(entries)} 条:")
                    print("-" * 40)
                    for e in entries:
                        print(f"  [#{e.id}] [{e.category}] *{e.importance}")
                        print(f"    {e.content[:150]}")
                        print()
                print()
            continue

        # 执行（带流式思考过程显示）
        _stream_state["started"] = False
        agent.on_token = _make_stream_printer()
        if agent.llm.enable_thinking:
            thinking_p, thinking_s = _make_thinking_printer()
            agent.on_thinking = thinking_p

        sig_handler = signal.signal(signal.SIGINT, lambda _sig, _frame: agent.interrupt())
        try:
            result = agent.run(user_input)
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        # 流式结束后换行
        if _stream_state["started"]:
            print()
            _stream_state["started"] = False

        rule = "─" * 40 if _use_color else "-" * 40
        print(f"\n{rule}")
        if agent._interrupted:
            print(f"{_icon('⚠️', '[INTERRUPTED]')} Agent: {result.answer}")
        else:
            print(f"{_icon('✅', '[OK]')} Agent: {result.answer}")
        print(f"   (耗时 {result.elapsed_seconds:.1f}s, {result.iterations} 轮迭代)")
        if result.plan:
            print(f"   (计划: {result.plan.completed_steps}/{result.plan.total_steps} 步完成)")
        print()


def list_models(api_key: str | None = None,
                openai_base_url: str | None = None) -> None:
    """列出可用模型"""
    llm = OpenAILLM(api_key=api_key, base_url=openai_base_url)
    models = llm.list_models()
    print(f"\n可用模型 ({len(models)} 个):")
    for m in models:
        print(f"  • {m}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent - 基于 OpenAI API 的工具调用型智能 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                               # 交互模式
  %(prog)s --model gpt-4o "搜索今天的科技新闻"               # 单次查询
  %(prog)s --think --model deepseek-v4-flash              # 启用推理显示
  %(prog)s --log-file agent.log "搜索新闻"                 # 记录交互日志
  %(prog)s --state-dir ~/my-sessions                       # 指定会话存储目录
  %(prog)s --no-resume                                     # 禁用自动恢复上次会话
  %(prog)s --list-models                                  # 列出可用模型
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="要执行的查询（不提供则进入交互模式）",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="模型名（可通过 AGENT_MODEL 环境变量或 .env 文件设置）",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API 密钥（也可通过 OPENAI_API_KEY 环境变量设置）",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        metavar="URL",
        help="API 基础地址（默认自动推断，也可通过 OPENAI_BASE_URL 环境变量设置）",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="生成温度（可通过 AGENT_TEMPERATURE 环境变量或 .env 文件设置）",
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
        "--tools-dir",
        default=None,
        metavar="PATH",
        help="自定义工具目录，启动时自动加载其中的 @tool 装饰的 Python 文件",
    )
    parser.add_argument(
        "--skills-dir",
        default=None,
        metavar="PATH",
        help="技能定义目录，启动时自动加载其中的 Skill.md 文件",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用启动时自动恢复上次会话状态",
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
        "--env-file",
        default=None,
        metavar="PATH",
        help="环境文件路径（默认自动查找项目根目录的 .env 文件）",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="LLM 交互日志文件路径（记录所有请求/响应/工具调用）",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        metavar="PATH",
        help="会话状态存储目录（默认 ~/.ai_agent/sessions，也可通过 AGENT_STATE_DIR 环境变量设置）",
    )
    parser.add_argument(
        "--memory-path",
        default=None,
        metavar="PATH",
        help="长期记忆数据库路径（默认 ~/.ai_agent/long_term_memory.db，也可通过 AGENT_MEMORY_PATH 环境变量设置）",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        metavar="N",
        help="最大上下文 token 数（默认 65536，也可通过 AGENT_MAX_CONTEXT_TOKENS 环境变量设置）",
    )
    parser.add_argument(
        "--max-tool-result-chars",
        type=int,
        default=None,
        metavar="N",
        help="单条工具结果最大字符数（默认 32768，也可通过 AGENT_MAX_TOOL_RESULT_CHARS 环境变量设置）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="LLM 单次输出最大 token 数（默认 4096，也可通过 AGENT_MAX_TOKENS 环境变量设置）",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        default=None,
        help="强制启用彩色/emoji 输出",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=None,
        help="禁用彩色/emoji 输出（适合 CI/管道场景）",
    )

    args = parser.parse_args()

    # 加载环境文件（指定路径 > 项目根目录 .env > 不加载）
    if args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()

    # 命令行参数的默认值从 env var / .env 中读取（滞后解析，确保 load_dotenv 已执行）
    if args.model is None:
        args.model = os.environ.get("AGENT_MODEL", "deepseek-v4-flash")
    if args.temperature is None:
        args.temperature = float(os.environ.get("AGENT_TEMPERATURE", "0.7"))
    if args.state_dir is None:
        args.state_dir = os.environ.get("AGENT_STATE_DIR", "~/.ai_agent/sessions")
    if args.memory_path is None:
        args.memory_path = os.environ.get("AGENT_MEMORY_PATH", "~/.ai_agent/long_term_memory.db")
    if args.max_context_tokens is None:
        val = os.environ.get("AGENT_MAX_CONTEXT_TOKENS", "")
        args.max_context_tokens = int(val) if val else 65536
    if args.max_tool_result_chars is None:
        val = os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", "")
        args.max_tool_result_chars = int(val) if val else 32768
    if args.max_tokens is None:
        val = os.environ.get("AGENT_MAX_TOKENS", "")
        args.max_tokens = int(val) if val else 4096

    # 配置日志（必须在其他模块导入之前完成，但这里是最早的调用点）
    setup_logging(verbose=args.verbose, log_file=args.log_file)

    # 原始响应 JSONL 路径（从 --log-file 派生）
    response_log_path = None
    if args.log_file:
        log_path = Path(args.log_file)
        response_log_path = str(log_path.with_suffix(".jsonl"))
        sys.stderr.write(f"{_icon('📝', '[LOG]')} 原始 LLM 响应: {response_log_path}\n")

    if args.list_models:
        list_models(
            api_key=args.api_key,
            openai_base_url=args.api_base_url,
        )
        return

    # 确定是否启用推理
    enable_thinking = args.think and not args.no_think

    # 创建 Agent（交互模式默认流式）
    stream_print = _make_stream_printer()
    thinking_print, thinking_state = _make_thinking_printer()
    agent = create_agent(
        model=args.model,
        api_key=args.api_key,
        openai_base_url=args.api_base_url,
        temperature=args.temperature,
        state_dir=args.state_dir,
        memory_path=args.memory_path,
        max_context_tokens=args.max_context_tokens,
        max_tool_result_chars=args.max_tool_result_chars,
        max_tokens=args.max_tokens,
        tools_dir=args.tools_dir,
        skills_dir=args.skills_dir,
        verbose=args.verbose,
        on_step=_print_step,
        on_token=stream_print if not args.query else None,
        on_thinking=thinking_print if (enable_thinking and not args.query) else None,
        response_log_path=response_log_path,
        enable_thinking=enable_thinking,
    )

    # 颜色模式: --no-color 优先, --color 其次, 默认自动检测 tty
    global _use_color
    if args.no_color:
        _use_color = False
    elif args.color:
        _use_color = True
    else:
        _use_color = sys.stdout.isatty()

    if args.no_planning:
        agent.config.enable_planning = False
    if args.no_resume:
        agent.config.auto_resume = False

    if args.query:
        # 单次查询模式
        print(f"\n查询: {args.query}")
        print(f"模型: {args.model}")
        print("-" * 40)
        # 单次查询也使用流式
        _stream_state["started"] = False
        agent.on_token = _make_stream_printer()
        signal.signal(signal.SIGINT, lambda _sig, _frame: agent.interrupt())
        try:
            result = agent.run(args.query)
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        if _stream_state["started"]:
            print()
            _stream_state["started"] = False
        if agent._interrupted:
            print(f"\n{_icon('⚠️', '[INTERRUPTED]')} {result.answer}")
        else:
            print(f"\n{result.answer}")
        print(f"\n(耗时 {result.elapsed_seconds:.1f}s, {result.iterations} 轮)")
    else:
        # 交互模式
        interactive_mode(agent)

    # 清理
    agent.long_term.close()


if __name__ == "__main__":
    main()
