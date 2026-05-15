# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run interactive agent mode
uv run ai-agent
uv run python -m ai_agent.main

# Single query mode
uv run python -m ai_agent.main "your query"

# Use OpenAI provider instead of Ollama
uv run python -m ai_agent.main --provider openai --model gpt-4o "query"

# List available models
uv run python -m ai_agent.main --list-models

# Enable thinking tokens display (for reasoning models)
uv run python -m ai_agent.main --think --model deepseek-r1:7b

# Log LLM interactions to file (generates .log and .jsonl)
uv run python -m ai_agent.main --log-file agent.log "query"

# Run example script (interactive menu)
uv run python ai_agent/example.py
```

## Architecture

### Core Loop (ReAct)

The agent runs a **Think → Act → Observe → Reflect** loop in `core/agent.py`:

1. **Think** - LLM evaluates the conversation + tool definitions, decides whether to call tools or respond
2. **Act** - Execute tool calls (sequentially or via ThreadPoolExecutor for concurrent tools)
3. **Observe** - Tool results are injected back into conversation history
4. **Reflect** - LLM sees observations, checks if plan is done, may trigger replanning on failures

Before the loop starts, an optional **Planner** (`core/planner.py`) estimates task complexity (1-10 heuristic) and can decompose complex tasks into dependent steps.

### Module Structure

```
ai_agent/
├── __init__.py          # Public API exports
├── config.py            # AgentConfig dataclass (all knobs)
├── main.py              # CLI entry point (argparse + interactive shell)
├── example.py           # Usage demos
├── core/
│   ├── agent.py         # Main Agent: ReAct loop, context trimming, parallel tool exec, error recovery
│   ├── memory.py        # ShortTermMemory (sliding window), WorkingMemory (task state), LongTermMemory (SQLite)
│   └── planner.py       # Complexity estimation, Plan/PlanStep, LLM-based planning & replanning
├── llm/
│   ├── base.py          # BaseLLM ABC, LLMResponse, StreamEvent
│   ├── ollama.py        # Ollama integration (native API + OpenAI-compat dual mode, streaming, think support)
│   └── openai.py        # OpenAI SDK integration w/ auto base URL inference, streaming, reasoning_content support
├── tools/
│   ├── base.py          # ToolDefinition, ToolParameter, @tool decorator
│   ├── registry.py      # ToolRegistry (register, lookup, execute, schema generation)
│   └── builtin/         # Sandbox-guarded built-in tools
│       ├── file_ops.py  # read/write/list/delete file
│       ├── shell.py     # run_shell_command
│       └── web_search.py # search_web + fetch_url (DuckDuckGo + requests/bs4)
└── utils/
    ├── security.py      # Path sandbox (symlink traversal prevention + directory jail), shell command allowlist
    └── token_utils.py   # CJK-aware token estimation for context window management
```

### Key Design Decisions

- **Two LLM backends**: OllamaLLM (local, supports both native Ollama API and OpenAI-compat mode) and OpenAILLM (OpenAI SDK for any OpenAI-compatible API including DeepSeek, Moonshot, etc.). The OpenAILLM infers base URL from model name patterns.
- **Tool calling**: Supports LLM-native function calling AND regex-based parsing of JSON / Chinese function-call style text. The `ToolCallParser` in `core/agent.py` handles three fallback strategies.
- **Security layer**: Thread-local `SecurityContext` set before each tool execution. `sandbox_path()` does two-phase path validation (normalized check + realpath symlink check). `validate_shell_command()` uses an allowlist + dangerous-pattern regex blocking. All built-in tools route through `check_path()` / `check_command()`.
- **Memory**: Three-tier — ShortTermMemory (sliding window of messages with OpenAI-compat tool_calls/reasoning_content fields), WorkingMemory (in-memory key-value store + step results), LongTermMemory (SQLite with keyword search and importance ranking).
- **Context management**: CJK-character-aware token estimation (no tiktoken dependency). `_trim_messages()` first truncates tool results, then drops old messages while preserving the first user message.
- **Error recovery**: `_categorize_error()` classifies tool failures into categories (tool_not_found, timeout, permission, etc.). Failed steps trigger `Planner.replan()` up to `max_replan_attempts`.
- **Streaming**: `on_token` and `on_thinking` callbacks for real-time token display. When `on_token` is set, `_call_llm()` automatically switches to `chat_stream()`.
- **Planning**: `estimate_complexity()` uses keyword heuristics (multi-task connectors, complex operation verbs, length). When complexity >= `plan_threshold_complexity`, `Planner.create_plan()` asks LLM to decompose. Falls back to simple text splitting on LLM failure.
