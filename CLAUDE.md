# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 常用命令

```bash
# 安装依赖
uv sync

# 交互式 Agent 模式
uv run ai-agent
uv run python -m ai_agent.main

# 单次查询模式
uv run python -m ai_agent.main "你的问题"

# 指定模型
uv run python -m ai_agent.main --model gpt-4o "问题"

# 列出可用模型
uv run python -m ai_agent.main --list-models

# 启用推理 token 显示（用于推理模型如 DeepSeek-R1）
uv run python -m ai_agent.main --think --model deepseek-v4-flash

# 记录 LLM 交互日志（生成 .log 和 .jsonl 文件）
uv run python -m ai_agent.main --log-file agent.log "问题"

# 禁用自动恢复上次会话
uv run python -m ai_agent.main --no-resume

# 指定会话存储目录
uv run python -m ai_agent.main --state-dir ~/my-sessions

# 指定长期记忆数据库路径
uv run python -m ai_agent.main --memory-path ~/my-memory.db

# 运行示例脚本（交互式菜单）
uv run python ai_agent/example.py
```

## 架构

### 核心循环（ReAct）

Agent 在 `core/agent.py` 中运行 **思考(Think) → 行动(Act) → 观察(Observe) → 反思(Reflect)** 循环：

1. **思考** — LLM 评估对话历史和工具定义，决定调用工具还是直接回复
2. **行动** — 执行工具调用（顺序执行，或通过 ThreadPoolExecutor 并发执行独立工具）
3. **观察** — 工具执行结果注入回对话历史
4. **反思** — LLM 查看观察结果，检查计划是否完成，失败时触发重新规划

循环开始前，可选的 **规划器** (`core/planner.py`) 通过启发式算法评估任务复杂度（1-10），将复杂任务分解为有依赖关系的步骤。

### 模块结构

```
ai_agent/
├── __init__.py          # 公共 API 导出
├── config.py            # AgentConfig 数据类（所有配置项）
├── main.py              # CLI 入口（argparse + 交互式 Shell）
├── example.py           # 使用示例
├── core/
│   ├── agent.py         # Agent 主类：ReAct 循环、上下文裁剪、并发执行、错误恢复、会话持久化
│   ├── memory.py        # 短期记忆（滑动窗口）、工作记忆（任务状态）、长期记忆（SQLite）
│   └── planner.py       # 复杂度评估、Plan/PlanStep、LLM 规划与重新规划
├── llm/
│   ├── base.py          # BaseLLM 抽象基类、LLMResponse、StreamEvent
│   └── openai.py        # OpenAI SDK 集成（自动推断 Base URL、流式、reasoning_content 支持）
├── tools/
│   ├── base.py          # ToolDefinition、ToolParameter、@tool 装饰器
│   ├── registry.py      # ToolRegistry（注册、查询、执行、Schema 生成）
│   └── builtin/         # 受安全沙箱保护的内置工具
│       ├── file_ops.py  # 读/写/列表/删除文件
│       ├── shell.py     # 执行 Shell 命令
│       └── web_search.py # 搜索网页 + 抓取 URL（DuckDuckGo + requests/bs4）
└── utils/
    ├── security.py      # 路径沙箱（符号链接遍历防御 + 目录隔离）、Shell 命令白名单
    └── token_utils.py   # CJK 感知的 Token 估算，用于上下文窗口管理
```

### 关键设计决策

- **LLM 后端**：基于 OpenAI SDK，兼容任何 OpenAI 兼容 API（DeepSeek、Moonshot、GPT 等），根据模型名自动推断 Base URL
- **工具调用**：同时支持 LLM 原生 Function Calling 和基于正则的 JSON / 中文函数调用风格文本解析。`core/agent.py` 中的 `ToolCallParser` 包含三级回退策略
- **安全层**：线程本地的 `SecurityContext`，每次工具执行前设置。`sandbox_path()` 做两阶段路径验证（规范化检查 + realpath 符号链接检查）。`validate_shell_command()` 使用白名单 + 危险模式正则拦截。所有内置工具通过 `check_path()` / `check_command()` 路由
- **三层记忆**：短期记忆（滑动窗口消息队列，支持 OpenAI 兼容的 tool_calls/reasoning_content 字段）、工作记忆（内存键值存储 + 步骤结果）、长期记忆（SQLite，关键词搜索 + 重要度排序）
- **上下文管理**：CJK 感知的 Token 估算（不依赖 tiktoken）。`_trim_messages()` 先截断工具结果，再丢弃早期消息，同时保留第一条用户消息
- **错误恢复**：`_categorize_error()` 将工具失败分类（tool_not_found、timeout、permission 等）。失败步骤触发 `Planner.replan()`，最多重试 `max_replan_attempts` 次
- **流式输出**：`on_token` 和 `on_thinking` 回调实现实时 Token 显示。设置 `on_token` 后 `_call_llm()` 自动切换为 `chat_stream()` 流式模式
- **任务规划**：`estimate_complexity()` 使用关键词启发式（多任务连接词、复杂操作动词、长度）。复杂度 >= `plan_threshold_complexity` 时 `Planner.create_plan()` 请求 LLM 分解任务，LLM 失败时降级为简单文本拆分
- **会话持久化**：`save_state()` / `load_state()` 将对话历史、计划、记忆序列化为 JSON 文件。`auto_snapshot_interval` 控制定期自动快照。每次 `run()` 完成后自动保存 `_auto_{path_hash}_{uuid8}` 格式的会话文件，同目录复用已有文件名。`resume_last_session()` 按 cwd 筛选，优先同目录手动保存 → 同目录自动快照 → 跨目录回退。通过 `--no-resume` / `--state-dir` / `AGENT_STATE_DIR` 控制行为
- **语义搜索**：通过 LLM Provider 的 `embeddings.create()` API 为每条记忆生成向量。`semantic_search()` 使用余弦相似度排序，在交互模式中用 `/memory semsearch <关键词>` 或简写 `/memory ss <关键词>` 调用
