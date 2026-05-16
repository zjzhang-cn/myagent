# AI Agent

基于 OpenAI API 的工具调用型智能 Agent，支持工具调用、任务规划与分解、三层记忆系统、流式输出与实时思考过程显示。

## 核心特性

- **Tool Calling** — 支持 LLM 原生 Function Calling + 多格式文本解析（JSON / 函数调用风格）
- **ReAct 循环** — Think → Act → Observe → Reflect，循环直到任务完成或达到最大迭代次数
- **流式输出** — 交互模式下实时逐 token 展示 LLM 推理和思考过程（`on_token` / `on_thinking` 回调）
- **模型推理（think）** — 支持 `--think` 参数，展示支持推理的模型的内部思考过程
- **并发工具执行** — 同一轮迭代中的多个独立工具调用使用线程池并发执行
- **任务规划** — 自动评估复杂度，将复杂任务拆解为可执行步骤（支持依赖关系）
- **错误恢复** — 检测工具执行失败，自动分类错误并触发计划重规划（replan）
- **上下文窗口管理** — CJK 感知的 token 估算，自动裁剪消息以适应模型上下文窗口
- **安全加固** — 路径沙箱阻止目录遍历和符号链接逃逸，命令白名单拦截危险 Shell 操作
- **三层记忆** — 短期记忆（对话上下文）+ 工作记忆（任务状态）+ 长期记忆（SQLite 持久化）
- **请求日志** — `--log-file` 将 LLM 完整交互记录到日志文件，JSONL 保存原始请求/响应

## 架构

```
用户输入 → 复杂度评估 → [可选规划] → ReAct 循环 ──→ 最终输出
                │                        │
                ▼                        ▼
           Planner                  ToolRegistry
         (任务分解/重规划)         (工具注册与执行)
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                    安全沙箱         并发执行          错误检测
                (路径验证+命令白名单) (ThreadPool)   (分类+replan)

记忆系统：短期记忆(上下文窗口裁剪) + 工作记忆(任务状态) + 长期记忆(SQLite持久化)
显示系统：on_step / on_token / on_thinking 回调 → 实时展示全过程
```

## 快速开始

### 前置条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 包管理器
- OpenAI API Key 或兼容 API 的密钥（如 DeepSeek、Moonshot 等）

### 安装

```bash
cd ~/Documents/myagent
uv sync
```

### 环境配置

将 `.env.example` 复制为 `.env` 并填入 API 密钥，或通过环境变量设置：

```bash
# 方式一：使用 .env 文件（推荐）
cp .env.example .env
# 然后编辑 .env 填入 OPENAI_API_KEY

# 方式二：通过环境变量
export OPENAI_API_KEY="sk-xxx"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
```

### 首次运行

```bash
# 交互模式
uv run ai-agent --model deepseek-v4-flash

# 单次查询
uv run python -m ai_agent.main --model deepseek-v4-flash "搜索今天的科技新闻"

# 列出可用模型
uv run python -m ai_agent.main --list-models
```

## CLI 参考

### 命令行参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `query` | — | 要执行的查询（位置参数，不提供则进入交互模式） | — |
| `--model` | `-m` | 模型名称 | `AGENT_MODEL` 环境变量 或 `deepseek-v4-flash` |
| `--api-key` | — | API 密钥 | `OPENAI_API_KEY` 环境变量 |
| `--api-base-url` | — | API 基础地址 | 根据模型名自动推断，或 `OPENAI_BASE_URL` 环境变量 |
| `--temperature` | `-t` | 生成温度 (0.0–2.0) | `AGENT_TEMPERATURE` 环境变量 或 `0.7` |
| `--env-file` | — | 指定 `.env` 文件路径 | 自动查找项目根目录的 `.env` |
| `--think` | — | 启用模型推理显示（需模型支持 think） | 关闭 |
| `--no-think` | — | 强制禁用模型推理 | 关闭 |
| `--no-planning` | — | 禁用任务规划 | 关闭（默认启用规划） |
| `--no-resume` | — | 禁用启动时自动恢复上次会话 | 关闭（默认自动恢复） |
| `--verbose` | `-v` | 显示 DEBUG 级别详细日志 | 关闭 |
| `--log-file` | — | LLM 交互日志路径（生成 `.log` 和 `.jsonl` 文件） | 无 |
| `--state-dir` | — | 会话状态存储目录 | `AGENT_STATE_DIR` 环境变量 或 `~/.ai_agent/sessions` |
| `--memory-path` | — | 长期记忆数据库路径 | `AGENT_MEMORY_PATH` 环境变量 或 `~/.ai_agent/long_term_memory.db` |
| `--max-context-tokens` | — | 最大上下文 token 数 | `AGENT_MAX_CONTEXT_TOKENS` 环境变量 或 `65536` |
| `--max-tool-result-chars` | — | 单条工具结果最大字符数 | `AGENT_MAX_TOOL_RESULT_CHARS` 环境变量 或 `32768` |
| `--max-tokens` | — | LLM 单次输出最大 token 数 | `AGENT_MAX_TOKENS` 环境变量 或 `4096` |
| `--list-models` | — | 列出 API 可用模型并退出 | — |

### 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | API 密钥（必填） |
| `OPENAI_BASE_URL` | API 基础地址 |
| `AGENT_MODEL` | 默认模型名（命令行 `--model` 优先级更高） |
| `AGENT_TEMPERATURE` | 默认温度（命令行 `--temperature` 优先级更高） |
| `AGENT_STATE_DIR` | 会话状态存储目录（命令行 `--state-dir` 优先级更高） |
| `AGENT_MEMORY_PATH` | 长期记忆数据库路径（命令行 `--memory-path` 优先级更高） |
| `AGENT_MAX_CONTEXT_TOKENS` | 最大上下文 token 数（命令行 `--max-context-tokens` 优先级更高，默认 65536） |
| `AGENT_MAX_TOOL_RESULT_CHARS` | 单条工具结果最大字符数（命令行 `--max-tool-result-chars` 优先级更高，默认 32768） |
| `AGENT_MAX_TOKENS` | LLM 单次输出最大 token 数（命令行 `--max-tokens` 优先级更高，默认 4096） |

所有环境变量也可通过 `.env` 文件设置。使用 `--env-file` 可指定自定义路径。

### 交互模式命令

交互模式下，输入以 `/` 开头或不带前缀均可：

**基础命令：**

| 命令 | 说明 |
|------|------|
| `/help`, `/h` | 显示帮助信息 |
| `/save [名称]` | 保存当前会话状态（不填名称则自动生成） |
| `/tools` | 列出所有可用工具及参数说明 |
| `/reset`, `/clear` | 重置 Agent 状态（清空对话历史和计划） |
| `/quit`, `/exit`, `/q` | 退出交互模式 |

**状态管理 (`/state`)：**

| 命令 | 说明 |
|------|------|
| `/state save <name>` | 保存当前会话状态（对话历史、计划、配置） |
| `/state load <name>` | 加载/切换到指定状态 |
| `/state list` | 列出所有已保存状态（标注自动快照） |
| `/state delete <name>` | 删除指定状态 |
| `/state prune` | 清理所有自动快照（`_auto_` 前缀） |

**记忆管理 (`/memory`)：**

| 命令 | 说明 |
|------|------|
| `/memory` | 查看概览（统计 + 最近 10 条） |
| `/memory stats` | 查看分类统计 |
| `/memory list [n]` | 列出最近 n 条记忆（默认 20） |
| `/memory search <关键词>` | 关键词搜索记忆 |
| `/memory semsearch <关键词>` | 语义搜索记忆（基于向量相似度） |
| `/memory ss <关键词>` | 语义搜索简写 |
| `/memory cat <分类>` | 按分类筛选记忆 |
| `/memory show <id>` | 查看指定编号记忆的完整内容 |
| `/memory delete <id>` | 删除指定编号的记忆 |
| `/memory forget <关键词>` | 模糊搜索并删除匹配的记忆 |
| `/memory add <内容> [分类]` | 手动添加一条记忆 |
| `/memory help` | 显示记忆管理帮助 |

### 使用示例

**交互模式：**

```bash
# 基础交互（自动显示思考过程、流式输出）
uv run ai-agent --model deepseek-v4-flash

# 启用模型推理显示（需模型支持 think）
uv run python -m ai_agent.main --think --model deepseek-v4-flash

# 详细日志模式
uv run python -m ai_agent.main --model gpt-4o --verbose
```

**单次查询：**

```bash
# 基础查询
uv run python -m ai_agent.main --model deepseek-v4-flash "分析这个项目结构"

# 指定 API 地址（非 OpenAI 兼容服务）
uv run python -m ai_agent.main \
  --model deepseek-v4-flash \
  --api-base-url https://api.deepseek.com/v1 \
  "搜索今天的科技新闻，保存到 news.txt"

# 自定义温度
uv run python -m ai_agent.main --model gpt-4o -t 0.3 "翻译这段文字"
```

**日志记录：**

```bash
# 记录完整 LLM 交互日志
uv run python -m ai_agent.main --log-file agent.log "搜索新闻"
# 生成 agent.log（可读日志）和 agent.jsonl（原始请求/响应）

# 结合 verbose 查看实时调试信息
uv run python -m ai_agent.main --log-file debug.log -v "分析代码"
```

**使用自定义环境文件：**

```bash
uv run python -m ai_agent.main --env-file .env.production --list-models
```

### 会话持久化

每次 `run()` 完成后会自动保存会话状态到 `state_dir`。文件名格式为 `_auto_{path_hash}_{uuid8}.json`，其中 `path_hash` 是当前工作目录的 SHA256 前 12 位。同目录下已存在自动保存时复用文件名，避免累积。

启动交互模式时默认自动恢复上次会话（`auto_resume=True`）：
- **按工作目录筛选**：优先匹配当前 `cwd` 的会话
- **同目录下**：手动保存优先于自动快照，按 `saved_at` 取最新
- **跨目录回退**：当前目录无匹配时加载任意目录的最新会话
- 使用 `--no-resume` 跳过自动恢复

```bash
# 禁用自动恢复
uv run python -m ai_agent.main --no-resume

# 指定会话存储目录
uv run python -m ai_agent.main --state-dir ~/my-sessions
```

## 编程方式使用

### 基础用法

```python
from ai_agent import Agent, tool
from ai_agent.config import AgentConfig

# 定义自定义工具
@tool(
    name="get_weather",
    description="获取指定城市的天气信息",
    params=[
        {"name": "city", "type": "string", "description": "城市名称", "required": True},
    ],
)
def get_weather(city: str) -> str:
    return f"{city}：晴，25°C"

# 创建 Agent 并注册工具
agent = Agent()
agent.add_tool(get_weather)

# 执行任务
result = agent.run("北京今天天气怎么样？")
print(result.answer)          # Agent 的最终回复
print(result.elapsed_seconds) # 耗时
print(result.iterations)      # ReAct 循环轮数
```

### 流式输出回调

通过 `on_token` 和 `on_thinking` 回调实时获取 LLM 逐 token 输出：

```python
from ai_agent import Agent

def on_token(token: str) -> None:
    """每个输出 token 时调用"""
    print(token, end="", flush=True)

def on_thinking(thinking: str) -> None:
    """每个推理 token 时调用（需模型支持 think）"""
    print(f"[思考] {thinking}", end="", flush=True)

agent = Agent(on_token=on_token, on_thinking=on_thinking)
result = agent.run("分析这个项目的代码结构")
```

### 实时思考过程回调

通过 `on_step` 回调实时获取 Agent 的思考、规划和工具调用过程：

```python
from ai_agent import Agent

def on_step(event: str, data: dict) -> None:
    """事件类型: start | planning | thinking | thinking_token | token | acting | observing | replanning | done"""
    if event == "planning":
        steps = data.get("steps", [])
        print(f"任务规划 ({len(steps)} 步):")
        for s in steps:
            print(f"  {s['id']}. {s['description']}")
    elif event == "thinking":
        print(f"  [思考 {data['iteration']}] {data['content'][:200]}")
    elif event == "acting":
        print(f"  [调用] {data['tool']}({data['arguments']})")
    elif event == "observing":
        print(f"  [结果] {data['result'][:120]}")
    elif event == "replanning":
        print(f"  [重规划 #{data['attempt']}] 失败步骤: {data['failed_count']}")

agent = Agent(on_step=on_step)
result = agent.run("搜索最新科技新闻，保存到文件")
```

## 内置工具

| 工具 | 描述 |
|------|------|
| `read_file(path, start_line?, end_line?)` | 读取文件内容（路径受沙箱限制） |
| `write_file(path, content, mode?)` | 写入文件（路径受沙箱限制） |
| `list_directory(path, pattern?)` | 列出目录内容（路径受沙箱限制） |
| `delete_file(path)` | 删除文件（路径受沙箱限制） |
| `run_shell_command(command, working_dir?, timeout?)` | 执行 Shell 命令（命令白名单 + 路径沙箱） |
| `search_web(query, max_results?)` | DuckDuckGo 搜索 |
| `fetch_url(url, max_chars?)` | 抓取网页纯文本 |

## 配置

```python
from ai_agent import AgentConfig

config = AgentConfig(
    # LLM 配置
    model="deepseek-v4-flash",         # 模型名
    api_key="sk-xxx",                  # API 密钥（也可通过 OPENAI_API_KEY 环境变量提供）
    openai_base_url="https://api.deepseek.com/v1",  # API 地址，不提供则自动推断
    temperature=0.7,
    max_tokens=4096,

    # Agent 行为
    max_iterations=15,                # ReAct 最大轮次
    max_tool_calls_per_iteration=3,   # 每轮最多工具调用数
    max_parallel_tools=5,             # 并发工具执行最大线程数
    parallel_tool_execution=True,     # 是否启用并发工具执行
    verbose=True,                     # 详细日志

    # 规划配置
    enable_planning=True,             # 启用任务规划
    plan_threshold_complexity=3,      # 复杂度 >= 此值时启动规划
    max_replan_attempts=2,            # 失败时最多重新规划次数

    # 上下文窗口管理
    max_context_tokens=65536,         # 最大上下文 token 数，超出自动裁剪
    max_tool_result_chars=32768,      # 单条工具结果最大字符数，超出截断

    # 记忆与状态配置
    short_term_memory_size=20,        # 短期记忆窗口大小
    long_term_memory_path="~/.ai_agent/long_term_memory.db",
    state_dir="~/.ai_agent/sessions", # 会话状态文件存储目录
    auto_snapshot_interval=0,         # 每 N 轮自动快照（0=关闭）
    auto_resume=True,                 # 启动时自动恢复上次会话

    # 安全配置
    workspace_directories=["."],      # 文件操作允许的目录列表
    shell_allowed_commands={          # Shell 命令白名单（48个默认命令）
        "ls", "cat", "echo", "pwd", "find", "grep", "head", "tail",
        "wc", "sort", "uniq", "cut", "tr", "date", "whoami", "which",
        "uname", "uptime", "df", "du", "env", "ps", "pgrep",
        "mkdir", "cp", "mv", "touch", "rm", "chmod",
        "tar", "gzip", "gunzip", "zip", "unzip",
        "python", "python3", "pip", "pip3",
        "git", "node", "npm", "npx", "make",
        "awk", "sed", "xargs", "curl", "wget",
    },
    shell_allow_all_commands=False,   # 设为 True 允许所有命令（不推荐）
)
agent = Agent(config=config)
```

### 安全说明

Agent 默认启用了两层安全保护：

**路径沙箱**：所有文件操作（`read_file`、`write_file`、`list_directory`、`delete_file`）的路径经过两阶段验证：
1. 规范化检查 —— 消除 `../` 等路径遍历尝试
2. 符号链接检查 —— 阻止通过符号链接逃逸到允许目录之外

默认仅允许操作当前工作目录。通过 `workspace_directories` 添加更多允许目录。

**命令白名单**：`run_shell_command` 默认限制为 48 个安全命令（只读查询 + 常用开发工具）。危险模式（如 `rm -rf /`、`curl|bash`、`chmod 777 /` 等）通过正则匹配拦截。如需执行任意命令，设置 `shell_allow_all_commands=True`。

## 项目结构

```
ai_agent/
├── __init__.py          # 包入口，导出所有公共 API
├── config.py            # AgentConfig 配置类
├── main.py              # CLI 主入口（含流式思考显示）
├── example.py           # 使用示例
├── core/
│   ├── agent.py         # 核心 Agent，ReAct 循环 + 上下文裁剪 + 并发执行 + 错误恢复
│   ├── memory.py        # 三层记忆系统（ShortTerm/Working/LongTerm）
│   └── planner.py       # 任务复杂度评估、规划与重规划（replan）
├── llm/
│   ├── base.py          # LLM 抽象接口（StreamEvent, LLMResponse）
│   └── openai.py        # OpenAI SDK 集成（流式 + think + 多 provider 兼容）
├── tools/
│   ├── base.py          # 工具基类 + @tool 装饰器
│   ├── registry.py      # 工具注册表
│   └── builtin/         # 内置工具（受安全模块保护）
│       ├── file_ops.py
│       ├── shell.py
│       └── web_search.py
└── utils/
    ├── token_utils.py   # Token 估算与文本截断
    └── security.py      # 路径沙箱、命令白名单、安全上下文
```

## 运行示例

```bash
uv run python ai_agent/example.py
# 然后选择: 1(基础), 2(自定义工具), 3(文件操作), 4(记忆系统)
```
