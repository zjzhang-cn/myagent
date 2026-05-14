# AI Agent

基于 Ollama 本地模型的工具调用型智能 Agent，支持工具调用、任务规划与分解、三层记忆系统、流式输出与实时思考过程显示。

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
- [Ollama](https://ollama.com) 已安装并运行
- 已拉取至少一个模型（如 `ollama pull qwen2.5:7b`）

### 安装

```bash
cd ~/Documents/myagent
uv sync
```

### 使用

**交互模式**（自动显示思考过程、流式输出）：

```bash
uv run ai-agent --model qwen2.5:7b
# 或
uv run python -m ai_agent.main --model qwen2.5:7b
```

**启用模型推理显示**（需模型支持 think）：

```bash
uv run python -m ai_agent.main --think --model qwen2.5:7b
```

**记录 LLM 交互日志**：

```bash
uv run python -m ai_agent.main --log-file agent.log "搜索新闻"
# 生成 agent.log（可读日志）和 agent.jsonl（原始请求/响应）
```

交互命令：
- 输入问题直接对话（思考过程实时显示）
- `tools` — 查看可用工具列表
- `memory` — 查看长期记忆
- `reset` — 重置 Agent 状态
- `quit` — 退出

**单次查询**：

```bash
uv run python -m ai_agent.main --model qwen2.5:7b "搜索今天的科技新闻，保存到 news.txt"
```

**列出可用模型**：

```bash
uv run python -m ai_agent.main --list-models
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
    model="qwen2.5:7b",              # Ollama 模型名
    ollama_host="http://localhost:11434",
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
    max_context_tokens=8192,          # 最大上下文 token 数，超出自动裁剪
    max_tool_result_chars=3000,       # 单条工具结果最大字符数，超出截断

    # 记忆配置
    short_term_memory_size=20,        # 短期记忆窗口大小
    long_term_memory_path="~/.ai_agent/long_term_memory.db",

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
│   └── ollama.py        # Ollama 集成（原生 API + OpenAI 兼容双模式 + 流式 + think）
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
cd ~/Documents/myagent
uv run python ai_agent/example.py
# 然后选择: 1(基础), 2(自定义工具), 3(文件操作), 4(记忆系统)
```
