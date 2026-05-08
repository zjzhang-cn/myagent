# AI Agent

基于 Ollama 本地模型的工具调用型智能 Agent，支持工具调用、任务规划与分解、三层记忆系统、实时思考过程显示。

## 核心特性

- **Tool Calling** — 支持 LLM 原生 Function Calling + 多格式文本解析（JSON / 函数调用风格）
- **ReAct 循环** — Think → Act → Observe → Reflect，循环直到任务完成或达到最大迭代次数
- **任务规划** — 自动评估复杂度，将复杂任务拆解为可执行步骤（支持依赖关系）
- **三层记忆** — 短期记忆（对话上下文）+ 工作记忆（任务状态）+ 长期记忆（SQLite 持久化）
- **实时显示** — 交互模式下展示 LLM 思考过程、工具调用及结果（可编程回调）

## 架构

```
用户输入 → 复杂度评估 → [可选规划] → ReAct 循环 ──→ 最终输出
                │                        │
                ▼                        ▼
           Planner                  ToolRegistry
         (任务分解)              (工具注册与执行)
                                        │
                                ┌───────┼───────┐
                                ▼       ▼       ▼
                           文件操作  Shell命令  网络搜索

记忆系统：短期记忆(对话上下文) + 工作记忆(任务状态) + 长期记忆(SQLite持久化)
显示系统：on_step 回调 → 实时展示 规划/思考/行动/观察 全过程
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

**交互模式**（自动显示思考过程）：

```bash
uv run ai-agent --model qwen2.5:7b
# 或
uv run python -m ai_agent.main --model qwen2.5:7b
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

### 实时思考过程回调

通过 `on_step` 回调实时获取 Agent 的思考、规划和工具调用过程：

```python
from ai_agent import Agent

def on_step(event: str, data: dict) -> None:
    """事件类型: start | planning | thinking | acting | observing | done"""
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

agent = Agent(on_step=on_step)
result = agent.run("搜索最新科技新闻，保存到文件")
```

## 内置工具

| 工具 | 描述 |
|------|------|
| `read_file(path, start_line?, end_line?)` | 读取文件内容 |
| `write_file(path, content, mode?)` | 写入文件 |
| `list_directory(path, pattern?)` | 列出目录内容 |
| `delete_file(path)` | 删除文件 |
| `run_shell_command(command, working_dir?, timeout?)` | 执行安全 Shell 命令 |
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
    verbose=True,                     # 详细日志

    # 规划配置
    enable_planning=True,             # 启用任务规划
    plan_threshold_complexity=3,      # 复杂度 >= 此值时启动规划

    # 记忆配置
    short_term_memory_size=20,        # 短期记忆窗口大小
    long_term_memory_path="~/.ai_agent/long_term_memory.db",
)
agent = Agent(config=config)
```

## 项目结构

```
ai_agent/
├── __init__.py          # 包入口，导出所有公共 API
├── config.py            # AgentConfig 配置类
├── main.py              # CLI 主入口（含实时思考显示）
├── example.py           # 使用示例
├── core/
│   ├── agent.py         # 核心 Agent，ReAct 循环 + on_step 回调
│   ├── memory.py        # 三层记忆系统（ShortTerm/Working/LongTerm）
│   └── planner.py       # 任务复杂度评估与规划（依赖解析）
├── llm/
│   ├── base.py          # LLM 抽象接口
│   └── ollama.py        # Ollama 集成（原生 API + OpenAI 兼容双模式）
└── tools/
    ├── base.py          # 工具基类 + @tool 装饰器
    ├── registry.py      # 工具注册表
    └── builtin/         # 内置工具
        ├── file_ops.py
        ├── shell.py
        └── web_search.py
```

## 运行示例

```bash
cd ~/Documents/myagent
uv run python ai_agent/example.py
# 然后选择: 1(基础), 2(自定义工具), 3(文件操作), 4(记忆系统)
```
