# AI Agent

基于 Ollama 本地模型的工具调用型智能 Agent，支持工具调用、任务规划与分解、三层记忆系统。

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
```

**ReAct 循环**：Think（LLM 推理） → Act（执行工具） → Observe（获取结果） → Reflect（更新记忆），循环直到任务完成或达到最大迭代次数。

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

**交互模式**：

```bash
uv run ai-agent --model qwen2.5:7b
# 或
uv run uv run python -m ai_agent.main --model qwen2.5:7b
```

交互命令：
- 输入问题直接对话
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

```python
from ai_agent import Agent, tool, ToolRegistry

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
    model="qwen2.5:7b",         # Ollama 模型名
    ollama_host="http://localhost:11434",
    temperature=0.7,
    max_iterations=15,           # ReAct 最大轮次
    enable_planning=True,        # 启用任务规划
    plan_threshold_complexity=3, # 复杂度阈值
    short_term_memory_size=20,   # 短期记忆窗口大小
    verbose=True,                # 详细日志
)
agent = Agent(config=config)
```

## 项目结构

```
ai_agent/
├── __init__.py          # 包入口
├── config.py            # AgentConfig 配置类
├── main.py              # CLI 主入口
├── example.py           # 使用示例
├── core/
│   ├── agent.py         # 核心 Agent，ReAct 循环
│   ├── memory.py        # 三层记忆系统
│   └── planner.py       # 任务复杂度评估与规划
├── llm/
│   ├── base.py          # LLM 抽象接口
│   └── ollama.py        # Ollama 集成（原生API + OpenAI兼容）
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
