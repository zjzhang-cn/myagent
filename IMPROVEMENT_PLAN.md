# AI Agent 改进计划

生成日期: 2026-05-15
基于代码版本: cc559cc
最后更新: 2026-05-31（重新扫描，发现 1 个测试回归 + 1 个代码缺陷）

---

## 优先级 P0 — 关键缺陷

### 1. 测试覆盖  ✅ 已完成 (2026-05-16) | ⚠️ 回归 (2026-05-31)

- **问题**: 整个项目零测试（无单元测试、集成测试），重构风险极高
- **方案**: 使用 `pytest` 补齐测试覆盖
  - 单元测试: `token_utils.py` (CJK检测、Token估算、截断)、`security.py` (路径沙箱、命令白名单、安全上下文)、`memory.py` (三种记忆类型、序列化、嵌入)、`planner.py` (步骤、计划、规划器)、`tools/registry.py` (注册、执行、描述)、`tools/builtin/` (文件操作、Shell)
  - 测试基础设施: `conftest.py` 公共 fixture、临时目录隔离
  - 测试文件 7 个，测试用例 175 个
  - 覆盖率验证: `pytest` + 覆盖率报告
- **回归**: 2026-05-31 扫描发现 `test_memory_entry_to_dict` 失败（174/175），`MemoryEntry` 类缺少 `to_dict()` 方法，详见 #1a

### 1a. `MemoryEntry` 缺少 `to_dict()` 方法导致测试失败 ⚠️ 新增 (2026-05-31)

- **问题**: `MemoryEntry` 类（`memory.py:333-352`）使用类属性注解风格定义字段（`id: int | None`, `category: str` 等），但没有 `@dataclass` 装饰器，也没有实现 `to_dict()` 方法。测试 `test_memory_entry_to_dict` 调用 `entry.to_dict()` 抛出 `AttributeError`
- **影响**: 测试套件 174/175 通过，1 个失败。任何依赖 `MemoryEntry.to_dict()` 的代码也会在运行时崩溃
- **方案**: 二选一
  - A) 添加 `@dataclass` 装饰器 + `to_dict()` 方法
  - B) 仅在类中添加 `to_dict()` 方法，手动序列化字段
- **位置**: `ai_agent/core/memory.py:333-352`（类定义）、`tests/test_memory.py:343-347`（失败测试）

### 2. 错误恢复差异化策略

- **问题**: `_categorize_error()` 识别了 7 种错误类型（`tool_not_found`, `timeout`, `parameter_error`, `permission_error` 等），但全部走 `replan()` 统一处理，不区分策略
- **方案**: 实现差异化的错误恢复路由
  - `timeout` → 重试（指数退避）
  - `permission_error` → 尝试替代工具或权限降级
  - `tool_not_found` → 检查工具名拼写后重试
  - `execution_error` → 收集上下文后 replan
  - 设置单步重试上限（当前只有全局 `max_replan_attempts`）

### 2a. plan_system JSON 格式说明语法错误 ⚠️ 新增 (2026-05-30)

- **问题**: `PromptsConfig.plan_system` 第 79 行 `'格式：{{"steps": [{{"id": 1,...}}]}}'` — 缺少外层闭合花括号。Python f-string 中 `{{` = 一个字面 `{`，`}}` = 一个字面 `}`。当前用 `}}` 结尾只输出了一个 `}`，导致 JSON 格式说明不完整，LLM 可能生成格式错误的 JSON
- **方案**: 修正为 `'格式：{{"steps": [{{"id": 1, "description": "...", "tool_hint": "tool_name", "dependencies": [], "skill": "技能名"}}]}}}}'`
- **位置**: `ai_agent/prompts.py:79`

### 3. 上下文裁剪的对话完整性  ✅ 已完成 (2026-05-15)

- **问题**: `_trim_messages()` 从后往前丢弃消息，可能导致 `assistant(tool_calls)` 缺少对应的 `tool` 回复，违反 OpenAI API 规范
- **方案**: 实现消息组感知的裁剪
  - 将消息序列划分为原子组: `user`, `assistant(tool_calls) + tool*`, `assistant(普通)`
  - 裁剪时仅丢弃完整原子组，确保不对 API 产生无效序列
- **实现**: 新增 `_partition_groups()` 静态方法，重写 `_trim_messages()` 以组为单位丢弃
  - `ai_agent/core/agent.py` — 2 个方法，约 100 行

### 4. 长期记忆语义搜索  ✅ 已完成 (2026-05-16)

- **问题**: LongTermMemory 仅靠 SQLite `LIKE` 关键词匹配搜索，无法关联语义相似的内容（如"简洁代码"和"干净的 style"）
- **方案**:
  - OpenAILLM.create_embedding() 调用 embeddings.create API，自动回退多个嵌入模型
  - memory_embeddings 表存储 float32 向量，add() 时同步生成
  - semantic_search() 余弦相似度排序，支持 min_score 阈值
  - /memory semsearch / /memory ss 交互命令，失败时回退到关键词匹配

### 4a. 并行工具执行泄漏安全上下文 ⚠️ 新增

- **问题**: `_execute_tools_parallel()` 在每个线程中调用 `set_security_context(ctx)` 但从未调用 `clear_security_context()`，导致线程本地安全上下文泄漏。相比之下 `_execute_tools_sequential()` 正确清理了上下文。
- **影响**: 后续操作可能使用过期的安全上下文，绕过路径沙箱或命令白名单
- **方案**: 在每个线程的 `_run_one()` 中添加 `try/finally`，在 finally 中调用 `clear_security_context()`
- **位置**: `ai_agent/core/agent.py:1926-1936`

### 4b. SQLite 连接非线程安全 ⚠️ 新增

- **问题**: `LongTermMemory` 在初始化时创建单个 `sqlite3.connect()` 连接，SQLite 连接默认非线程安全。启用并行工具执行时，并发写入会导致 `sqlite3.ProgrammingError`
- **方案**: 使用 `sqlite3.connect(db_path, check_same_thread=False)` 或使用连接池 / 每线程连接
- **位置**: `ai_agent/core/memory.py` `LongTermMemory.__init__`

### 4c. `recall` 未使用语义搜索 ⚠️ 新增

- **问题**: `LongTermMemory.recall()` 仅调用 `self.search()`（SQL LIKE 关键词匹配），尽管已实现 `semantic_search()` 余弦相似度排序。嵌入向量已存储但从未用于检索
- **方案**: `recall()` 优先使用 `semantic_search()`（当 embedding_fn 可用时），关键词搜索作为回退
- **位置**: `ai_agent/core/memory.py` `LongTermMemory.recall()`

### 4d. `remember` 去重使用前 50 字符关键词匹配 ⚠️ 新增

- **问题**: `LongTermMemory.remember()` 用 `content[:50]` 做关键词搜索来检测重复，前 50 字符相同但内容完全不同的记忆会被错误合并
- **方案**: 升级为 `semantic_search()`，相似度 > 0.85 时合并或更新而非新建
- **位置**: `ai_agent/core/memory.py` `LongTermMemory.remember()`

### 4e. `_emit` 静默吞掉回调异常 ⚠️ 新增

- **问题**: `_emit()` 用 `logger.debug()` 记录回调异常，即使在 verbose 模式下也看不到。如果 `on_step`、`on_token` 或 `on_thinking` 回调抛出异常（如 UI 回调网络问题），错误静默丢失
- **方案**: 提升到 `logger.warning()` 或 `logger.error()`，并可选地重新抛出
- **位置**: `ai_agent/core/agent.py:969-975`

### 4f. `_simple_chat` 绕过消息裁剪 ⚠️ 新增

- **问题**: Planner 使用的 `_simple_chat()` 直接调用 `self.llm.chat(messages, tools=None)`，绕过 `_trim_messages()` 和 `_validate_tool_messages()`
- **影响**: 当 Planner 的对话历史增长到超 token 限制时，会直接触发 API 错误
- **方案**: 在 `_simple_chat()` 中加入轻量级 token 估算和截断
- **位置**: `ai_agent/core/agent.py:1851-1854`

### 4g. `_parse_sdk_response` 假设 choices 非空 ⚠️ 新增

- **问题**: `openai.py` 中 `choice = response.choices[0]` 在 API 返回空 choices 数组时抛出 `IndexError`
- **方案**: 添加 `if not response.choices` 检查，返回明确错误信息
- **位置**: `ai_agent/llm/openai.py:400`

---

## 优先级 P1 — 架构欠缺

### 5. 多 LLM 后端  ✅ 已完成 (2026-05-22)

- **问题**: 只有 OpenAI SDK 实现，无 Anthropic/Google 等支持
- **方案**: 在 `BaseLLM` 基础上扩展工厂模式
  - `OpenAILLM`（已有）
  - `AnthropicLLM`（Claude API）✅
  - `GoogleLLM`（Gemini API）
  - `LLM_TYPE` 配置显式选择后端
  - 统一 tool_use 格式转换

### 5a. OpenAI o1/o3 不兼容 streaming ⚠️ 新增 (2026-05-30)

- **问题**: `_build_params()` 已处理 o1/o3 不支持 temperature 的差异，但这两个系列也不支持 `stream=True`。当用户设置 `on_token` 回调时，对 o1/o3 模型会 API 报错
- **方案**: 检测 o1/o3 模型时自动强制 `stream=False`，并发出 warning
- **位置**: `ai_agent/llm/openai.py:348-371`

### 5b. Anthropic 流式 usage 统计丢失 input_tokens ⚠️ 新增 (2026-05-30)

- **问题**: Anthropic 流式协议中，`message_start` 事件的 `input_tokens` 可能未正确填充（部分 API 版本中为 0）。实际 input_tokens 在 `message_delta` 中只携带 `output_tokens`，导致 prompt_tokens 统计可能丢失
- **方案**: 同时从 `message_start` 和 `message_delta` 收集 usage，通过 cumulative 方式合并
- **位置**: `ai_agent/llm/anthropic.py:191-197, 199-204`

### 5c. 长期记忆 embedding_fn 对 Anthropic 静默失效 ⚠️ 新增 (2026-05-30)

- **问题**: `agent.py:440` 中 `hasattr(self.llm, 'create_embedding')` — `AnthropicLLM` 没有此方法，语义搜索静默回退到关键词匹配，无任何日志提示用户
- **方案**: 为 AnthropicLLM 实现 `create_embedding()`（通过 Anthropic 原生 embedding API 或复用配置中的 OpenAI 兼容端点），或在初始化时显式 warning
- **位置**: `ai_agent/core/agent.py:440`

### 6. API Server 模式

- **问题**: Agent 只能 CLI 运行，无法嵌入其他系统
- **方案**: 添加 FastAPI / Starlette 服务
  - `POST /chat` — 单轮对话（无状态）
  - `POST /sessions` / `GET /sessions/:id/chat` — 有状态会话
  - WebSocket 流式输出
  - 健康检查端点 `/health`
  - OpenAPI 文档

### 7. Shell 安全深度防御

- **问题**: `python` 在白名单中，可执行任意代码（如 `python -c "import os; os.system('rm -rf /')"`）
- **方案**:
  - 引入 `subprocess` 参数静态分析（禁用 `-c` 参数中的危险模式）
  - 或使用 `pyjail` / `nsjail` 沙箱执行
  - 新增网络访问黑白名单（阻止内网 SSRF）
  - 文件读写配额限制（单文件最大、单会话总量）

### 8. 持久化改进  ✅ 已完成 (2026-05-15)

- **已实现**:
  - `Agent.save_state(name)` / `load_state(name)` / `list_states()` / `delete_state(name)`
  - Message、ShortTermMemory、WorkingMemory、Plan 序列化/反序列化
  - 交互模式 `/state save/load/list/delete/prune` 命令，`/state prune` 清理自动快照
  - `auto_snapshot_interval` 配置: 每 N 轮迭代自动保存 `_auto_` 前缀快照
  - `state_compress_threshold` 配置: 超长字符串自动 zlib+base85 压缩存储
  - `schema_version` 版本兼容性标记，支持向前/向后兼容

### 8a. 消息验证 O(n²) 复杂度 ⚠️ 新增

- **问题**: `_validate_tool_messages()` 的三遍扫描中，`_find_prev_assistant()` 对每个 tool 消息反向扫描整个 cleaned 列表。当对话历史包含大量 tool_call + tool 消息对（15 轮 × 3 工具 = 90 条 tool 消息），复杂度为 O(n²)
- **方案**: 第一遍扫描时构建 `tool_call_id → assistant_index` 映射表，后续查找 O(1)
- **位置**: `ai_agent/core/agent.py:1516-1521`

### 8b. 语义搜索全量加载嵌入向量 ⚠️ 新增

- **问题**: `semantic_search()` 使用 `SELECT memory_id, embedding, dim FROM memory_embeddings` 无 LIMIT，每次查询加载全表并计算所有余弦相似度。1 万条 × 1536 维 float32 = ~60MB 加载 + 1500 万次浮点运算
- **方案**: 使用 `LIMIT 500` 先缩小候选集（按更新时间/重要度排序），或用 faiss/chroma 做近似最近邻
- **位置**: `ai_agent/core/memory.py` `semantic_search()`

### 8c. 无工具调用断路器 ⚠️ 新增

- **问题**: 当某个工具持续失败（如 `search_web` 网络不可用），Agent 在重新规划中反复调用它（最多 `max_replan_attempts=2` 次 × 原始次数），浪费 Token 预算
- **方案**: 相同工具+相同参数单次会话失败 ≥ 2 次后暂停该工具，提示 LLM 使用替代方案
- **位置**: `ai_agent/core/agent.py` 工具执行部分

### 8d. Agent 类过于庞大 ⚠️ 新增

- **问题**: `core/agent.py` 约 2020 行，混合了：系统提示词构建、消息裁剪、tool 消息验证、顺序/并行工具执行、错误分类、状态序列化/压缩/快照、会话恢复、记忆集成
- **方案**: 拆分为：`MessageValidator`（消息验证）→ `StateManager`（序列化/压缩/快照/恢复）→ `AgentCore`（ReAct 循环 + 工具执行 + 错误处理），保持 Agent 作为外观类
- **位置**: `ai_agent/core/agent.py`

### 8e. 状态压缩深拷贝加倍峰值内存 ⚠️ 新增

- **问题**: `_compress_state_data()` 在递归压缩前执行 `data = copy.deepcopy(data)`，为 50+ 条消息的对话历史创建完整副本，临时加倍内存占用
- **方案**: 就地压缩，或在原地修改后再更新引用
- **位置**: `ai_agent/core/agent.py:1084-1112`

---

## 优先级 P2 — 功能增强

### 9. 流式中断与进程管理  ✅ 已完成 (2026-05-22)

- **问题**: Shell 工具无进程管理，流式输出无中断机制
- **实现**:
  - `ai_agent/tools/builtin/processes.py` — ProcessManager 单例、后台线程输出收集
  - `run_shell_command(background=True)` — Popen 异步执行，返回 PID 和轮询提示
  - 新增 `poll_process(pid)` / `list_processes()` / `kill_process(pid)` 工具
  - Agent 新增 `_interrupted` 标志 + `interrupt()` 方法
  - `reset()` 时自动终止所有追踪的后台进程

### 10. 计划器增强  ✅ 已完成 (2026-05-15)

- **实现**:
  - `Planner._llm_verify_complexity()` — 边界值时用 LLM 确认是否需要规划
  - `Plan.get_executable_steps()` — 返回所有依赖已满足的步骤（并行执行候选）
  - `_simple_decompose()` 增强 — 自动推断顺序依赖和并行关系

### 10a. replan 通知和 failed_steps_info 未使用 Prompts 模板 ⚠️ 新增 (2026-05-30)

- **问题**: `agent.py:904-909` 的 replan 通知直接用了硬编码 f-string，`_collect_failed_steps_info()` 的标题和行格式也全部硬编码。但 `PromptsConfig` 中已定义了 `replan_notification`、`failed_steps_header`、`failed_step_line`、`completed_steps_header`、`completed_step_line` 等模板字段，却从未被引用。用户自定义 PromptsConfig 不会生效
- **方案**: 改为使用 `self.prompts` 对应模板的 `.format()`
- **位置**: `ai_agent/core/agent.py:904-909, 1994-2019`

### 11. Token 用量监控

- **问题**: LLM 调用的 `usage` 从未被汇总或报告
- **方案**:
  - 在 Agent 中累计 `total_prompt_tokens` / `total_completion_tokens`
  - 逐轮打印 Token 消耗
  - 设置 Token 预算上限（达到后自动缩减上下文）
  - 记录到日志 / JSONL

### 12. 记忆自动压缩  ✅ 部分完成 (2026-05-16)

_已通过 add() 同步生成嵌入向量 + semantic_search() 语义搜索实现基础能力。_

- **剩余**:
  - `remember()` 的相似检测目前用 `content[:50]` 关键词，应升级为 `semantic_search()`
  - 相似超过 0.85 时合并或更新而非新建
  - 低重要度记忆（<=2）定期自动归档或删除

### 13. CLI 体验改进  ✅ 已完成 (2026-05-22)

- **实现**:
  - `prompt_toolkit.PromptSession` + `FileHistory` — 持久化历史，行内编辑
  - `NestedCompleter` — `/state` `/memory` 子命令 Tab 补齐
  - `_icon()` / `_use_color` — 全局 emoji/颜色开关，自动检测 isatty
  - `Ctrl+C` 取消输入（不退出），`Ctrl+D` 退出

### 13a. SIGINT 无法中断进行中的 LLM 调用 ⚠️ 新增 (2026-05-30)

- **问题**: signal handler 设置 `_interrupted=True`，但正在执行的 `self.llm.chat()` 是同步阻塞调用，不会被打断。需等到 API 返回才能退出。对大模型推理或慢速网络，中断延迟可能数分钟
- **方案**: 使用 `threading.Event` + 后台线程探测中断信号，或使用异步 HTTP 请求（httpx）支持真正的取消
- **位置**: `ai_agent/main.py:892`, `ai_agent/core/agent.py:1033-1035`

### 14. 搜索引擎 Provider 冗余

- **问题**: `search_web` 仅依赖 DuckDuckGo 非官方 HTML 解析，容易被封或格式变动，搜索不可靠
- **方案**:
  - 支持多 Provider 架构：DuckDuckGo（现有）+ 可选 Provider（SearXNG / SerpAPI / Google CSE）
  - Provider 切换通过配置项 `search_provider` 控制
  - 不可用时自动 fallback 到备用 Provider

### 15. 多模态工具扩展

- **问题**: Agent 无法处理图片内容，即使底层 LLM 支持多模态
- **方案**:
  - 新增 `read_image(path)` 工具，返回 base64 + MIME 类型用于 LLM 分析
  - 配置项 `enable_vision` 控制启用（需模型支持）
  - 图片路径同样受路径沙箱保护

### 16. Skills 技能系统  ✅ 已完成 (2026-05-19)

- **实现**:
  - `ai_agent/core/skills.py` — Skill/SkillRegistry/SkillStep、Skill.md YAML frontmatter 解析
  - 渐进式披露：元数据注入系统提示词 → LLM 调用 `use_skill` 工具加载完整内容
  - `skills_dir` 配置项（支持 `:` 分隔多路径）、`--skills-dir` CLI 参数
  - 示例技能：code-explainer、research-save、info-fetcher

### 17. 首次运行引导

- **问题**: 未配置 API key 时直接 `Missing credentials` 报错退出，体验突兀
- **方案**:
  - 检测到未配置时输出交互式引导菜单
  - `1) 输入密钥  2) 查看配置说明  3) 退出` 选择
  - 引导用户创建 `.env` 文件并写入密钥

---

## 优先级 P3 — 代码质量

| # | 问题 | 方案 |
|---|------|------|
| 17 | `tools = tools if tools else None` 应简化 | 去掉不必要的三元表达式 |
| 18 | `_force_summary()` 丢弃全部上下文 | 保留最后几轮消息作为总结依据 |
| 19 | `interactive_mode()` 中 memory 命令占 150+ 行 | 抽离到单独的函数或文件 |
| 20 | `BaseLLM.chat()` 返回类型标注为 `dict`，实际返回 `LLMResponse` | 修正类型标注 |
| 21 | 并发执行时 `SecurityContext` 线程泄露 | 使用 `threading.local` 隔离，`try/finally` 保证清理（同 4a） |
| 22 | `_build_security_context()` 每轮工具执行都重建 | 缓存上下文或惰性初始化 |
| 23 | `DEFAULT_SAFE_COMMANDS` 在 `security.py` 和 `config.py` 中有两份且不同步 | 统一到 security.py 作为权威来源，config.py 从 security 导入 |
| 24 | `search_web` 依赖非官方 DuckDuckGo HTML 端点 | 改为官方 API 或 SerpAPI/Google CSE（同 14） |
| 25 | 无代码格式化/类型检查 | 增加 `ruff` 配置 + pre-commit hook 自动格式化 |
| 26 | 无 CI 流程 | GitHub Actions: `pytest` + `ruff check`，PR 自动触发 |
| 27 | README 仅中文，海外用户门槛高 | 补充英文版 README 或双语对照章节 |
| 28 | `example.py` 未覆盖新功能 | 补充语义搜索、会话持久化、自动恢复、技能系统、Anthropic 后端、用户确认的演示 |
| 29 | 无贡献指南 | 新增 `CONTRIBUTING.md`，说明开发环境、PR 流程、编码规范 |
| 30 | `load_tools_from_directory` 永久修改 `sys.path` | 改为临时插入 + finally 清理，或使用 `importlib` 完全限定加载 |
| 31 | 路径沙箱写保护白名单覆盖不全 | 补充 `.aws`、`.kube`、`.docker`、`.vault`、`.npmrc` 等凭据目录到 `FORBIDDEN_DIRS` |
| 32 | Web 工具 `import` 在函数体内 | `search_web`/`fetch_url` 中的 `import requests`/`from bs4 import BeautifulSoup` 移到模块顶层，减少每次调用 100-200ms 导入开销 |
| 33 | `_replan_count` 与 `max_iterations` 独立约束 | 重新规划后迭代计数不重置，可能导致新计划在剩余迭代中无法完成；考虑重新规划时重置迭代计数的策略 |

### P3 新增 (2026-05-30)

| # | 问题 | 方案 |
|---|------|------|
| 34 | `LLM 重试参数硬编码` — `_call_llm()` 和 `_call_llm_stream()` 中 `max_retries=3`、`sleep(1)`、`sleep(2)` 硬编码 | 提取为 `AgentConfig` 配置项 `llm_max_retries: int = 3`、`llm_retry_delay_base: float = 1.0` |
| 35 | `create_agent()` 中 `on_step: callable` 类型标注不规范 | 改为 `from collections.abc import Callable` + `on_step: Callable | None = None` |
| 36 | `short_term_memory_size: int = 20` 命名误导 — 实际是消息数而非对话轮数，20 条消息约覆盖 3-5 轮 | 改名为 `short_term_max_messages` 或改为轮数计数并翻倍默认值 |
| 37 | `AnthropicLLM` 默认 model 值已过时 — `"claude-sonnet-4-20250514"` | 更新为最新的默认模型名（如 `"claude-sonnet-4-6-20251101"`） |
| 38 | `PromptsConfig.plan_system` JSON 示例缺失外层闭合花括号 | 修正 Python f-string 中的花括号转义（同 2a） |

---

## 实施建议

### 第一轮：快速修复（1-2 天）
```
2a   plan_system JSON 格式修复          4a   并发安全上下文泄露
4b   SQLite check_same_thread           4e   _emit 异常提升到 warning
4g   choices 空数组检查                  5c   Anthropic embedding 警告
10a  replan/failed_steps 用 prompts     35   类型标注修正
37   更新 Anthropic 默认 model
```

### 第二轮：中等改动（3-5 天）
```
4c   recall 语义搜索                    4d   remember 语义去重
4f   _simple_chat 消息裁剪              8a   消息验证 O(n²) → O(n)
8c   工具断路器                          2    错误恢复差异化策略
5a   o1/o3 streaming 兼容               5b   Anthropic 流式 usage 修复
13a  SIGINT 中断改进                     23   命令白名单统一
34   重试参数可配置
```

### 第三轮：大型重构（5-10 天）
```
8d   Agent 类拆分                       8b   语义搜索性能优化
8e   状态压缩内存优化                    7    Shell 深度防御
11   Token 用量监控                      17   首次运行引导
14   搜索引擎 Provider 冗余              15   多模态工具扩展
6    API Server 模式
```

### 持续改进
```
30   sys.path 临时插入                   32   Web import 移到顶层
25   ruff + pyright 配置                 26   GitHub Actions CI
28   example.py 更新                     29   CONTRIBUTING.md
27   英文 README                         31   路径沙箱凭据目录
33   replan 迭代计数策略                 36   配置项命名修正
```

---

*此文档由代码审计自动生成，每轮改进后建议同步更新。*
