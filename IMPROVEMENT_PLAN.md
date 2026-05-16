# AI Agent 改进计划

生成日期: 2026-05-15
基于代码版本: cc559cc
最后更新: 2026-05-16 (测试覆盖完成)

---

## 优先级 P0 — 关键缺陷

### 1. 测试覆盖  ✅ 已完成 (2026-05-16)

- **问题**: 整个项目零测试（无单元测试、集成测试），重构风险极高
- **方案**: 使用 `pytest` 补齐测试覆盖
  - 单元测试: `token_utils.py` (CJK检测、Token估算、截断)、`security.py` (路径沙箱、命令白名单、安全上下文)、`memory.py` (三种记忆类型、序列化、嵌入)、`planner.py` (步骤、计划、规划器)、`tools/registry.py` (注册、执行、描述)、`tools/builtin/` (文件操作、Shell)
  - 测试基础设施: `conftest.py` 公共 fixture、临时目录隔离
  - 测试文件 7 个，测试用例 175 个
  - 覆盖率验证: `pytest` + 覆盖率报告

### 2. 错误恢复差异化策略

- **问题**: `_categorize_error()` 识别了 7 种错误类型（`tool_not_found`, `timeout`, `parameter_error`, `permission_error` 等），但全部走 `replan()` 统一处理，不区分策略
- **方案**: 实现差异化的错误恢复路由
  - `timeout` → 重试（指数退避）
  - `permission_error` → 尝试替代工具或权限降级
  - `tool_not_found` → 检查工具名拼写后重试
  - `execution_error` → 收集上下文后 replan
  - 设置单步重试上限（当前只有全局 `max_replan_attempts`）

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

---

## 优先级 P1 — 架构欠缺

### 5. 多 LLM 后端

- **问题**: 只有 OpenAI SDK 实现，无 Anthropic/Google 等支持
- **方案**: 在 `BaseLLM` 基础上扩展工厂模式
  - `OpenAILLM`（已有）
  - `AnthropicLLM`（Claude API）
  - `GoogleLLM`（Gemini API）
  - 自动路由: 根据模型名前缀选择后端
  - 统一 tool_use 格式转换

### 6. API Server 模式

- **问题**: Agent 只能 CLI 运行，无法嵌入其他系统
- **方案**: 添加 FastAPI / Starlette 服务
  - `POST /chat` — 单轮对话（无状态）
  - `POST /sessions` / `GET /sessions/:id/chat` — 有状态会话
  - WebSocket 流式输出
  - 健康检查端点 `/health`
  - OpenAPI 文档

### 7. Shell 安全深度防御

- **问题**: `python` 在白名单中，可执行任意代码
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
  - 涉及文件:
    - `ai_agent/config.py` — 新增 `auto_snapshot_interval`、`state_compress_threshold`
    - `ai_agent/core/agent.py` — 新增 `_STATE_SCHEMA_VERSION`、`_compress_value`/`_decompress_value`、`_compress_state_data`/`_decompress_state_data`、自动快照注入
    - `ai_agent/main.py` — 状态列表显示版本/快照标记、新增 `state prune`

---

## 优先级 P2 — 功能增强

### 9. 流式中断与进程管理

- **问题**: Shell 工具无进程管理，流式输出无中断机制
- **方案**:
  - Shell 工具返回 `pid`，新增 `kill_process(pid)`、`list_processes()` 工具
  - 支持异步执行 + 轮询结果
  - 流式输出支持 `SIGINT` 中断

### 10. 计划器增强  ✅ 已完成 (2026-05-15)

- **问题**: `estimate_complexity()` 关键词匹配脆弱，`_simple_decompose()` 按标点拆分不准确
- **方案**:
  - 引入 LLM 二次验证: 关键词评分 → LLM 确认（减少误触）
  - DAG 依赖图: 步骤间依赖关系解析，`get_executable_steps()` 返回可并行步骤
  - 支持步骤级 `dependencies` 标注
- **实现**:
  - `Planner._llm_verify_complexity()` — 边界值时用 LLM 确认是否需要规划，避免误触发
  - `Plan.get_executable_steps()` — 返回所有依赖已满足的步骤（并行执行候选）
  - `_simple_decompose()` 增强 — 自动推断顺序依赖和并行关系
  - `_parse_plan()` 增强 — 从 LLM 响应提取 `dependencies` 字段
  - `format_for_prompt()` 增强 — 显示可并行步骤

### 11. Token 用量监控

- **问题**: LLM 调用的 `usage` 从未被汇总或报告
- **方案**:
  - 在 Agent 中累计 `total_prompt_tokens` / `total_completion_tokens`
  - 逐轮打印 Token 消耗
  - 设置 Token 预算上限（达到后自动缩减上下文）
  - 记录到日志 / JSONL

### 12. 记忆自动压缩

- **问题**: 短期记忆到长期记忆无自动流转，记忆积累不衰减
- **方案**:
  - 短 → 长: 每轮对话结束后，LLM 自动提取关键信息存入长期记忆
  - 长 → 遗忘: 重要度低于阈值的记忆自动归档
  - 定期摘要: 多条低重要度记忆合并为摘要

---

## 优先级 P3 — 代码质量

| # | 问题 | 方案 |
|---|------|------|
| 13 | `tools = tools if tools else None` 应简化 | 去掉不必要的三元表达式 |
| 14 | `_force_summary()` 丢弃全部上下文 | 保留最后几轮消息作为总结依据 |
| 15 | `interactive_mode()` 中 memory 命令占 150+ 行 | 抽离到单独的函数 |
| 16 | `BaseLLM.chat()` 返回类型标注为 `dict`，实际返回 `LLMResponse` | 修正类型标注 |
| 17 | 并发执行时 `SecurityContext` 线程泄露 | 使用 `threading.local` 隔离，`try/finally` 保证清理 |
| 18 | `_build_security_context()` 每轮工具执行都重建 | 缓存上下文或惰性初始化 |
| 19 | `DEFAULT_SAFE_COMMANDS` 在 `security.py` 和 `config.py` 中有两份 | 统一到一处 |
| 20 | `search_web` 依赖非官方 DuckDuckGo HTML 端点 | 改为官方 API 或 SerpAPI/Google CSE |

---

## 实施建议

1. **P0 先做** — 测试覆盖是后续所有重构的基础。没有测试之前，不要碰 P1/P2 的架构改造
2. **P1 选做** — 多 LLM 后端的成本/收益最高，建议第二个推进；API Server 看是否需要嵌入现有系统
3. **P2/P3 穿插** — 可以每个迭代带 1-2 个小改进

---

*此文档由代码审计自动生成，每轮改进后建议同步更新。*
