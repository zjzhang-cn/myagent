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

### 12. 记忆自动压缩  ✅ 已完成 (2026-05-16)

_已通过 add() 同步生成嵌入向量 + semantic_search() 语义搜索实现基础能力。_

- **剩余**:
  - `remember()` 的相似检测目前用 `content[:50]` 关键词，应升级为 `semantic_search()`
  - 相似超过 0.85 时合并或更新而非新建
  - 低重要度记忆（<=2）定期自动归档或删除

### 13. CLI 体验改进

- **问题**: 交互模式缺少历史记录、Tab 补齐、颜色控制等基础能力
- **方案**:
  - 引入 `prompt_toolkit` 或 `readline` 实现上下键翻历史、行内编辑
  - `/state` `/memory` 子命令 Tab 自动补齐
  - `--color` / `--no-color` 参数控制终端输出样式（方便 CI/管道场景）

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

### 16. Skills 技能系统

- **问题**: Agent 只有工具调用（tool calling），没有可复用的"技能"（skills）。每次执行类似任务（如代码审查、文档生成、搜索总结）都需要用户从零描述步骤，无法像 Claude Code 的 `/review` 一样一键触发
- **参考**: 遵循 Claude Code Skills 规范设计
- **方案**:
  - **技能定义**: 每个 skill 为一个 Markdown 文件，包含 YAML frontmatter 元数据和步骤内容
    ```yaml
    ---
    name: review
    description: 审查当前分支的代码变更，给出改进建议
    metadata:
      type: code-review
    triggers:
      - /review
      - review this code
    ---
    ```
  - **存储结构**: `~/.ai_agent/skills/` 目录下按 `.md` 文件自动发现，支持子目录和命名空间（如 `code/review.md` → `code:review`）
  - **技能注册**: `SkillRegistry` 扫描技能目录 + 内置技能，构建名称 → 技能映射。支持 `@skill` 装饰器注册 Python 实现的动态技能
  - **自动触发**: 交互模式收到用户输入时，按 `triggers` 列表匹配（前缀匹配 `/name` + 语义匹配自然语言），匹配成功则自动加载技能指令
  - **交互命令**:
    - `/skills` — 列出可用技能（含描述和触发词）
    - `/skill <name> [args]` — 直接执行指定技能
    - `/skills refresh` — 重新扫描技能目录
  - **技能模板语法**（Markdown 中可用的指令）:
    - `{{param}}` — 占位符参数，执行时由用户填充或 LLM 推断
    - 步骤列表（`1.` `2.`）— ReAct 步骤模板
    - `tool:tool_name(args)` — 指定步骤调用的工具
  - **关键设计**:
    - Skills 不独立于 Agent 运行，而是作为 system prompt 的一部分注入 ReAct 循环
    - 技能触发后，其步骤模板和参数被展开到 LLM 的上下文提示中
    - 参数填充支持: 用户显式提供 → LLM 推断 → 交互式询问，三级回退
    - 内置技能随项目分发（`ai_agent/skills/` 目录），用户技能从 `~/.ai_agent/skills/` 加载
  - **配置项**: `skills_dir`（技能目录，默认 `~/.ai_agent/skills`）、`enable_skills`（默认 True）
  - **估算**: 2-3 天核心框架 + markdown 解析，后续逐步扩充内置技能

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
| 21 | 并发执行时 `SecurityContext` 线程泄露 | 使用 `threading.local` 隔离，`try/finally` 保证清理 |
| 22 | `_build_security_context()` 每轮工具执行都重建 | 缓存上下文或惰性初始化 |
| 23 | `DEFAULT_SAFE_COMMANDS` 在 `security.py` 和 `config.py` 中有两份 | 统一到一处 |
| 24 | `search_web` 依赖非官方 DuckDuckGo HTML 端点 | 改为官方 API 或 SerpAPI/Google CSE |
| 25 | 无代码格式化/类型检查 | 增加 `ruff` 配置 + pre-commit hook 自动格式化 |
| 26 | 无 CI 流程 | GitHub Actions: `pytest` + `ruff check`，PR 自动触发 |
| 27 | README 仅中文，海外用户门槛高 | 补充英文版 README 或双语对照章节 |
| 28 | `example.py` 未覆盖新功能 | 补充语义搜索、会话持久化、自动恢复的演示 |
| 29 | 无贡献指南 | 新增 `CONTRIBUTING.md`，说明开发环境、PR 流程、编码规范 |

---

## 实施建议

1. **P0 先做** — 测试覆盖 ✅。下一步推荐错误恢复差异化（改动小收益高）
2. **P1 选做** — 多 LLM 后端的成本/收益最高；API Server 看是否需要嵌入现有系统
3. **P2 按需** — Token 监控（20 行）、首次运行引导（提升首次体验）容易做；搜索 Provider 和多模态扩展取决于使用场景
4. **P3 穿插** — ruff + pre-commit + CI 可一次搞定，后续每轮自动保持代码质量。文档改进可在其他任务间隙顺手推进

---

*此文档由代码审计自动生成，每轮改进后建议同步更新。*
