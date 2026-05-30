"""
记忆系统 — 三层记忆架构

三层记忆各司其职：

    ShortTermMemory  — 短期记忆：滑动窗口对话历史（最近 N 轮）
        存储用户消息、LLM 回复和工具执行结果，维持对话连贯性。
        当消息超过窗口大小时，旧消息自动丢弃。

    WorkingMemory    — 工作记忆：当前任务的运行时状态
        键值存储 + 任务状态 + 步骤执行结果。
        用于在 Agent 的不同迭代之间传递中间状态。

    LongTermMemory   — 长期记忆：持久化知识存储（SQLite）
        存储用户偏好、学习到的知识和重要对话结果。
        支持关键词搜索和语义搜索（需要 embedding_fn）。

记忆流转关系：
    短期记忆 ←→ 对话上下文（每次 LLM 调用时读取）
    工作记忆 ←→ 任务执行状态（跨迭代保持）
    长期记忆 ←→ 持久化存储（跨会话保持，启动时检索相关记忆）
"""

import json
import logging
import math
import os
import sqlite3
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ============================================================
# 消息数据结构
# ============================================================

@dataclass
class Message:
    """单条对话消息

    支持 OpenAI 和 Anthropic 两种格式的所有字段。

    Attributes:
        role: 消息角色 — "user" | "assistant" | "system" | "tool"
        content: 消息文本内容
        timestamp: 消息创建时间戳
        metadata: 扩展字段，可包含：
            - tool_calls: assistant 消息的工具调用数组（OpenAI 格式）
            - tool_call_id: tool 消息对应的工具调用 ID（OpenAI 要求）
            - reasoning_content: 推理模型的思考过程（DeepSeek-R1）
            - reasoning_signature: 推理签名（Anthropic 扩展思考要求回传）
    """
    role: str       # 消息角色："user" | "assistant" | "system" | "tool"
    content: str    # 消息文本内容
    timestamp: float = field(default_factory=time.time)  # 创建时间
    metadata: dict = field(default_factory=dict)  # 扩展字段

    def to_dict(self) -> dict:
        """转为 LLM API 格式（OpenAI/Anthropic 兼容）

        处理的关键场景：
            - assistant(tool_calls): 需要携带 tool_calls 数组
            - tool 消息: 需要 tool_call_id 与 assistant 消息匹配
            - 推理模型: reasoning_content 和 reasoning_signature 需原样传回
        """
        d: dict = {"role": self.role, "content": self.content}
        if self.role == "assistant":
            if self.metadata.get("tool_calls"):
                d["tool_calls"] = self.metadata["tool_calls"]
            # 推理模型要求 reasoning_content 原样传回
            if self.metadata.get("reasoning_content"):
                d["reasoning_content"] = self.metadata["reasoning_content"]
            # Anthropic extended thinking 要求 signature 原样传回
            if self.metadata.get("reasoning_signature"):
                d["reasoning_signature"] = self.metadata["reasoning_signature"]
        if self.role == "tool" and self.metadata.get("tool_call_id"):
            d["tool_call_id"] = self.metadata["tool_call_id"]
        return d

    def to_serializable(self) -> dict:
        """全字段序列化（含 timestamp 和完整的 metadata），用于会话持久化"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_serializable(data: dict) -> "Message":
        """从序列化数据恢复 Message 对象"""
        return Message(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class ShortTermMemory:
    """短期记忆 — 滑动窗口对话历史

    使用 collections.deque 实现固定大小的滑动窗口。
    当消息数量超过 max_size 时，最早的消息自动淘汰。

    核心方法：
        add() / add_user() / add_assistant() / add_tool_result() — 添加消息
        to_messages() — 转换为 LLM API 格式的消息列表
        estimate_tokens() — 估算当前所有消息的 token 数
        to_serializable() / from_serializable() — JSON 序列化/反序列化
    """

    def __init__(self, max_size: int = 20):
        """
        Args:
            max_size: 滑动窗口大小，默认保留最近 20 条消息
        """
        self._messages: deque[Message] = deque(maxlen=max_size)
        self.max_size = max_size
        self.system_prompt: str | None = None  # 系统提示词（始终保留，不计入窗口）

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词（会出现在 to_messages() 的最前面）"""
        self.system_prompt = prompt

    def add(self, role: str, content: str, **metadata) -> None:
        """添加一条消息（通用方法）

        Args:
            role: 消息角色
            content: 消息文本
            **metadata: 扩展字段（如 tool_calls, tool_call_id 等）
        """
        self._messages.append(Message(role=role, content=content, metadata=metadata))

    def add_user(self, content: str) -> None:
        """添加用户消息"""
        self.add("user", content)

    def add_assistant(self, content: str, tool_calls: list[dict] | None = None,
                      reasoning_content: str = "", reasoning_signature: str = "") -> None:
        """添加 assistant 消息，可附带 tool_calls、reasoning_content 和 reasoning_signature

        Args:
            content: LLM 的文本回复
            tool_calls: OpenAI 格式的工具调用数组
            reasoning_content: 推理模型的内部思考过程（DeepSeek-R1）
            reasoning_signature: 推理签名（Anthropic 扩展思考）
        """
        kwargs = {"tool_calls": tool_calls or []}
        if reasoning_content:
            kwargs["reasoning_content"] = reasoning_content
        if reasoning_signature:
            kwargs["reasoning_signature"] = reasoning_signature
        self.add("assistant", content, **kwargs)

    def add_tool_result(self, tool_name: str, result: str, tool_call_id: str = "") -> None:
        """添加工具执行结果消息（OpenAI 兼容：带 tool_call_id）"""
        self.add("tool", result, tool_name=tool_name, tool_call_id=tool_call_id)

    def get_recent(self, n: int | None = None) -> list[Message]:
        """获取最近的 n 条消息"""
        messages = list(self._messages)
        if n is not None:
            messages = messages[-n:]
        return messages

    def to_messages(self) -> list[dict]:
        """转换为 LLM API 格式的消息列表"""
        result = []
        if self.system_prompt:
            result.append({"role": "system", "content": self.system_prompt})
        for msg in self._messages:
            result.append(msg.to_dict())
        return result

    def clear(self) -> None:
        """清空记忆（保留系统提示词）"""
        self._messages.clear()

    def to_serializable(self) -> dict:
        """序列化短期记忆为可 JSON 序列化的 dict"""
        return {
            "max_size": self.max_size,
            "system_prompt": self.system_prompt,
            "messages": [msg.to_serializable() for msg in self._messages],
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "ShortTermMemory":
        """从序列化数据恢复短期记忆"""
        mem = cls(max_size=data.get("max_size", 20))
        if data.get("system_prompt"):
            mem.set_system_prompt(data["system_prompt"])
        for msg_data in data.get("messages", []):
            mem._messages.append(Message.from_serializable(msg_data))
        return mem

    def summary(self) -> str:
        """生成记忆摘要"""
        return f"短期记忆: {len(self._messages)} 条消息, 窗口大小: {self.max_size}"

    def estimate_tokens(self) -> int:
        """估算当前所有消息的总 token 数（不含 system prompt）"""
        from ai_agent.utils.token_utils import estimate_message_tokens
        return sum(estimate_message_tokens(msg.to_dict()) for msg in self._messages)

    def __len__(self) -> int:
        return len(self._messages)


# ============================================================
# 工作记忆 — 当前任务的运行时状态
# ============================================================

class WorkingMemory:
    """工作记忆：当前任务状态、中间结果和规划进度

    与短期记忆的区别：
        • 短期记忆 = 对话流（消息序列），滑动窗口淘汰
        • 工作记忆 = 任务状态（键值对），任务结束后重置

    核心功能：
        • 键值存储 — set/get/remove 管理临时变量
        • 任务状态 — current_task / task_state 追踪任务生命周期
        • 步骤结果 — 记录每个执行步骤的成功/失败和输出
    """

    def __init__(self):
        self._store: dict[str, Any] = {}      # 键值存储（临时变量）
        self._current_task: str = ""           # 当前任务描述
        self._task_state: str = "idle"         # 任务状态：idle | planning | executing | done
        self._step_results: list[dict] = []    # 步骤执行结果列表

    # --- 键值存储 ---
    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def remove(self, key: str) -> None:
        self._store.pop(key, None)

    # --- 任务状态 ---
    @property
    def current_task(self) -> str:
        return self._current_task

    @current_task.setter
    def current_task(self, task: str) -> None:
        self._current_task = task
        self._task_state = "planning"

    @property
    def task_state(self) -> str:
        return self._task_state

    @task_state.setter
    def task_state(self, state: str) -> None:
        self._task_state = state

    # --- 步骤结果 ---
    def add_step_result(self, step_name: str, result: str, success: bool = True) -> None:
        self._step_results.append({
            "step": step_name,
            "result": result,
            "success": success,
            "time": time.time(),
        })

    def get_step_results(self) -> list[dict]:
        return list(self._step_results)

    def clear_step_results(self) -> None:
        self._step_results.clear()

    # --- 上下文构建 ---
    def build_context(self) -> str:
        """构建工作记忆上下文文本"""
        parts = []
        if self._current_task:
            parts.append(f"当前任务: {self._current_task}")
            parts.append(f"任务状态: {self._task_state}")
        if self._step_results:
            parts.append("已完成步骤:")
            for sr in self._step_results[-5:]:
                status = "✓" if sr["success"] else "✗"
                parts.append(f"  {status} {sr['step']}: {sr['result'][:200]}")
        if self._store:
            parts.append("工作变量: " + json.dumps(self._store, ensure_ascii=False, default=str))

        return "\n".join(parts) if parts else "(工作记忆为空)"

    def reset(self) -> None:
        """重置工作记忆"""
        self._store.clear()
        self._current_task = ""
        self._task_state = "idle"
        self._step_results.clear()

    def to_serializable(self) -> dict:
        """序列化工作记忆为可 JSON 序列化的 dict"""
        return {
            "store": self._store,
            "current_task": self._current_task,
            "task_state": self._task_state,
            "step_results": self._step_results,
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "WorkingMemory":
        """从序列化数据恢复工作记忆"""
        wm = cls()
        wm._store = data.get("store", {})
        wm._current_task = data.get("current_task", "")
        wm._task_state = data.get("task_state", "idle")
        wm._step_results = data.get("step_results", [])
        return wm


# ============================================================
# 长期记忆 — 持久化的知识存储（SQLite + 语义搜索）
# ============================================================

@dataclass
class MemoryEntry:
    """长期记忆条目

    Attributes:
        id: 数据库主键（None 表示未保存）
        category: 记忆分类 — "fact"(事实) | "preference"(偏好) | "learning"(学习) | "note"(笔记)
        content: 记忆内容文本
        created_at / updated_at: 创建/更新时间（ISO 格式字符串）
        importance: 重要程度 1-5，越高越重要，影响搜索排序
    """
    id: int | None
    category: str       # 分类："fact", "preference", "learning", "note"
    content: str        # 记忆内容
    created_at: str     # 创建时间（ISO 格式）
    updated_at: str     # 更新时间（ISO 格式）
    importance: int = 1 # 重要程度 1-5


class LongTermMemory:
    """长期记忆 — SQLite 持久化 + 可选语义搜索

    启动时自动创建数据库和表。支持：
        • remember()  — 添加/更新记忆（自动去重）
        • recall()    — 检索相关记忆
        • search()    — 关键词搜索
        • semantic_search() — 语义搜索（需要 embedding_fn）
        • forget()    — 删除记忆

    使用方式：
        # 基础用法（仅关键词搜索）
        memory = LongTermMemory()

        # 启用语义搜索（需要 LLM 支持 embeddings）
        memory = LongTermMemory(embedding_fn=lambda text: llm.create_embedding(text))
    """

    def __init__(
        self,
        db_path: str = "~/.ai_agent/long_term_memory.db",
        embedding_fn: Callable[[str], list[float] | None] | None = None,
    ):
        """
        Args:
            db_path: SQLite 数据库路径（支持 ~ 展开）
            embedding_fn: 嵌入函数，接受文本返回 float32 向量列表。
                          提供后启用语义搜索功能。
        """
        self.db_path = os.path.expanduser(db_path)
        self._embedding_fn = embedding_fn
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库（含自动迁移）"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)

        # 主表
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'fact',
                content TEXT NOT NULL,
                importance INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(category)
        """)

        # 嵌入向量表（按行存储 float32）
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL DEFAULT 1536,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
        """)

        # 自动迁移：为旧数据库添加 embedding 表
        existing_tables = {
            r[0] for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "memory_embeddings" not in existing_tables:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL DEFAULT 1536,
                    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
            """)

        self._conn.commit()

    def add(
        self,
        content: str,
        category: str = "fact",
        importance: int = 1,
    ) -> int:
        """添加一条记忆，返回 id（会同步生成嵌入向量）"""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO memories (category, content, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (category, content, importance, now, now),
        )
        memory_id = cursor.lastrowid

        # 异步生成嵌入向量
        self._store_embedding(memory_id, content)

        self._conn.commit()
        return memory_id

    def _store_embedding(self, memory_id: int, content: str) -> None:
        """生成并存储嵌入向量（embedding_fn 未设置时跳过）"""
        if not self._embedding_fn:
            return
        try:
            vector = self._embedding_fn(content)
            if vector is None:
                return
            blob = struct.pack(f"{len(vector)}f", *vector)
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding, dim) "
                "VALUES (?, ?, ?)",
                (memory_id, blob, len(vector)),
            )
        except Exception as e:
            logger.debug(f"存储嵌入向量失败 (id={memory_id}): {e}")

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def semantic_search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """语义搜索记忆（需要 embedding_fn）

        Args:
            query: 查询文本
            limit: 最大返回条数
            min_score: 最小相似度阈值（0.0-1.0）

        Returns:
            [(MemoryEntry, similarity_score), ...]，按相似度降序排列
        """
        if not self._embedding_fn:
            logger.warning("语义搜索不可用：未设置 embedding_fn")
            return []

        query_vec = self._embedding_fn(query)
        if query_vec is None:
            return []

        # 加载所有嵌入向量
        rows = self._conn.execute(
            "SELECT memory_id, embedding, dim FROM memory_embeddings"
        ).fetchall()

        if not rows:
            return []

        # 逐一计算余弦相似度
        scored = []
        for memory_id, blob, dim in rows:
            try:
                vec = list(struct.unpack(f"{dim}f", blob))
                score = self._cosine_similarity(query_vec, vec)
                if score < min_score:
                    continue
                entry = self.get(memory_id)
                if entry:
                    scored.append((entry, score))
            except Exception as e:
                logger.debug(f"语义搜索跳过 id={memory_id}: {e}")

        # 按相似度降序排列
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def get(self, memory_id: int) -> MemoryEntry | None:
        """获取单条记忆"""
        row = self._conn.execute(
            "SELECT id, category, content, importance, created_at, updated_at "
            "FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row:
            return MemoryEntry(
                id=row[0], category=row[1], content=row[2],
                importance=row[3], created_at=row[4], updated_at=row[5],
            )
        return None

    def search(
        self,
        query: str = "",
        category: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """搜索记忆（关键词匹配）"""
        sql = "SELECT id, category, content, importance, created_at, updated_at FROM memories WHERE 1=1"
        params: list = []

        if query:
            sql += " AND content LIKE ?"
            params.append(f"%{query}%")
        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [
            MemoryEntry(
                id=r[0], category=r[1], content=r[2],
                importance=r[3], created_at=r[4], updated_at=r[5],
            )
            for r in rows
        ]

    def list_all(
        self,
        category: str | None = None,
        limit: int = 50,
    ) -> list[MemoryEntry]:
        """列出所有记忆"""
        return self.search(query="", category=category, limit=limit)

    def update(
        self,
        memory_id: int,
        content: str | None = None,
        importance: int | None = None,
        category: str | None = None,
    ) -> bool:
        """更新记忆"""
        fields = []
        params = []
        if content is not None:
            fields.append("content = ?")
            params.append(content)
        if importance is not None:
            fields.append("importance = ?")
            params.append(importance)
        if category is not None:
            fields.append("category = ?")
            params.append(category)

        if not fields:
            return False

        fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(memory_id)

        self._conn.execute(
            f"UPDATE memories SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        self._conn.commit()
        return True

    def delete(self, memory_id: int) -> bool:
        """删除记忆"""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return True

    def clear_all(self) -> int:
        """删除全部长期记忆，返回删除条数"""
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        return count

    def remember(self, content: str, importance: int = 3, category: str = "fact") -> int:
        """记住重要信息（智能去重）"""
        # 检查是否已存在相似内容
        existing = self.search(query=content[:50], limit=1)
        if existing:
            # 更新现有记忆
            self.update(existing[0].id, content=content, importance=importance, category=category)
            return existing[0].id
        return self.add(content=content, importance=importance, category=category)

    def recall(self, query: str, limit: int = 5) -> str:
        """回忆相关记忆，返回格式化文本"""
        entries = self.search(query=query, limit=limit)
        if not entries:
            return "没有找到相关记忆。"

        lines = [f"找到 {len(entries)} 条相关记忆:"]
        for e in entries:
            lines.append(f"\n[{e.category}] (重要度:{e.importance}) {e.content}")
        return "\n".join(lines)

    def reindex_all(self) -> tuple[int, int]:
        """重新为所有记忆计算嵌入向量，返回 (成功数, 失败数)"""
        if not self._embedding_fn:
            return 0, 0
        rows = self._conn.execute("SELECT id, content FROM memories").fetchall()
        success = 0
        failed = 0
        for mem_id, content in rows:
            try:
                vector = self._embedding_fn(content)
                if vector is None:
                    failed += 1
                    continue
                blob = struct.pack(f"{len(vector)}f", *vector)
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding, dim) "
                    "VALUES (?, ?, ?)",
                    (mem_id, blob, len(vector)),
                )
                success += 1
            except Exception as e:
                logger.warning(f"重新嵌入失败 (id={mem_id}): {e}")
                failed += 1
        self._conn.commit()
        return success, failed

    def forget(self, query: str) -> int:
        """遗忘匹配的记忆，返回删除条数"""
        entries = self.search(query=query, limit=100)
        count = 0
        for e in entries:
            self.delete(e.id)
            count += 1
        return count

    def stats(self) -> dict:
        """获取记忆统计"""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        cats = self._conn.execute(
            "SELECT category, COUNT(*) FROM memories GROUP BY category"
        ).fetchall()
        return {"total": total, "by_category": dict(cats)}

    def close(self) -> None:
        self._conn.close()
