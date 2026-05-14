"""
记忆系统：短期记忆、工作记忆、长期记忆

- ShortTermMemory: 维护最近 N 轮对话上下文
- WorkingMemory: 存储当前任务中的中间结果和状态
- LongTermMemory: 持久化的知识存储（文件/SQLite）
"""

import json
import logging
import os
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# 短期记忆
# ============================================================

@dataclass
class Message:
    """单条消息"""
    role: str       # "user" | "assistant" | "system" | "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转为 LLM API 格式"""
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """短期记忆：滑动窗口对话历史"""

    def __init__(self, max_size: int = 20):
        self._messages: deque[Message] = deque(maxlen=max_size)
        self.max_size = max_size
        self.system_prompt: str | None = None

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.system_prompt = prompt

    def add(self, role: str, content: str, **metadata) -> None:
        """添加一条消息"""
        self._messages.append(Message(role=role, content=content, metadata=metadata))

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self.add("tool", result, tool_name=tool_name)

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
# 工作记忆
# ============================================================

class WorkingMemory:
    """工作记忆：当前任务状态、中间结果、规划进度"""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._current_task: str = ""
        self._task_state: str = "idle"  # idle | planning | executing | done
        self._step_results: list[dict] = []

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


# ============================================================
# 长期记忆
# ============================================================

@dataclass
class MemoryEntry:
    """长期记忆条目"""
    id: int | None
    category: str       # "fact", "preference", "learning", "note"
    content: str
    created_at: str
    updated_at: str
    importance: int = 1  # 1-5, 越高越重要

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "importance": self.importance,
        }


class LongTermMemory:
    """长期记忆：持久化知识存储，默认使用 SQLite"""

    def __init__(self, db_path: str = "~/.ai_agent/long_term_memory.db"):
        self.db_path = os.path.expanduser(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
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
        self._conn.commit()

    def add(
        self,
        content: str,
        category: str = "fact",
        importance: int = 1,
    ) -> int:
        """添加一条记忆，返回 id"""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO memories (category, content, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (category, content, importance, now, now),
        )
        self._conn.commit()
        return cursor.lastrowid

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
