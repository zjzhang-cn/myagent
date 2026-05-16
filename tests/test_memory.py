"""Tests for ai_agent.core.memory."""

import os
import tempfile
import time

import pytest

from ai_agent.core.memory import (
    ShortTermMemory,
    WorkingMemory,
    LongTermMemory,
    MemoryEntry,
    Message,
)


# ===== Message =====

class TestMessage:
    def test_basic_message(self):
        msg = Message(role="user", content="hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content="",
            metadata={"tool_calls": [{"name": "read_file"}]},
        )
        d = msg.to_dict()
        assert "tool_calls" in d

    def test_message_with_reasoning(self):
        msg = Message(
            role="assistant",
            content="think step",
            metadata={"reasoning_content": "internal thoughts"},
        )
        d = msg.to_dict()
        assert d["reasoning_content"] == "internal thoughts"

    def test_tool_message_with_call_id(self):
        msg = Message(
            role="tool",
            content="result",
            metadata={"tool_call_id": "call_123"},
        )
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_123"

    def test_serialization_roundtrip(self):
        original = Message(
            role="assistant",
            content="hello",
            metadata={"tool_calls": [{"name": "test"}], "extra": "data"},
        )
        data = original.to_serializable()
        restored = Message.from_serializable(data)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata

    def test_no_reasoning_when_empty(self):
        msg = Message(role="assistant", content="hello", metadata={})
        d = msg.to_dict()
        assert "reasoning_content" not in d


# ===== ShortTermMemory =====

class TestShortTermMemory:
    def test_add_and_get(self, short_term_memory):
        short_term_memory.add_user("hello")
        short_term_memory.add_assistant("hi")
        messages = short_term_memory.get_recent()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_system_prompt_in_to_messages(self, short_term_memory):
        messages = short_term_memory.to_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "测试提示词"

    def test_max_size(self):
        mem = ShortTermMemory(max_size=3)
        for i in range(5):
            mem.add_user(f"msg {i}")
        assert len(mem) == 3

    def test_clear(self, short_term_memory):
        short_term_memory.add_user("hello")
        short_term_memory.clear()
        assert len(short_term_memory) == 0
        # System prompt should be preserved
        assert short_term_memory.system_prompt == "测试提示词"

    def test_get_recent_with_n(self, short_term_memory):
        for i in range(5):
            short_term_memory.add_user(f"msg {i}")
        recent = short_term_memory.get_recent(2)
        assert len(recent) == 2
        assert recent[0].content == "msg 3"
        assert recent[1].content == "msg 4"

    def test_serialization_roundtrip(self, short_term_memory):
        short_term_memory.add_user("hello")
        short_term_memory.add_assistant("world", tool_calls=[{"name": "test"}])
        short_term_memory.add_tool_result("test", "result", tool_call_id="tc1")

        data = short_term_memory.to_serializable()
        restored = ShortTermMemory.from_serializable(data)

        assert restored.max_size == short_term_memory.max_size
        assert restored.system_prompt == short_term_memory.system_prompt
        assert len(restored) == 3

        msgs = restored.get_recent()
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[2].role == "tool"

    def test_summary(self, short_term_memory):
        summary = short_term_memory.summary()
        assert "短期记忆" in summary
        assert "5" in summary  # max_size

    def test_add_tool_result_with_id(self, short_term_memory):
        short_term_memory.add_tool_result("search", "results", tool_call_id="tc_1")
        msg = short_term_memory.get_recent(1)[0]
        assert msg.role == "tool"
        assert msg.metadata.get("tool_call_id") == "tc_1"

    def test_empty_memory_summary(self):
        mem = ShortTermMemory(max_size=10)
        s = mem.summary()
        assert "0 条消息" in s

    def test_estimate_tokens(self, short_term_memory):
        short_term_memory.add_user("hello world")
        tokens = short_term_memory.estimate_tokens()
        assert tokens >= 1


# ===== WorkingMemory =====

class TestWorkingMemory:
    def test_set_and_get(self, working_memory):
        working_memory.set("key", "value")
        assert working_memory.get("key") == "value"

    def test_get_default(self, working_memory):
        assert working_memory.get("nonexistent", "default") == "default"

    def test_remove(self, working_memory):
        working_memory.set("key", "value")
        working_memory.remove("key")
        assert working_memory.get("key") is None

    def test_task_state_flow(self, working_memory):
        working_memory.current_task = "do something"
        assert working_memory.current_task == "do something"
        assert working_memory.task_state == "planning"
        working_memory.task_state = "executing"
        assert working_memory.task_state == "executing"

    def test_step_results(self, working_memory):
        working_memory.add_step_result("step 1", "result 1", success=True)
        working_memory.add_step_result("step 2", "result 2", success=False)
        results = working_memory.get_step_results()
        assert len(results) == 2
        assert results[0]["step"] == "step 1"
        assert results[0]["success"]
        assert not results[1]["success"]

    def test_clear_step_results(self, working_memory):
        working_memory.add_step_result("step 1", "result 1")
        working_memory.clear_step_results()
        assert len(working_memory.get_step_results()) == 0

    def test_build_context(self, working_memory):
        working_memory.current_task = "test task"
        working_memory.add_step_result("step 1", "done", success=True)
        context = working_memory.build_context()
        assert "test task" in context
        assert "step 1" in context
        assert "✓" in context

    def test_build_context_empty(self, working_memory):
        context = working_memory.build_context()
        assert "空" in context

    def test_reset(self, working_memory):
        working_memory.current_task = "some task"
        working_memory.set("key", "value")
        working_memory.add_step_result("step 1", "result")
        working_memory.reset()
        assert working_memory.current_task == ""
        assert working_memory.task_state == "idle"
        assert working_memory.get("key") is None
        assert len(working_memory.get_step_results()) == 0

    def test_serialization_roundtrip(self, working_memory):
        working_memory.current_task = "task"
        working_memory.set("key", "value")
        working_memory.add_step_result("step 1", "result")

        data = working_memory.to_serializable()
        restored = WorkingMemory.from_serializable(data)

        assert restored.current_task == "task"
        assert restored.get("key") == "value"
        assert len(restored.get_step_results()) == 1

    def test_build_context_with_variables(self, working_memory):
        working_memory.set("var1", "test_value")
        context = working_memory.build_context()
        assert "var1" in context
        assert "test_value" in context


# ===== LongTermMemory =====

class TestLongTermMemory:
    def test_add_and_get(self, long_term_memory):
        mid = long_term_memory.add("test content", category="fact", importance=3)
        entry = long_term_memory.get(mid)
        assert entry is not None
        assert entry.content == "test content"
        assert entry.category == "fact"
        assert entry.importance == 3

    def test_get_nonexistent(self, long_term_memory):
        entry = long_term_memory.get(9999)
        assert entry is None

    def test_keyword_search(self, long_term_memory):
        long_term_memory.add("人工智能技术发展", category="fact")
        long_term_memory.add("Python 编程语言", category="learning")
        long_term_memory.add("今天天气很好", category="note")

        results = long_term_memory.search(query="天气", limit=10)
        assert len(results) == 1
        assert "天气" in results[0].content

        results = long_term_memory.search(query="编程", limit=10)
        assert len(results) == 1
        assert "编程" in results[0].content

    def test_search_no_results(self, long_term_memory):
        long_term_memory.add("test content")
        results = long_term_memory.search(query="nonexistent")
        assert len(results) == 0

    def test_search_by_category(self, long_term_memory):
        long_term_memory.add("技术内容", category="tech")
        long_term_memory.add("生活内容", category="life")

        results = long_term_memory.search(category="tech", limit=10)
        assert len(results) == 1
        assert results[0].category == "tech"

    def test_list_all(self, long_term_memory):
        for i in range(5):
            long_term_memory.add(f"content {i}")
        entries = long_term_memory.list_all(limit=10)
        assert len(entries) == 5

    def test_list_all_with_category(self, long_term_memory):
        long_term_memory.add("tech", category="tech")
        long_term_memory.add("life", category="life")
        entries = long_term_memory.list_all(category="tech", limit=10)
        assert len(entries) == 1

    def test_update_content(self, long_term_memory):
        mid = long_term_memory.add("original content")
        updated = long_term_memory.update(mid, content="updated content")
        assert updated
        entry = long_term_memory.get(mid)
        assert entry.content == "updated content"

    def test_update_importance(self, long_term_memory):
        mid = long_term_memory.add("content", importance=1)
        long_term_memory.update(mid, importance=5)
        entry = long_term_memory.get(mid)
        assert entry.importance == 5

    def test_update_nonexistent(self, long_term_memory):
        # Updating a nonexistent ID returns True but affects 0 rows
        updated = long_term_memory.update(9999, content="new")
        assert updated  # Currently always returns True

    def test_delete(self, long_term_memory):
        mid = long_term_memory.add("content to delete")
        assert long_term_memory.get(mid) is not None
        long_term_memory.delete(mid)
        assert long_term_memory.get(mid) is None

    def test_remember_new(self, long_term_memory):
        mid = long_term_memory.remember("new important fact", importance=4, category="fact")
        entry = long_term_memory.get(mid)
        assert entry.importance == 4

    def test_remember_duplicate_updates(self, long_term_memory):
        mid1 = long_term_memory.remember("same content", importance=2)
        mid2 = long_term_memory.remember("same content", importance=5)
        # Should update existing, not create new
        assert mid1 == mid2
        entry = long_term_memory.get(mid1)
        assert entry.importance == 5

    def test_recall_with_results(self, long_term_memory):
        long_term_memory.add("important fact about AI", importance=3)
        text = long_term_memory.recall("AI", limit=5)
        assert "AI" in text
        assert "important" in text

    def test_recall_no_results(self, long_term_memory):
        text = long_term_memory.recall("nonexistent", limit=5)
        assert "没有找到" in text

    def test_forget(self, long_term_memory):
        long_term_memory.add("content to forget A", category="fact")
        long_term_memory.add("content to forget B", category="fact")
        long_term_memory.add("keep this", category="fact")

        count = long_term_memory.forget("to forget")
        assert count == 2
        assert len(long_term_memory.list_all()) == 1

    def test_stats(self, long_term_memory):
        long_term_memory.add("a", category="fact")
        long_term_memory.add("b", category="fact")
        long_term_memory.add("c", category="note")
        stats = long_term_memory.stats()
        assert stats["total"] == 3
        assert stats["by_category"]["fact"] == 2
        assert stats["by_category"]["note"] == 1

    def test_memory_entry_to_dict(self):
        entry = MemoryEntry(id=1, category="fact", content="test", created_at="now", updated_at="now", importance=3)
        d = entry.to_dict()
        assert d["id"] == 1
        assert d["importance"] == 3

    def test_embedding_table_created(self, long_term_memory):
        # Embedding table should be created on init
        rows = long_term_memory._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
        ).fetchall()
        assert len(rows) == 1

    def test_semantic_search_no_embedding_fn(self, long_term_memory):
        # Without embedding_fn, semantic_search returns empty
        long_term_memory.add("some content")
        results = long_term_memory.semantic_search("test query", limit=5)
        assert results == []

    def test_semantic_search_with_fn(self, long_term_memory):
        # Mock embedding function that returns a simple vector
        def fake_embed(text):
            # Return a deterministic vector based on text length
            return [hash(text) % 100 / 100.0, 0.5, 0.5]

        mem = long_term_memory
        mem._embedding_fn = fake_embed

        mem.add("apple banana fruit")
        mem.add("dog cat animal")

        results = mem.semantic_search("fruit", limit=10)
        # Should return results with scores
        assert len(results) > 0
        assert all(isinstance(r[1], float) for r in results)

    def test_cosine_similarity(self, long_term_memory):
        sim = long_term_memory._cosine_similarity([1, 0], [1, 0])
        assert abs(sim - 1.0) < 0.001

        sim = long_term_memory._cosine_similarity([1, 0], [0, 1])
        assert abs(sim - 0.0) < 0.001

        sim = long_term_memory._cosine_similarity([], [])
        assert sim == 0.0

    def test_multiple_adds(self, long_term_memory):
        ids = []
        for i in range(10):
            ids.append(long_term_memory.add(f"content {i}", category="test"))
        assert len(ids) == 10
        assert ids[-1] > ids[0]

    def test_reopen_reuses_data(self, long_term_memory):
        mid = long_term_memory.add("persistent data")
        long_term_memory.close()

        # Reopen same DB
        mem2 = LongTermMemory(db_path=long_term_memory.db_path)
        entry = mem2.get(mid)
        assert entry is not None
        assert entry.content == "persistent data"
        mem2.close()
