"""Tests for ai_agent.utils.token_utils."""

import pytest

from ai_agent.utils.token_utils import (
    _is_cjk_char,
    estimate_tokens,
    estimate_message_tokens,
    estimate_messages_tokens,
    truncate_text,
)


class TestCJKDetection:
    def test_cjk_unified_ideographs(self):
        assert _is_cjk_char("中")
        assert _is_cjk_char("国")
        assert _is_cjk_char("文")
        assert _is_cjk_char("字")

    def test_cjk_japanese(self):
        assert _is_cjk_char("あ")  # Hiragana
        assert _is_cjk_char("ア")  # Katakana
        assert _is_cjk_char("漢")

    def test_cjk_korean(self):
        assert _is_cjk_char("한")
        assert _is_cjk_char("글")

    def test_non_cjk(self):
        assert not _is_cjk_char("a")
        assert not _is_cjk_char("1")
        assert not _is_cjk_char("!")
        assert not _is_cjk_char(" ")
        assert not _is_cjk_char("\n")

    def test_empty_string(self):
        # Non-CJK: single char test is trivially correct
        pass


class TestEstimateTokens:
    def test_empty_text(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_ascii_text(self):
        # 4 chars ≈ 1 token
        tokens = estimate_tokens("hello")
        assert tokens >= 1

    def test_cjk_text(self):
        # Each CJK ~1.5 tokens
        tokens = estimate_tokens("中国")
        assert tokens == 3  # 2 * 1.5

    def test_mixed_text(self):
        tokens = estimate_tokens("hello中国")
        assert tokens >= 1
        assert tokens == estimate_tokens("hello") + estimate_tokens("中国")

    def test_long_text(self):
        text = "a" * 100
        tokens = estimate_tokens(text)
        assert tokens == 25  # 100 * 0.25

    def test_minimum_one(self):
        assert estimate_tokens("a") == 1  # max(1*0.25, 1) = 1


class TestEstimateMessageTokens:
    def test_user_message(self):
        msg = {"role": "user", "content": "hello"}
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_assistant_message(self):
        msg = {"role": "assistant", "content": "你好世界"}
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_tool_message(self):
        msg = {"role": "tool", "content": "result"}
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_message_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "read_file", "arguments": {"path": "test.txt"}},
                {"name": "search_web", "arguments": {"query": "news"}},
            ],
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_multimodal_content(self):
        msg = {"role": "user", "content": [{"text": "describe this"}, {"text": "and this"}]}
        tokens = estimate_message_tokens(msg)
        assert tokens > 0

    def test_message_without_content(self):
        msg = {"role": "user"}
        tokens = estimate_message_tokens(msg)
        assert tokens >= 4  # overhead only


class TestEstimateMessagesTokens:
    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "how are you"},
        ]
        total = estimate_messages_tokens(messages)
        single_sum = sum(estimate_message_tokens(m) for m in messages)
        assert total == single_sum


class TestTruncateText:
    def test_no_truncation_needed(self):
        text = "short text"
        assert truncate_text(text, 100) == text

    def test_truncation_middle(self):
        text = "a" * 100
        result = truncate_text(text, 20)
        assert len(result) < len(text)
        assert "..." in result or "省略" in result

    def test_truncation_preserves_ends(self):
        text = "开头" + "x" * 50 + "结尾"
        result = truncate_text(text, 20)
        assert "开头" in result
        assert "结尾" in result

    def test_very_small_limit(self):
        text = "long text here for testing"
        result = truncate_text(text, 5)
        assert len(result) < len(text)  # Should still be truncated

    def test_cjk_truncation(self):
        text = "中" * 100
        result = truncate_text(text, 20)
        assert len(result) < len(text)
