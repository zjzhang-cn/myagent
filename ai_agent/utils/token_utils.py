"""
Token 估算工具

为上下文窗口管理提供轻量级 token 估算。不依赖 tiktoken 等外部库，
使用基于字符类型的启发式算法：CJK 字符 ≈ 1.5 tokens，其他字符 ≈ 0.25 tokens。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Unicode 范围：CJK 统一表意文字 + 标点
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
    (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), # CJK Compatibility Ideographs Supplement
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0x1100, 0x11FF),   # Hangul Jamo
]


def _is_cjk_char(ch: str) -> bool:
    """判断字符是否属于 CJK 范围"""
    cp = ord(ch)
    for start, end in _CJK_RANGES:
        if start <= cp <= end:
            return True
    return False


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。

    CJK 字符通常 1 个字符 ≈ 1-2 tokens，这里取 1.5。
    拉丁/其他字符通常 4 个字符 ≈ 1 token，这里取 0.25。

    此外加上每条消息约 4 tokens 的开销（role + formatting）。
    """
    if not text:
        return 0

    cjk_count = 0
    other_count = 0

    for ch in text:
        if _is_cjk_char(ch):
            cjk_count += 1
        else:
            other_count += 1

    # CJK: ~1.5 tokens per char; Other: ~0.25 tokens per char (4 chars/token)
    estimated = int(cjk_count * 1.5 + other_count * 0.25)
    return max(estimated, 1)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """
    估算单条消息的 token 数量（含 role 和 formatting 开销）。
    """
    overhead = 4  # role + message delimiters
    tokens = overhead

    # Role token
    role = message.get("role", "")
    tokens += estimate_tokens(role)

    # Content token
    content = message.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)
    elif isinstance(content, list):
        # 多模态内容（罕见）
        for part in content:
            if isinstance(part, dict) and "text" in part:
                tokens += estimate_tokens(part["text"])

    # Tool call overhead (if present)
    if "tool_calls" in message:
        for tc in message["tool_calls"]:
            tokens += estimate_tokens(tc.get("name", ""))
            args = tc.get("arguments", {})
            if isinstance(args, dict):
                tokens += estimate_tokens(str(args))
            tokens += 4  # tool_call formatting

    return tokens


def estimate_messages_tokens(messages: list[dict]) -> int:
    """
    估算消息列表的总 token 数量。
    """
    return sum(estimate_message_tokens(msg) for msg in messages)


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    截断文本到指定字符数，保留开头和结尾各一半。

    Args:
        text: 原始文本
        max_chars: 最大字符数
        suffix: 截断标记

    Returns:
        截断后的文本
    """
    if len(text) <= max_chars:
        return text

    half = (max_chars - len(suffix)) // 2
    if half <= 0:
        return text[:max_chars - len(suffix)] + suffix

    omitted = len(text) - max_chars + len(suffix)
    return (
        text[:half]
        + f"\n...[已省略 {omitted} 字符]...\n"
        + text[-half:]
    )
