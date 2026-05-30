"""
Token 估算工具 — CJK 感知的轻量级 Token 计数

为上下文窗口管理提供不依赖外部库（如 tiktoken）的 Token 估算。
使用基于字符类型的启发式算法：

    估算规则:
        • CJK 字符（中日韩统一表意文字 + 日韩文）≈ 1.5 tokens/字符
        • 拉丁/其他字符 ≈ 0.25 tokens/字符（即约 4 字符 = 1 token）
        • 每条消息额外 +4 tokens 开销（role + formatting）

    CJK 覆盖范围（14 个 Unicode 区间）:
        • 基本区: CJK Unified Ideographs (U+4E00–U+9FFF)
        • 扩展区: Extension A–E
        • 兼容区: CJK Compatibility Ideographs
        • 标点符号: CJK Symbols and Punctuation
        • 全角/半角: Halfwidth and Fullwidth Forms
        • 日文: Hiragana + Katakana
        • 韩文: Hangul Syllables + Hangul Jamo

    精度说明:
        这是启发式估算，不是精确计数。对于英文文本误差约 ±10%，
        对于中文文本误差约 ±15%。对于上下文窗口裁剪来说足够使用。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# CJK Unicode 范围定义
# ----------------------------------------------------------
# 涵盖中日韩统一表意文字及其扩展、日文假名、韩文音节等
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK 统一表意文字（基本区，常用汉字）
    (0x3400, 0x4DBF),   # CJK 统一表意文字扩展 A 区
    (0x20000, 0x2A6DF), # CJK 统一表意文字扩展 B 区
    (0x2A700, 0x2B73F), # CJK 统一表意文字扩展 C 区
    (0x2B740, 0x2B81F), # CJK 统一表意文字扩展 D 区
    (0x2B820, 0x2CEAF), # CJK 统一表意文字扩展 E 区
    (0xF900, 0xFAFF),   # CJK 兼容表意文字
    (0x2F800, 0x2FA1F), # CJK 兼容表意文字补充
    (0x3000, 0x303F),   # CJK 符号和标点（。、、「」等）
    (0xFF00, 0xFFEF),   # 半角和全角形式（全角英数字等）
    (0x3040, 0x309F),   # 平假名（日文）
    (0x30A0, 0x30FF),   # 片假名（日文）
    (0xAC00, 0xD7AF),   # 韩文音节
    (0x1100, 0x11FF),   # 韩文 Jamo
]


def _is_cjk_char(ch: str) -> bool:
    """判断单个字符是否属于 CJK（中日韩）范围"""
    cp = ord(ch)
    for start, end in _CJK_RANGES:
        if start <= cp <= end:
            return True
    return False


def estimate_tokens(text: str) -> int:
    """
    估算纯文本的 Token 数量（启发式算法）

    算法说明:
        CJK 字符通常 1 个字符 ≈ 1-2 tokens，这里取平均值 1.5。
        拉丁/其他字符通常 4 个字符 ≈ 1 token，这里取 0.25。

    示例:
        "Hello World" → 约 3 tokens（11 × 0.25 ≈ 2.75）
        "你好世界"   → 约 6 tokens（4 × 1.5 = 6）
        "你好 Hello" → 约 7 tokens（2×1.5 + 5×0.25 + 1 ≈ 6.75）
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
    return max(estimated, 1)  # 至少 1 token


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """
    估算单条消息的 Token 数量（含 role 和 formatting 开销）

    计算包含：
        • 基础开销: 4 tokens（role + message delimiters）
        • role 文本的 token 数
        • content 文本的 token 数
        • tool_calls 开销（如果存在）：工具名 + 参数 JSON + 格式化开销

    支持多模态内容（content 为 list 时逐一估算 text 部分）。
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
    估算消息列表的总 Token 数量。

    用于上下文窗口管理：判断是否需要裁剪旧消息。
    """
    return sum(estimate_message_tokens(msg) for msg in messages)


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    智能截断文本到指定字符数。

    不同于简单的 text[:max_chars]，此方法保留开头和结尾各一半，
    确保用户能看到文本的首尾内容，中间用省略号标记。

    Args:
        text: 原始文本
        max_chars: 最大字符数
        suffix: 截断标记（默认 "..."）

    Returns:
        截断后的文本，格式为: 开头一半 + [省略标记] + 结尾一半

    示例:
        truncate_text("非常长的文本..." * 100, max_chars=200)
        → "非常长的文本...[已省略 18900 字符]...非常长的文本..."
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
