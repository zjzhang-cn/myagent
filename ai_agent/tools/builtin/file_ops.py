"""
文件操作工具

所有路径操作均通过安全模块的路径沙箱进行验证，
确保 Agent 只能访问配置允许的目录。
"""

import os
import glob as glob_mod

from ai_agent.tools.base import tool
from ai_agent.utils.security import check_path


@tool(
    name="read_file",
    description="读取文件内容，可以指定读取的行范围",
    params=[
        {"name": "path", "type": "string", "description": "文件路径", "required": True},
        {"name": "start_line", "type": "number", "description": "起始行号（从1开始）", "required": False},
        {"name": "end_line", "type": "number", "description": "结束行号（含）", "required": False},
    ],
)
def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """读取文件内容（路径受沙箱限制）"""
    try:
        safe_path = check_path(path, must_exist=True, for_write=False)
    except PermissionError as e:
        return f"权限拒绝: {e}"
    except FileNotFoundError as e:
        return f"文件不存在: {e}"

    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line is not None:
            start_idx = max(0, start_line - 1)
            lines = lines[start_idx:]
        if end_line is not None:
            end_idx = min(len(lines), end_line - (start_line or 1) + 1)
            lines = lines[:end_idx]

        if len(lines) > 500:
            return "".join(lines[:500]) + f"\n...(截断，共 {len(lines)} 行，只展示前500行)"

        return "".join(lines)
    except Exception as e:
        return f"读取文件失败: {e}"


@tool(
    name="write_file",
    description="写入内容到文件（覆盖写或追加）",
    params=[
        {"name": "path", "type": "string", "description": "文件路径", "required": True},
        {"name": "content", "type": "string", "description": "要写入的内容", "required": True},
        {"name": "mode", "type": "string", "description": "写入模式: 'w'=覆盖, 'a'=追加", "required": False},
    ],
)
def write_file(path: str, content: str, mode: str = "w") -> str:
    """写入文件（路径受沙箱限制）"""
    try:
        safe_path = check_path(path, must_exist=False, for_write=True)
    except PermissionError as e:
        return f"权限拒绝: {e}"

    try:
        os.makedirs(os.path.dirname(safe_path) or ".", exist_ok=True)
        with open(safe_path, mode, encoding="utf-8") as f:
            f.write(content)
        return f"成功写入文件: {safe_path} ({len(content)} 字符)"
    except Exception as e:
        return f"写入文件失败: {e}"


@tool(
    name="list_directory",
    description="列出目录中的文件和子目录",
    params=[
        {"name": "path", "type": "string", "description": "目录路径", "required": True},
        {"name": "pattern", "type": "string", "description": "glob 匹配模式，如 '*.py'", "required": False},
    ],
)
def list_directory(path: str, pattern: str = "*") -> str:
    """列出目录内容（路径受沙箱限制）"""
    try:
        safe_path = check_path(path, must_exist=True, for_write=False)
    except PermissionError as e:
        return f"权限拒绝: {e}"
    except FileNotFoundError as e:
        return f"目录不存在: {e}"

    if not os.path.isdir(safe_path):
        return f"不是目录: {safe_path}"

    try:
        search_pattern = os.path.join(safe_path, pattern)
        items = glob_mod.glob(search_pattern)

        if not items:
            return f"目录 '{safe_path}' 中没有匹配 '{pattern}' 的文件"

        result_lines = []
        for item in sorted(items):
            name = os.path.basename(item)
            item_type = "D" if os.path.isdir(item) else "F"
            size = ""
            if os.path.isfile(item):
                s = os.path.getsize(item)
                if s < 1024:
                    size = f"{s}B"
                elif s < 1024 * 1024:
                    size = f"{s / 1024:.1f}KB"
                else:
                    size = f"{s / (1024 * 1024):.1f}MB"
            result_lines.append(f"[{item_type}] {name} {size}".strip())

        return f"目录 '{safe_path}' (共 {len(items)} 项):\n" + "\n".join(result_lines[:200])
    except Exception as e:
        return f"列出目录失败: {e}"


@tool(
    name="delete_file",
    description="删除指定的文件或空目录",
    params=[
        {"name": "path", "type": "string", "description": "要删除的文件路径", "required": True},
    ],
)
def delete_file(path: str) -> str:
    """删除文件（路径受沙箱限制）"""
    try:
        safe_path = check_path(path, must_exist=True, for_write=True)
    except PermissionError as e:
        return f"权限拒绝: {e}"
    except FileNotFoundError as e:
        return f"文件不存在: {e}"

    try:
        if os.path.isdir(safe_path):
            os.rmdir(safe_path)
            return f"成功删除空目录: {safe_path}"
        else:
            os.remove(safe_path)
            return f"成功删除文件: {safe_path}"
    except OSError as e:
        return f"删除失败: {e}"
