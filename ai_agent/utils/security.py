"""
安全工具模块

提供路径沙箱和命令白名单功能，防止 Agent 访问未授权资源或执行危险命令。

通过线程本地安全上下文，工具函数无需修改签名即可获取安全配置。
Agent 在执行工具前自动设置上下文，工具在执行时读取。
"""

import logging
import os
import re
import shlex
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认可信 Shell 命令白名单（只读 + 常用开发工具）
DEFAULT_SAFE_COMMANDS: set[str] = {
    # 文件查看
    "ls", "cat", "head", "tail", "less", "more",
    # 信息查询
    "pwd", "echo", "date", "whoami", "which", "whereis", "type",
    "uname", "hostname", "uptime",
    # 文本处理（只读）
    "grep", "egrep", "fgrep", "wc", "sort", "uniq", "cut", "tr",
    "awk", "sed", "xargs",
    # 文件查找
    "find", "locate",
    # 磁盘/内存查看
    "df", "du", "free", "env", "printenv",
    # 进程查看
    "ps", "pgrep", "top",
    # 网络诊断（只读）
    "ping", "curl", "wget",
    # 开发工具
    "python", "python3", "pip", "pip3",
    "git", "node", "npm", "npx",
    "make", "cmake",
    # 文件操作（受限）
    "mkdir", "cp", "mv", "touch", "rm",
    # 权限
    "chmod", "chown",
    # 压缩
    "tar", "gzip", "gunzip", "zip", "unzip",
}

# 额外需要警告的命令（可能有风险）
_COMMANDS_NEED_WARNING: set[str] = {
    "rm", "chmod", "chown", "mv", "cp",
    "pip", "pip3", "npm", "npx",
    "curl", "wget", "git",
}


def sandbox_path(
    path: str,
    allowed_dirs: list[str],
    must_exist: bool = False,
    for_write: bool = False,
) -> str:
    """
    解析并验证路径是否在允许的目录内。

    Args:
        path: 用户提供的路径（可能包含 ~, ../, 符号链接等）
        allowed_dirs: 允许访问的目录列表
        must_exist: 是否要求路径必须已存在（读操作）
        for_write: 是否为写操作（更严格的检查）

    Returns:
        解析后的绝对路径

    Raises:
        PermissionError: 路径不在允许的目录内
        FileNotFoundError: must_exist 为 True 但路径不存在
    """
    # Step 1: 展开 ~ 和 ~user
    expanded = os.path.expanduser(path)

    # Step 2: 解析为绝对路径
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(expanded)

    # Step 3: 规范化路径（消除 .., . 等）—— 第一道防线：路径遍历检测
    normalized = os.path.normpath(expanded)

    # Step 3b: 提前检查规范化路径是否在允许目录内（符号链接解析之前）
    # 这样可以捕获 ../ 路径遍历攻击
    normalized_obj = Path(normalized)
    is_within_allowed = False
    for allowed_dir in allowed_dirs:
        allowed_expanded = os.path.expanduser(allowed_dir)
        if not os.path.isabs(allowed_expanded):
            allowed_expanded = os.path.abspath(allowed_expanded)
        allowed_normalized = os.path.normpath(allowed_expanded)
        allowed_path_obj = Path(allowed_normalized)
        try:
            normalized_obj.relative_to(allowed_path_obj)
            is_within_allowed = True
            break
        except ValueError:
            continue

    if not is_within_allowed:
        raise PermissionError(
            f"路径访问被拒绝：'{path}' 不在允许的目录范围内（路径遍历检测）。\n"
            f"  规范化路径: {normalized}\n"
            f"  允许的目录: {', '.join(allowed_dirs)}"
        )

    # Step 4: 解析符号链接得到真实路径（第二道防线：符号链接逃逸检测）
    try:
        if os.path.exists(normalized):
            real_path = os.path.realpath(normalized)
        else:
            # 路径不存在，尝试解析最近的已存在父目录
            parent = normalized
            while parent and not os.path.exists(parent):
                parent = os.path.dirname(parent)
            if parent and parent != "/":
                real_parent = os.path.realpath(parent)
                rel_part = os.path.relpath(normalized, parent)
                real_path = os.path.normpath(os.path.join(real_parent, rel_part))
            else:
                real_path = normalized
    except Exception as e:
        raise PermissionError(f"路径解析失败 '{path}': {e}")

    # Step 5: 再次检查真实路径是否在允许目录内（符号链接逃逸检测）
    real_path_obj = Path(real_path)
    is_allowed = False
    for allowed_dir in allowed_dirs:
        allowed_expanded = os.path.expanduser(allowed_dir)
        if not os.path.isabs(allowed_expanded):
            allowed_expanded = os.path.abspath(allowed_expanded)
        allowed_real = os.path.realpath(allowed_expanded) if os.path.exists(allowed_expanded) else os.path.normpath(allowed_expanded)
        allowed_path_obj = Path(allowed_real)
        try:
            real_path_obj.relative_to(allowed_path_obj)
            is_allowed = True
            break
        except ValueError:
            continue

    if not is_allowed:
        raise PermissionError(
            f"路径访问被拒绝：'{path}' 的真实路径不在允许的目录范围内。\n"
            f"  解析后路径: {real_path}\n"
            f"  允许的目录: {', '.join(allowed_dirs)}"
        )

    # Step 6: 写操作额外安全检查
    if for_write:
        # 不允许写入到隐藏目录（如 .git, .ssh 等）
        path_parts = real_path_obj.parts
        for part in path_parts:
            if part.startswith(".") and part not in (".", ".."):
                if part in (".git", ".ssh", ".gnupg", ".config", ".local"):
                    raise PermissionError(
                        f"写入操作被拒绝：不允许修改 '{part}' 目录中的文件。"
                        f"这是系统或配置目录，修改可能导致安全问题。"
                    )

    # Step 7: 存在性检查
    if must_exist and not os.path.exists(real_path):
        raise FileNotFoundError(f"文件不存在: {real_path}")

    return real_path


def validate_shell_command(
    command: str,
    allowed_commands: set[str],
    allow_all: bool = False,
) -> tuple[bool, str]:
    """
    验证 Shell 命令是否在允许的范围内。

    Args:
        command: 要执行的命令字符串
        allowed_commands: 允许的命令白名单
        allow_all: 是否允许所有命令（跳过检查）

    Returns:
        (is_safe, message) 元组
    """
    if allow_all:
        return True, ""

    if not command or not command.strip():
        return False, "命令为空"

    # 检查危险模式（使用正则表达式匹配）
    dangerous_patterns = [
        r"rm\s+-rf\s+/(\*|\s*$|$)",     # rm -rf / or rm -rf /*
        r"mkfs\.",                        # filesystem creation
        r"dd\s+if=",                      # raw disk operations
        r">\s*/dev/sd[a-z]",              # redirect to raw disk
        r">\s*/dev/nvme",                 # redirect to NVMe
        r"chmod\s+-R\s+777\s+/",         # recursive 777 on root
        r"chmod\s+777\s+/",              # 777 on root
        r":\(\)\s*\{\s*:\|:&\s*\};:",     # fork bomb
        r"mv\s+/\*\s+/dev/null",         # move all to void
        r"curl\s+.*\|\s*(ba)?sh",         # curl pipe to shell
        r"wget\s+.*\|\s*(ba)?sh",         # wget pipe to shell
        r"wget\s+-O\s+-\s+http",          # wget output to stdout
        r"/dev/sd[a-z]\d",                # raw disk device reference
        r"sudo\s+rm\s+-rf\s+/",           # sudo rm -rf /
        r"sudo\s+dd\s+",                  # sudo dd
        r"sudo\s+mkfs",                   # sudo mkfs
    ]
    cmd_compact = re.sub(r'\s+', ' ', command.lower().strip())
    for pattern in dangerous_patterns:
        if re.search(pattern, cmd_compact, re.IGNORECASE):
            return False, f"危险命令模式被拦截: {pattern}"

    # 提取基础命令（处理管道和重定向）
    try:
        tokens = shlex.split(command)
    except ValueError:
        # shlex 无法解析，尝试简单分割
        tokens = command.split()

    if not tokens:
        return False, "无法解析命令"

    base_cmd = os.path.basename(tokens[0])

    # 检查是否在白名单中
    if base_cmd not in allowed_commands:
        return False, (
            f"命令 '{base_cmd}' 不在允许列表中。\n"
            f"允许的命令（部分）: {', '.join(sorted(list(allowed_commands))[:25])}..."
        )

    # 对某些命令发出警告但允许执行
    warning = ""
    if base_cmd in _COMMANDS_NEED_WARNING:
        warning = f"⚠️ 命令 '{base_cmd}' 可能有风险，请确保了解其影响。"

    return True, warning


def get_allowed_directories(config_dirs: list[str]) -> list[str]:
    """
    获取允许的目录列表，包含默认值。

    始终包含当前工作目录，并去重、解析 ~ 路径。
    """
    dirs = set()
    for d in config_dirs:
        expanded = os.path.expanduser(d)
        abs_path = os.path.abspath(expanded)
        dirs.add(abs_path)

    # 始终包含当前工作目录
    cwd = os.getcwd()
    dirs.add(cwd)

    return sorted(dirs)


# ============================================================
# 线程本地安全上下文
# ============================================================

@dataclass
class SecurityContext:
    """安全上下文，由 Agent 在工具执行前设置"""
    allowed_directories: list[str] = field(default_factory=lambda: ["."])
    allowed_commands: set[str] = field(default_factory=set)
    allow_all_commands: bool = False
    max_file_read_bytes: int = 1_000_000  # 读文件最大字节数
    enabled: bool = True  # 是否启用安全检查


_security_context = threading.local()


def set_security_context(ctx: SecurityContext) -> None:
    """设置当前线程的安全上下文"""
    _security_context.ctx = ctx


def get_security_context() -> SecurityContext:
    """获取当前线程的安全上下文，如果没有设置则返回默认（允许所有）"""
    if not hasattr(_security_context, "ctx") or _security_context.ctx is None:
        return SecurityContext(enabled=False)
    return _security_context.ctx


def clear_security_context() -> None:
    """清除当前线程的安全上下文"""
    if hasattr(_security_context, "ctx"):
        _security_context.ctx = None


def check_path(path: str, must_exist: bool = False, for_write: bool = False) -> str:
    """
    便捷函数：使用当前安全上下文验证路径。

    Args:
        path: 要验证的路径
        must_exist: 是否要求路径存在
        for_write: 是否为写操作

    Returns:
        解析后的安全路径

    Raises:
        PermissionError: 路径不在允许范围内
    """
    ctx = get_security_context()
    if not ctx.enabled:
        return os.path.abspath(os.path.expanduser(path))
    return sandbox_path(path, ctx.allowed_directories, must_exist, for_write)


def check_command(command: str) -> tuple[bool, str]:
    """
    便捷函数：使用当前安全上下文验证命令。

    Returns:
        (is_safe, message) 元组
    """
    ctx = get_security_context()
    if not ctx.enabled:
        return True, ""
    return validate_shell_command(command, ctx.allowed_commands, ctx.allow_all_commands)
