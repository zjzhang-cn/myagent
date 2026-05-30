"""
安全工具模块 — 路径沙箱 + 命令白名单

提供双重安全防护，防止 Agent 访问未授权资源或执行危险命令：

    第一道防线 — 路径沙箱:
        sandbox_path() 实现两阶段路径验证：
            Phase 1: normpath 规范化 → 检查是否在允许目录内（路径遍历检测）
            Phase 2: realpath 解析符号链接 → 再次检查（符号链接逃逸检测）
        写操作额外拦截：禁止写入 .git / .ssh / .gnupg 等敏感隐藏目录

    第二道防线 — 命令白名单:
        validate_shell_command() 实现双层检查：
            Layer 1: 正则匹配危险模式（rm -rf /, fork bomb, curl|sh 等）
            Layer 2: shlex 解析基础命令 → 检查是否在白名单中

    安全上下文 — 线程本地:
        SecurityContext 存储在 threading.local 中，工具无需修改函数签名。
        Agent 在工具执行前自动设置上下文，工具通过 get_security_context() 读取。

使用方式：
    # 路径验证
    safe_path = check_path(user_path, must_exist=True, for_write=False)

    # 命令验证
    is_safe, msg = check_command("ls -la")
"""

import logging
import os
import re
import shlex
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 默认安全 Shell 命令白名单（只读 + 常用开发工具）
# ----------------------------------------------------------

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
    # 文件操作（受限，有额外警告）
    "mkdir", "cp", "mv", "touch", "rm",
    # 权限变更（受限，有额外警告）
    "chmod", "chown",
    # 压缩归档
    "tar", "gzip", "gunzip", "zip", "unzip",
}

# 额外需要警告的命令（允许执行但可能有风险）
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
    security_ctx: "SecurityContext | None" = None,
) -> str:
    """
    解析并验证路径是否在允许的目录内 — 两阶段验证

    阶段一：路径遍历检测
        1. 展开 ~ 和 ~user
        2. 解析为绝对路径
        3. normpath 规范化（消除 .., . 等）
        4. 检查规范化路径是否在允许目录内（Path.relative_to）

    阶段二：符号链接逃逸检测
        5. realpath 解析符号链接得到真实路径
        6. 再次检查真实路径是否在允许目录内
        7. 对于不存在的路径，解析最近已存在父目录的 realpath

    写操作额外检查：
        8. 禁止写入 .git / .ssh / .gnupg / .config / .local 等敏感隐藏目录

    存在性检查：
        9. must_exist=True 时检查路径是否存在

    Args:
        path: 用户提供的路径（可能包含 ~, ../, 符号链接等）
        allowed_dirs: 允许访问的目录列表
        must_exist: 是否要求路径必须已存在（读操作设为 True）
        for_write: 是否为写操作（触发额外安全检查）
        security_ctx: 安全上下文（用于权限回调）

    Returns:
        解析后的安全绝对路径

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
        # 检查权限回调
        if security_ctx and security_ctx.on_permission_denied:
            if security_ctx.on_permission_denied(path, f"路径 '{path}' 不在允许的目录范围内"):
                return normalized
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
        if security_ctx and security_ctx.on_permission_denied:
            if security_ctx.on_permission_denied(path, f"路径 '{path}' 解析后不在允许的目录范围内"):
                return real_path
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
                    if security_ctx and security_ctx.on_permission_denied:
                        if security_ctx.on_permission_denied(path, f"写入隐藏目录 '{part}' 被拦截"):
                            return real_path
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
    验证 Shell 命令是否在允许的范围内 — 双层检查

    第一层 — 正则匹配危险模式:
        • rm -rf / 或 rm -rf /* 等破坏性删除
        • mkfs.* 文件系统创建
        • dd if= 原始磁盘操作
        • > /dev/sd* / > /dev/nvme* 重定向到磁盘设备
        • chmod -R 777 / 或 chmod 777 / 全局权限变更
        • fork bomb 模式 :(){ :|:& };:
        • curl|sh 或 wget|sh 管道到 shell 执行
        • sudo rm -rf /, sudo dd, sudo mkfs 等 sudo 危险操作

    第二层 — 命令白名单检查:
        使用 shlex.split 解析命令，提取基础命令名。
        检查是否在 allowed_commands 白名单中。
        对高风险命令（rm, chmod, curl 等）发出警告但允许执行。

    Args:
        command: 要执行的命令字符串
        allowed_commands: 允许的命令白名单
        allow_all: 是否允许所有命令（跳过检查，仅用于调试）

    Returns:
        (is_safe, message) 元组：
            is_safe=True 表示命令可以执行
            message 可能是空字符串（完全安全）或警告信息
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

    处理逻辑：
        1. 展开 ~ 路径
        2. 转换为绝对路径
        3. 始终包含当前工作目录
        4. 去重并排序
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
    """安全上下文 — 由 Agent 在工具执行前通过线程本地存储设置

    设计目的：
        工具函数（如 read_file, run_shell_command）无需修改函数签名即可获取安全配置。
        Agent 在执行每个工具前自动设置 SecurityContext，工具通过 check_path/check_command 读取。

    Attributes:
        allowed_directories: 文件操作允许的目录列表
        allowed_commands: Shell 命令白名单
        allow_all_commands: 是否允许所有命令（跳过检查）
        max_file_read_bytes: 读文件的最大字节数限制（默认 1MB）
        enabled: 是否启用安全检查（设为 False 关闭所有检查）
        on_permission_denied: 权限拒绝回调 (path, reason) → bool。
                              返回 True 允许本次操作（覆盖拒绝决定）
    """
    allowed_directories: list[str] = field(default_factory=lambda: ["."])
    allowed_commands: set[str] = field(default_factory=set)
    allow_all_commands: bool = False
    max_file_read_bytes: int = 1_000_000  # 读文件最大字节数（1MB）
    enabled: bool = True  # 是否启用安全检查
    on_permission_denied: Callable[[str, str], bool] | None = None
    """权限拒绝回调 (path, reason) -> bool。返回 True 允许本次操作"""


# 线程本地存储 — 每个线程独立的安全上下文
_security_context = threading.local()


def set_security_context(ctx: SecurityContext) -> None:
    """设置当前线程的安全上下文（Agent 在工具执行前调用）"""
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
    return sandbox_path(path, ctx.allowed_directories, must_exist, for_write, security_ctx=ctx)


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
