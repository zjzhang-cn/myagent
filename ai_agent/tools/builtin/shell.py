"""
Shell 命令执行工具

通过安全模块的命令白名单和路径沙箱进行双重验证，
确保 Agent 只能执行允许的命令且在指定的目录范围内。
"""

import os
import subprocess

from ai_agent.tools.base import tool
from ai_agent.utils.security import check_command, check_path


@tool(
    name="run_shell_command",
    description="执行 Shell 命令。background=True 时异步执行并返回 PID，之后可用 poll_process 读取输出、list_processes 查看状态、kill_process 终止。仅限安全命令。",
    params=[
        {"name": "command", "type": "string", "description": "要执行的 Shell 命令", "required": True},
        {"name": "working_dir", "type": "string", "description": "工作目录（可选，受路径沙箱限制）", "required": False},
        {"name": "timeout", "type": "number", "description": "超时秒数，默认30（background=True 时忽略）", "required": False},
        {"name": "background", "type": "boolean", "description": "是否后台异步执行，默认 false", "required": False},
    ],
    requires_approval=True,
)
def run_shell_command(
    command: str,
    working_dir: str | None = None,
    timeout: int = 30,
    background: bool = False,
) -> str:
    """安全地执行 Shell 命令（命令白名单 + 路径沙箱）"""
    # Step 1: 命令白名单检查
    is_safe, msg = check_command(command)
    if not is_safe:
        return f"⚠️ 安全拦截：{msg}"

    # Step 2: 工作目录沙箱检查
    cwd = None
    if working_dir:
        try:
            safe_dir = check_path(working_dir, must_exist=True, for_write=False)
            cwd = safe_dir
        except PermissionError as e:
            return f"权限拒绝：工作目录 {e}"
        except FileNotFoundError as e:
            return f"工作目录不存在: {e}"
    else:
        cwd = os.getcwd()

    # Step 3: 后台模式 — 异步执行
    if background:
        from ai_agent.tools.builtin.processes import get_process_manager
        pm = get_process_manager()
        info = pm.start(command, cwd=cwd)
        tip = f" 用 poll_process(pid={info.pid}) 读取输出，list_processes() 查看状态，kill_process(pid={info.pid}) 终止。"
        return (
            f"后台进程已启动\n"
            f"PID: {info.pid}\n"
            f"命令: {command}\n"
            f"提示:{tip}"
        )

    # Step 4: 前台模式 — 同步执行（原有逻辑）
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            executable="/bin/bash",
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n[stderr]\n"
            output += result.stderr

        if not output:
            output = "(无输出)"

        if len(output) > 4000:
            output = output[:4000] + f"\n...(截断，共 {len(output)} 字符)"

        prefix = f"命令: {command}\n"
        if msg:  # 警告信息
            prefix += f"{msg}\n"
        if result.returncode != 0:
            prefix += f"退出码: {result.returncode}\n"
        return prefix + output

    except subprocess.TimeoutExpired:
        return f"命令超时 ({timeout}s): {command}"
    except Exception as e:
        return f"命令执行异常: {e}"
