"""
Shell 命令执行工具
"""

import subprocess

from ai_agent.tools.base import tool

# 危险命令黑名单
_BLOCKED_COMMANDS = {
    "rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:", "> /dev/sda",
    "chmod -R 777 /", "mv /* /dev/null",
}


@tool(
    name="run_shell_command",
    description="执行 Shell 命令并返回输出。仅限安全命令，危险操作会被拦截。",
    params=[
        {"name": "command", "type": "string", "description": "要执行的 Shell 命令", "required": True},
        {"name": "working_dir", "type": "string", "description": "工作目录（可选）", "required": False},
        {"name": "timeout", "type": "number", "description": "超时秒数，默认30", "required": False},
    ],
)
def run_shell_command(
    command: str,
    working_dir: str | None = None,
    timeout: int = 30,
) -> str:
    """安全地执行 Shell 命令"""
    import os

    # 安全校验
    cmd_lower = command.lower().replace(" ", "")
    for blocked in _BLOCKED_COMMANDS:
        blocked_compact = blocked.replace(" ", "")
        if blocked_compact in cmd_lower:
            return f"⚠️ 安全拦截：命令包含危险操作 ({blocked})，已被拒绝执行。"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.expanduser(working_dir) if working_dir else None,
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
        if result.returncode != 0:
            prefix += f"退出码: {result.returncode}\n"
        return prefix + output

    except subprocess.TimeoutExpired:
        return f"命令超时 ({timeout}s): {command}"
    except Exception as e:
        return f"命令执行异常: {e}"
