"""
进程管理工具 — 后台进程的完整生命周期管理

提供后台进程的启动、查询、轮询、终止功能。
Shell 工具通过此模块的 ProcessManager 单例追踪后台进程。

设计要点：
    • ProcessManager 是模块级单例，跨工具调用共享状态
    • 每个后台进程启动时分配独立的输出收集线程（daemon 线程）
    • 线程安全：使用 threading.Lock 保护共享状态
    • 优雅终止：先 SIGTERM → 等待 5s → SIGKILL

生命周期：
    start() → 启动进程 + 输出收集线程
    poll()  → 检查进程是否仍在运行
    read_output() → 读取已收集的输出
    kill()  → 终止进程（SIGTERM → SIGKILL）
    cleanup() → 清理已终止进程的记录
"""

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

from ai_agent.tools.base import tool


@dataclass
class ProcessInfo:
    """后台进程的运行时信息

    Attributes:
        pid: 操作系统进程 ID
        command: 执行的命令字符串
        start_time: 启动时间戳
        _process: subprocess.Popen 对象（内部使用）
        _output_chunks: 输出片段列表（由后台线程填充）

    Properties:
        is_running: 进程是否仍在运行
        returncode: 退出码（None = 仍在运行）
        elapsed: 已运行秒数
        output: 已收集的完整输出文本
    """
    pid: int
    command: str
    start_time: float = field(default_factory=time.time)
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _output_chunks: list[str] = field(default_factory=list, repr=False)

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def returncode(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def output(self) -> str:
        return "".join(self._output_chunks)


class ProcessManager:
    """后台进程管理器 — 模块级单例，线程安全

    职责：
        • start() — 启动后台进程，分配输出收集线程
        • kill()  — 优雅终止进程（SIGTERM → 5s → SIGKILL）
        • get()   — 按 PID 获取进程信息
        • list_all() — 列出所有追踪的进程
        • poll()  — 轮询进程状态（返回退出码或 None）
        • read_output() — 读取已收集的输出
        • cleanup() — 清理已终止进程的记录
        • kill_all() — 终止所有追踪的进程
    """

    def __init__(self):
        self._processes: dict[int, ProcessInfo] = {}  # PID → ProcessInfo
        self._lock = threading.Lock()  # 线程安全锁

    def start(self, command: str, **popen_kwargs) -> ProcessInfo:
        """启动后台进程并追踪（后台线程收集输出）"""
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            executable="/bin/bash",
            **popen_kwargs,
        )
        info = ProcessInfo(pid=proc.pid, command=command, _process=proc)
        with self._lock:
            self._processes[proc.pid] = info

        # 后台线程持续收集输出
        def _collect_output():
            try:
                for line in proc.stdout:
                    info._output_chunks.append(line)
            except Exception:
                pass

        t = threading.Thread(target=_collect_output, daemon=True)
        t.start()

        return info

    def kill(self, pid: int) -> bool:
        """终止指定进程，返回是否成功"""
        with self._lock:
            info = self._processes.get(pid)
        if info is None or info._process is None:
            return False
        try:
            info._process.terminate()
            try:
                info._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                info._process.kill()
                info._process.wait()
            return True
        except Exception:
            return False

    def get(self, pid: int) -> ProcessInfo | None:
        with self._lock:
            return self._processes.get(pid)

    def list_all(self) -> list[ProcessInfo]:
        with self._lock:
            return list(self._processes.values())

    def poll(self, pid: int) -> int | None:
        """轮询进程状态，返回退出码或 None（仍在运行）"""
        info = self.get(pid)
        if info is None or info._process is None:
            return None
        return info._process.poll()

    def read_output(self, pid: int) -> str | None:
        """读取后台进程的当前已收集输出"""
        info = self.get(pid)
        if info is None:
            return None
        return info.output or None

    def cleanup(self):
        """清理所有已终止的进程记录"""
        with self._lock:
            stale = [pid for pid, info in self._processes.items() if not info.is_running]
            for pid in stale:
                del self._processes[pid]

    def kill_all(self):
        """终止所有追踪的进程"""
        with self._lock:
            pids = list(self._processes.keys())
        for pid in pids:
            self.kill(pid)


# 模块级单例
_process_manager = ProcessManager()


def get_process_manager() -> ProcessManager:
    return _process_manager


# ------------------------------------------------------------
# 工具定义
# ------------------------------------------------------------

@tool(
    name="list_processes",
    description="列出所有后台运行的进程及其状态",
    params=[],
)
def list_processes() -> str:
    """列出当前所有后台进程"""
    processes = _process_manager.list_all()
    if not processes:
        return "当前没有后台进程在运行。"

    lines = [f"后台进程 ({len(processes)} 个):"]
    for info in processes:
        status = "运行中" if info.is_running else f"已结束 (退出码: {info.returncode})"
        lines.append(
            f"  PID {info.pid} | {status} | "
            f"运行 {info.elapsed:.1f}s | {info.command[:80]}"
        )
    return "\n".join(lines)


@tool(
    name="poll_process",
    description="轮询后台进程状态，读取已产生的输出。返回进程状态和当前输出。",
    params=[
        {"name": "pid", "type": "number", "description": "要轮询的进程 PID", "required": True},
    ],
)
def poll_process(pid: int) -> str:
    """读取后台进程的当前输出，不阻塞"""
    info = _process_manager.get(pid)
    if info is None:
        return f"未找到 PID {pid} 的后台进程。用 list_processes 查看可用的 PID。"

    output = info.output
    status = "运行中" if info.is_running else f"已结束 (退出码: {info.returncode})"
    lines = [
        f"PID {pid}: {status} | 运行 {info.elapsed:.1f}s",
        f"命令: {info.command[:120]}",
    ]
    if output:
        out_display = output[:3000]
        if len(output) > 3000:
            out_display += f"\n...(截断，共 {len(output)} 字符)"
        lines.append(f"输出:\n{out_display}")
    else:
        lines.append("(尚无输出)")
    return "\n".join(lines)


@tool(
    name="kill_process",
    description="终止指定 PID 的后台进程。先用 list_processes 查看 PID。",
    params=[
        {"name": "pid", "type": "number", "description": "要终止的进程 PID", "required": True},
    ],
    requires_approval=True,
)
def kill_process(pid: int) -> str:
    """终止后台进程"""
    info = _process_manager.get(pid)
    if info is None:
        return f"未找到 PID {pid} 的后台进程。可用的 PID: {[p.pid for p in _process_manager.list_all()]}"

    command = info.command[:80]
    if not info.is_running:
        return f"PID {pid} 的进程已结束（退出码: {info.returncode}）: {command}"

    if _process_manager.kill(pid):
        return f"已终止进程 PID {pid}: {command}"
    else:
        return f"无法终止进程 PID {pid}: {command}"
