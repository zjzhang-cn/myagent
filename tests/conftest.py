"""pytest fixtures for AI Agent tests."""

import os
import tempfile

import pytest

from ai_agent.config import AgentConfig
from ai_agent.core.memory import LongTermMemory, ShortTermMemory, WorkingMemory
from ai_agent.utils.security import SecurityContext, set_security_context


@pytest.fixture
def temp_dir():
    """创建一个临时目录用于文件操作测试。"""
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        # Use realpath to resolve symlinks (e.g., /var -> /private/var on macOS)
        real_d = os.path.realpath(d)
        os.chdir(real_d)
        yield real_d
        os.chdir(cwd)


@pytest.fixture
def temp_file(temp_dir):
    """在临时目录中创建一个测试文件。"""
    path = os.path.join(temp_dir, "test.txt")
    with open(path, "w") as f:
        f.write("hello world\nline 2\nline 3\n")
    return path


@pytest.fixture
def short_term_memory():
    """创建一个短期记忆实例。"""
    mem = ShortTermMemory(max_size=5)
    mem.set_system_prompt("测试提示词")
    return mem


@pytest.fixture
def working_memory():
    """创建一个工作记忆实例。"""
    return WorkingMemory()


@pytest.fixture
def long_term_memory():
    """创建一个临时长期记忆实例（SQLite）。"""
    db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db.close()
    mem = LongTermMemory(db_path=db.name)
    yield mem
    mem.close()
    os.unlink(db.name)


@pytest.fixture
def security_context(temp_dir):
    """设置一个默认的安全上下文。"""
    ctx = SecurityContext(
        allowed_directories=[temp_dir],
        allowed_commands={"ls", "cat", "echo", "pwd", "python3"},
        enabled=True,
    )
    set_security_context(ctx)
    return ctx


@pytest.fixture
def agent_config():
    """创建一个默认的 AgentConfig。"""
    return AgentConfig(
        model="deepseek-v4-flash",
        max_iterations=3,
        max_context_tokens=4096,
        enable_planning=False,
        short_term_memory_size=5,
    )
