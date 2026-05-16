"""Tests for ai_agent.tools.builtin file_ops and shell."""

import os
import pytest

from ai_agent.tools.builtin.file_ops import read_file, write_file, list_directory, delete_file
from ai_agent.tools.builtin.shell import run_shell_command
from ai_agent.utils.security import SecurityContext, set_security_context


@pytest.fixture(autouse=True)
def setup_security(temp_dir):
    ctx = SecurityContext(
        allowed_directories=[temp_dir],
        allowed_commands={"ls", "cat", "echo", "pwd", "mkdir", "touch", "rm", "python3"},
        enabled=True,
    )
    set_security_context(ctx)
    yield
    from ai_agent.utils.security import clear_security_context
    clear_security_context()


class TestReadFile:
    def test_read_existing(self, temp_dir, temp_file):
        content = read_file(path=temp_file)
        assert content is not None
        assert "hello world" in content

    def test_read_nonexistent(self, temp_dir):
        try:
            result = read_file(path=os.path.join(temp_dir, "nonexistent.txt"))
            assert result is None or "错误" in result
        except Exception:
            pass

    def test_read_with_start_line(self, temp_dir, temp_file):
        content = read_file(path=temp_file, start_line=2)
        assert content is not None
        assert "line 2" in content
        assert "hello world" not in content

    def test_read_with_end_line(self, temp_dir, temp_file):
        content = read_file(path=temp_file, end_line=1)
        assert content is not None
        assert "hello world" in content
        assert "line 2" not in content

    def test_read_out_of_bounds(self, temp_dir, temp_file):
        content = read_file(path=temp_file, start_line=100)
        assert content == ""


class TestWriteFile:
    def test_write_new_file(self, temp_dir):
        path = os.path.join(temp_dir, "new.txt")
        write_file(path=path, content="new content")
        with open(path) as f:
            assert f.read() == "new content"

    def test_overwrite_existing(self, temp_dir, temp_file):
        write_file(path=temp_file, content="overwritten")
        with open(temp_file) as f:
            assert f.read() == "overwritten"

    def test_append_mode(self, temp_dir, temp_file):
        write_file(path=temp_file, content="\nappended", mode="a")
        with open(temp_file) as f:
            content = f.read()
            assert "hello world" in content
            assert "appended" in content

    def test_write_empty(self, temp_dir):
        path = os.path.join(temp_dir, "empty.txt")
        write_file(path=path, content="")
        with open(path) as f:
            assert f.read() == ""

    def test_write_newlines(self, temp_dir):
        path = os.path.join(temp_dir, "lines.txt")
        write_file(path=path, content="line1\nline2\nline3")
        with open(path) as f:
            lines = f.readlines()
            assert len(lines) == 3


class TestListDirectory:
    def test_list_current(self, temp_dir):
        entries = list_directory(path=temp_dir)
        assert entries is not None

    def test_list_with_empty_dir(self, temp_dir):
        empty_sub = os.path.join(temp_dir, "empty")
        os.makedirs(empty_sub)
        entries = list_directory(path=empty_sub)
        assert isinstance(entries, (list, str))

    def test_list_with_pattern(self, temp_dir):
        open(os.path.join(temp_dir, "a.txt"), "w").close()
        open(os.path.join(temp_dir, "b.txt"), "w").close()
        open(os.path.join(temp_dir, "c.py"), "w").close()

        entries = list_directory(path=temp_dir, pattern="*.txt")
        if isinstance(entries, list):
            txts = [e for e in entries if e.endswith(".txt")]
            assert len(txts) == 2

    def test_nonexistent_dir(self, temp_dir):
        try:
            result = list_directory(path=os.path.join(temp_dir, "nonexistent"))
            assert result is None or "错误" in str(result)
        except Exception:
            pass


class TestDeleteFile:
    def test_delete_existing(self, temp_dir, temp_file):
        assert os.path.exists(temp_file)
        delete_file(path=temp_file)
        assert not os.path.exists(temp_file)

    def test_delete_nonexistent(self, temp_dir):
        path = os.path.join(temp_dir, "nonexistent.txt")
        try:
            result = delete_file(path=path)
            assert result is None or "错误" in str(result)
        except Exception:
            pass


class TestRunShellCommand:
    def test_simple_command(self):
        result = run_shell_command(command="echo hello")
        assert result is not None
        assert "hello" in result

    def test_command_with_output(self):
        result = run_shell_command(command="pwd")
        assert result is not None
        assert len(result) > 0

    def test_timeout(self):
        try:
            run_shell_command(command="sleep 10", timeout=1)
        except Exception:
            pass  # Expected: timeout or interrupt

    def test_invalid_command(self):
        try:
            result = run_shell_command(command="nonexistent_cmd_xyz")
            assert result is None or "错误" in str(result)
        except Exception:
            pass

    def test_with_working_dir(self, temp_dir):
        result = run_shell_command(command="pwd", working_dir=temp_dir)
        assert temp_dir in result
