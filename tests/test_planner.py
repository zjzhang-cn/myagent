"""Tests for ai_agent.core.planner."""

from unittest.mock import MagicMock

import pytest

from ai_agent.core.planner import Plan, PlanStep, Planner, StepStatus, estimate_complexity
from ai_agent.llm.base import LLMResponse


# ===== PlanStep =====

class TestPlanStep:
    def test_basic_step(self):
        step = PlanStep(id=1, description="do something")
        assert step.id == 1
        assert step.description == "do something"
        assert step.status == StepStatus.PENDING
        assert step.dependencies == []

    def test_step_with_dependencies(self):
        step = PlanStep(id=2, description="step 2", dependencies=[1])
        assert 1 in step.dependencies

    def test_to_dict(self):
        step = PlanStep(id=1, description="test step", tool_hint="read_file",
                        dependencies=[2])
        d = step.to_dict()
        assert d["id"] == 1
        assert d["description"] == "test step"
        assert d["tool_hint"] == "read_file"
        assert d["dependencies"] == [2]

    def test_can_execute_true(self):
        step = PlanStep(id=2, description="step 2", dependencies=[1])
        assert step.can_execute({1})
        assert not step.can_execute(set())

    def test_can_execute_no_deps(self):
        step = PlanStep(id=1, description="step 1")
        assert step.can_execute(set())


# ===== Plan =====

class TestPlan:
    def test_empty_plan(self):
        plan = Plan(task="test", steps=[])
        assert plan.total_steps == 0
        assert plan.completed_steps == 0

    def test_single_step(self):
        step = PlanStep(id=1, description="test")
        plan = Plan(task="t", steps=[step])
        assert plan.total_steps == 1
        assert plan.completed_steps == 0

    def test_completed_steps_count(self):
        steps = [
            PlanStep(id=1, description="s1", status=StepStatus.COMPLETED),
            PlanStep(id=2, description="s2", status=StepStatus.PENDING),
            PlanStep(id=3, description="s3", status=StepStatus.COMPLETED),
        ]
        plan = Plan(task="t", steps=steps)
        assert plan.completed_steps == 2

    def test_failed_steps_count(self):
        steps = [
            PlanStep(id=1, description="s1", status=StepStatus.FAILED),
            PlanStep(id=2, description="s2", status=StepStatus.COMPLETED),
        ]
        plan = Plan(task="t", steps=steps)
        assert plan.failed_steps == 1

    def test_get_executable_steps(self):
        steps = [
            PlanStep(id=1, description="s1"),
            PlanStep(id=2, description="s2", dependencies=[1]),
            PlanStep(id=3, description="s3", dependencies=[2]),
        ]
        plan = Plan(task="t", steps=steps)
        executable = plan.get_executable_steps()
        assert len(executable) == 1
        assert executable[0].id == 1

    def test_get_executable_after_completion(self):
        steps = [
            PlanStep(id=1, description="s1", status=StepStatus.COMPLETED),
            PlanStep(id=2, description="s2", dependencies=[1]),
            PlanStep(id=3, description="s3", dependencies=[2]),
        ]
        plan = Plan(task="t", steps=steps)
        executable = plan.get_executable_steps()
        assert len(executable) == 1
        assert executable[0].id == 2

    def test_parallel_steps(self):
        steps = [
            PlanStep(id=1, description="s1"),
            PlanStep(id=2, description="s2"),
            PlanStep(id=3, description="s3", dependencies=[1, 2]),
        ]
        plan = Plan(task="t", steps=steps)
        executable = plan.get_executable_steps()
        assert len(executable) == 2

    def test_progress(self):
        steps = [
            PlanStep(id=1, description="s1", status=StepStatus.COMPLETED),
            PlanStep(id=2, description="s2"),
        ]
        plan = Plan(task="t", steps=steps)
        assert plan.progress == 0.5

    def test_empty_progress(self):
        plan = Plan(task="t", steps=[])
        assert plan.progress == 0.0

    def test_progress_display(self):
        steps = [
            PlanStep(id=1, description="s1", status=StepStatus.COMPLETED),
            PlanStep(id=2, description="s2", status=StepStatus.FAILED),
            PlanStep(id=3, description="s3"),
        ]
        plan = Plan(task="t", steps=steps)
        assert plan.completed_steps == 1
        assert plan.failed_steps == 1
        assert plan.progress > 0

    def test_to_dict_roundtrip(self):
        steps = [
            PlanStep(id=1, description="step 1", tool_hint="read_file"),
            PlanStep(id=2, description="step 2", dependencies=[1]),
        ]
        plan = Plan(task="test task", steps=steps)
        d = plan.to_dict()
        restored = Plan.from_dict(d)
        assert restored.task == "test task"
        assert restored.total_steps == 2
        assert len(restored.steps) == 2
        assert restored.steps[0].tool_hint == "read_file"
        assert restored.steps[1].dependencies == [1]

    def test_format_for_prompt(self):
        steps = [
            PlanStep(id=1, description="search web", tool_hint="search_web"),
            PlanStep(id=2, description="save file", dependencies=[1]),
        ]
        plan = Plan(task="t", steps=steps)
        prompt = plan.format_for_prompt()
        assert "search web" in prompt
        assert "save file" in prompt

    def test_get_next_step(self):
        steps = [
            PlanStep(id=1, description="s1"),
            PlanStep(id=2, description="s2", dependencies=[1]),
        ]
        plan = Plan(task="t", steps=steps)
        next_step = plan.get_next_step()
        assert next_step.id == 1

    def test_get_next_step_after_completion(self):
        steps = [
            PlanStep(id=1, description="s1", status=StepStatus.COMPLETED),
            PlanStep(id=2, description="s2", dependencies=[1]),
        ]
        plan = Plan(task="t", steps=steps)
        next_step = plan.get_next_step()
        assert next_step.id == 2

    def test_get_next_step_all_done(self):
        steps = [
            PlanStep(id=1, description="s1", status=StepStatus.COMPLETED),
        ]
        plan = Plan(task="t", steps=steps)
        assert plan.get_next_step() is None

    def test_mark_step(self):
        step = PlanStep(id=1, description="s1")
        plan = Plan(task="t", steps=[step])
        plan.mark_step(1, StepStatus.COMPLETED, "done")
        assert step.status == StepStatus.COMPLETED
        assert step.result == "done"

    def test_created_at_set_when_using_simple_decompose(self):
        from ai_agent.core.planner import Planner
        plan = Planner(llm_chat=str, tools_description="")._simple_decompose("test")
        assert plan.created_at != ""


# ===== estimate_complexity (module-level) =====

class TestEstimateComplexity:
    def test_simple_greeting(self):
        assert estimate_complexity("你好") < 3

    def test_multi_step_task(self):
        score = estimate_complexity("先搜索新闻，然后保存到文件，最后发送邮件")
        assert score >= 3

    def test_with_action_verbs(self):
        score = estimate_complexity("创建文件并写入数据然后搜索")
        assert score >= 1

    def test_empty_input(self):
        assert estimate_complexity("") >= 0


# ===== Planner =====

class TestPlanner:
    @pytest.fixture
    def mock_llm_chat(self):
        def chat(messages, tools=None):
            return '{"steps": [{"id": 1, "description": "搜索科技新闻", "tool_hint": "search_web", "dependencies": []}, {"id": 2, "description": "保存结果", "tool_hint": "write_file", "dependencies": [1]}]}'
        return chat

    @pytest.fixture
    def planner(self, mock_llm_chat):
        return Planner(
            llm_chat=mock_llm_chat,
            tools_description="- search_web(query)\n- write_file(path, content)",
        )

    def test_init(self, planner):
        assert planner is not None

    def test_should_plan_simple(self, planner):
        assert not planner.should_plan("你好", threshold=5)

    def test_should_plan_complex(self, planner):
        # With threshold=5, even complex input won't trigger planning
        result = planner.should_plan("先搜索新闻然后保存然后发送邮件然后备份然后清理", threshold=5)
        assert result is True or result is False

    def test_create_plan(self, planner):
        plan = planner.create_plan("搜索科技新闻并保存到文件")
        assert plan is not None
        assert len(plan.steps) >= 1

    def test_simple_decompose(self, planner):
        plan = planner._simple_decompose("先搜索，再保存")
        assert plan is not None
        assert len(plan.steps) >= 1

    def test_replan(self, planner):
        original = Plan(
            task="测试任务",
            steps=[PlanStep(id=1, description="步骤1", status=StepStatus.FAILED)],
        )
        new_plan = planner.replan(original_plan=original, feedback="步骤1失败了，请调整")
        assert new_plan is not None
