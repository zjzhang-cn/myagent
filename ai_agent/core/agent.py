"""
核心 Agent 循环（ReAct 模式）

Agent 运行流程：
1. 接收用户输入 → 检查是否需要规划
2. ReAct 循环：思考(Think) → 行动(Act) → 观察(Observe) → 反思(Reflect)
3. 返回最终结果
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from ai_agent.config import AgentConfig
from ai_agent.core.memory import LongTermMemory, ShortTermMemory, WorkingMemory
from ai_agent.core.planner import Plan, Planner, StepStatus
from ai_agent.llm.base import BaseLLM, LLMResponse
from ai_agent.tools.registry import ToolRegistry
from ai_agent.utils.security import SecurityContext, set_security_context, clear_security_context
from ai_agent.utils.token_utils import estimate_message_tokens, estimate_messages_tokens, truncate_text

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    """Agent 单次循环记录"""
    iteration: int
    state: AgentState
    thought: str = ""
    action: str = ""
    observation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    plan: Plan | None = None
    iterations: int = 0
    elapsed_seconds: float = 0.0


class ToolCallParser:
    """解析 LLM 响应中的工具调用（支持多种格式）"""

    @classmethod
    def parse(cls, text: str) -> list[dict]:
        """从文本中提取工具调用列表"""
        tool_calls = []

        # 方法 1: 从 markdown 代码块中提取 JSON
        code_blocks = re.findall(
            r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text
        )
        for block in code_blocks:
            calls = cls._extract_json_tool_calls(block)
            if calls:
                tool_calls.extend(calls)

        if not tool_calls:
            # 方法 2: 从原始文本中提取 JSON tool_call
            tool_calls = cls._extract_json_tool_calls(text)

        # 方法 3: 函数调用风格（如果前两种都没匹配到）
        if not tool_calls:
            tool_calls = cls._parse_function_style(text)

        return tool_calls

    @classmethod
    def _extract_json_tool_calls(cls, text: str) -> list[dict]:
        """从文本中提取 JSON 格式的 tool_call"""
        tool_calls = []
        for json_obj in cls._find_json_objects(text, key_hint="tool_call"):
            try:
                data = json.loads(json_obj)
                tc = data.get("tool_call", data)
                # 如果 tool_call 是数组
                if isinstance(tc, list):
                    for item in tc:
                        if isinstance(item, dict) and "name" in item:
                            tool_calls.append(cls._normalize(item))
                elif isinstance(tc, dict) and "name" in tc:
                    tool_calls.append(cls._normalize(tc))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return tool_calls

    @classmethod
    def _find_json_objects(cls, text: str, key_hint: str = "") -> list[str]:
        """在文本中定位完整的 JSON 对象（处理嵌套括号）"""
        results = []
        idx = 0
        while True:
            brace_start = text.find("{", idx)
            if brace_start == -1:
                break

            # 快速检查：附近是否包含 key_hint
            search_end = min(brace_start + 800, len(text))
            if key_hint and key_hint not in text[brace_start:search_end]:
                idx = brace_start + 1
                continue

            # 用计数器匹配完整的 JSON 对象
            depth = 0
            in_string = False
            escape = False
            for i in range(brace_start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[brace_start:i + 1]
                        # 确保是有效 JSON
                        try:
                            json.loads(candidate)
                            results.append(candidate)
                        except json.JSONDecodeError:
                            pass
                        idx = i + 1
                        break
            else:
                idx = brace_start + 1

        return results

    @classmethod
    def _parse_function_style(cls, text: str) -> list[dict]:
        """解析 '调用工具: tool_name(arg1=val1)' 风格的文本"""
        pattern = re.compile(
            r'(?:调用|calling|使用|执行)\s*(?:工具\s*)?[：:]\s*(\w+)\s*\(\s*([^)]*)\s*\)',
        )
        tool_calls = []
        for match in pattern.finditer(text):
            name = match.group(1)
            args_str = match.group(2)
            args = {}
            for arg_match in re.finditer(
                r'(\w+)\s*=\s*("[^"]*"|\'[^\']*\'|\S+)', args_str
            ):
                key = arg_match.group(1)
                val = arg_match.group(2).strip('"\'')
                args[key] = val
            tool_calls.append({"name": name, "arguments": args})
        return tool_calls

    @staticmethod
    def _normalize(tc: dict) -> dict:
        """标准化工具调用格式"""
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        return {"name": tc["name"], "arguments": args}


class Agent:
    """AI Agent 主类

    使用示例:
        agent = Agent(config=AgentConfig())
        result = agent.run("帮我搜索一下今天的天气，然后保存到文件")
        print(result.answer)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: BaseLLM | None = None,
        tool_registry: ToolRegistry | None = None,
        on_step: Callable[[str, dict], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ):
        """
        Args:
            config: Agent 配置
            llm: LLM 实例
            tool_registry: 工具注册表
            on_step: 步骤回调 (event_type, data) -> None
                     event_type: "planning" | "thinking" | "acting" | "observing" | "done" | "token"
            on_token: 流式 token 回调 (token: str) -> None
                     设置后将使用流式模式，在 LLM 推理时逐 token 推送
            on_thinking: 流式推理回调 (thinking: str) -> None
                     当模型支持 think 时，逐 token 推送推理内容
        """
        self.config = config or AgentConfig()

        # LLM
        if llm is None:
            from ai_agent.llm.ollama import OllamaLLM
            llm = OllamaLLM(
                model=self.config.model,
                host=self.config.ollama_host,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        self.llm = llm

        # 工具注册表
        if tool_registry is None:
            from ai_agent.tools.registry import get_registry
            tool_registry = get_registry()
        self.tool_registry = tool_registry

        # 记忆系统
        self.short_term = ShortTermMemory(
            max_size=self.config.short_term_memory_size
        )
        self.working = WorkingMemory()
        self.long_term = LongTermMemory(
            db_path=self.config.long_term_memory_path
        )
        self.short_term.set_system_prompt(self.config.system_prompt)

        # 规划器
        self.planner = Planner(
            llm_chat=self._simple_chat,
            tools_description=self.tool_registry.describe_for_prompt(),
        )

        self.state = AgentState.IDLE
        self.current_plan: Plan | None = None
        self._replan_count: int = 0  # 当前任务中已重新规划次数
        self.on_step = on_step
        self.on_token = on_token
        self.on_thinking = on_thinking

    # ----------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------

    def run(self, user_input: str) -> AgentResult:
        """
        执行 Agent 主循环

        Args:
            user_input: 用户输入

        Returns:
            AgentResult: 执行结果
        """
        start_time = time.time()
        steps: list[AgentStep] = []

        # Step 0: 将用户输入加入短期记忆
        self._replan_count = 0  # 重置重新规划计数
        self.short_term.add_user(user_input)
        self.state = AgentState.IDLE

        # 检索长期记忆中的相关知识
        relevant_memories = self.long_term.recall(user_input, limit=3)
        if "没有找到相关记忆" not in relevant_memories:
            logger.info(f"长期记忆检索: {relevant_memories[:200]}")

        # --- 回调：开始处理 ---
        self._emit("start", {"query": user_input})

        # Step 1: 判断是否需要规划
        if self.config.enable_planning and self.planner.should_plan(
            user_input, threshold=self.config.plan_threshold_complexity
        ):
            self.state = AgentState.PLANNING
            logger.info("任务复杂度较高，启动规划...")
            self.current_plan = self.planner.create_plan(user_input)
            self.working.current_task = user_input
            logger.info(f"规划完成: {self.current_plan.total_steps} 个步骤")
            self._emit("planning", {
                "steps": [
                    {"id": s.id, "description": s.description}
                    for s in self.current_plan.steps
                ],
            })

        # Step 2: ReAct 循环
        final_answer = ""
        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"--- ReAct 循环 {iteration}/{self.config.max_iterations} ---")

            # 2a. 思考阶段
            self.state = AgentState.THINKING
            messages = self._build_messages()

            if self.config.verbose:
                plan_info = ""
                if self.current_plan:
                    plan_info = f" | 计划: {self.current_plan.completed_steps}/{self.current_plan.total_steps}"
                logger.info(f"Think{plan_info} | 上下文消息数: {len(messages)}")

            response = self._call_llm(messages)
            step = AgentStep(iteration=iteration, state=AgentState.THINKING)
            step.thought = response.content
            steps.append(step)

            # 2b. 解析工具调用
            tool_calls = response.tool_calls

            # 如果 LLM 原生不支持 tool_calls，用正则解析
            if not tool_calls:
                tool_calls = ToolCallParser.parse(response.content)

            # --- 回调：思考（包含工具调用信息） ---
            self._emit("thinking", {
                "iteration": iteration,
                "content": response.content,
                "tool_calls": [
                    {"name": tc["name"], "arguments": tc.get("arguments", {})}
                    for tc in tool_calls
                ],
            })

            if tool_calls:
                # 2c. 执行工具
                self.state = AgentState.ACTING
                limited_calls = tool_calls[:self.config.max_tool_calls_per_iteration]

                # 收集有效的工具调用信息
                tool_infos: list[dict] = []
                for tc in limited_calls:
                    t_name = tc.get("name", "")
                    t_args = tc.get("arguments", {})
                    if t_name:
                        tool_infos.append({"name": t_name, "arguments": t_args})

                if tool_infos:
                    step.action = "; ".join(
                        f"{ti['name']}({json.dumps(ti['arguments'], ensure_ascii=False)})"
                        for ti in tool_infos
                    )

                    # --- 回调：行动（所有工具） ---
                    for ti in tool_infos:
                        logger.info(f"  Act -> {ti['name']}")
                        self._emit("acting", {
                            "tool": ti["name"],
                            "arguments": ti["arguments"],
                        })

                    # 执行工具（并发或顺序）
                    use_parallel = (
                        self.config.parallel_tool_execution
                        and len(tool_infos) > 1
                    )
                    if use_parallel:
                        logger.info(f"  ⚡ 并发执行 {len(tool_infos)} 个工具...")
                        exec_start = time.time()
                        results = self._execute_tools_parallel(tool_infos)
                        exec_elapsed = time.time() - exec_start
                        logger.info(
                            f"  并发执行完成，耗时 {exec_elapsed:.2f}s "
                            f"(工具数: {len(tool_infos)})"
                        )
                    else:
                        results = self._execute_tools_sequential(tool_infos)

                    # 处理每个工具的执行结果
                    observation_parts = []
                    combined_obs: list[str] = []
                    for ti, result in zip(tool_infos, results):
                        t_name = ti["name"]
                        t_args = ti["arguments"]

                        # 日志
                        logger.debug(
                            f"[工具执行] {t_name}({json.dumps(t_args, ensure_ascii=False)}) "
                            f"-> {result[:300]!r}"
                        )

                        # 观察阶段
                        self.state = AgentState.OBSERVING
                        observation_parts.append(
                            f"[工具: {t_name}]\n"
                            f"输入: {json.dumps(t_args, ensure_ascii=False)}\n"
                            f"输出: {result}"
                        )
                        combined_obs.append(f"{t_name}: {result[:200]}")

                        # --- 回调：观察 ---
                        self._emit("observing", {
                            "tool": t_name,
                            "result": result,
                        })

                        # 记录到记忆（截断过长结果）
                        truncated_result = truncate_text(
                            result, self.config.max_tool_result_chars
                        )
                        self.short_term.add_tool_result(t_name, truncated_result)

                        # 更新计划步骤（检测失败）
                        if self.current_plan:
                            next_step = self.current_plan.get_next_step()
                            if next_step:
                                matched = (
                                    t_name == next_step.tool_hint or
                                    next_step.description.lower() in result.lower()
                                )
                                if matched or not next_step.tool_hint:
                                    is_error = self._is_error_result(result)
                                    if is_error:
                                        error_cat = self._categorize_error(result)
                                        logger.warning(
                                            f"步骤 {next_step.id} 执行失败 "
                                            f"({error_cat}): {result[:100]}"
                                        )
                                        self.current_plan.mark_step(
                                            next_step.id, StepStatus.FAILED, result
                                        )
                                    else:
                                        self.current_plan.mark_step(
                                            next_step.id, StepStatus.COMPLETED, result
                                        )

                        # 工作记忆中记录
                        self.working.set(f"_last_tool_{t_name}", result)
                        self.working.add_step_result(
                            step_name=t_name, result=result
                        )

                    step.observation = " | ".join(combined_obs)

                # 将工具结果反馈给 LLM
                observation_text = "\n\n".join(observation_parts)
                self.short_term.add_assistant(f"[工具执行结果]\n{observation_text}")

            else:
                # 没有工具调用，视为最终答案
                self.state = AgentState.DONE
                final_answer = response.content
                self.short_term.add_assistant(final_answer)
                step.state = AgentState.DONE
                self._emit("done", {"answer": final_answer})
                break

            # 检查是否需要重新规划（有失败步骤且未达到次数上限）
            if (self.current_plan and
                self.current_plan.failed_steps > 0 and
                self._replan_count < self.config.max_replan_attempts):

                failed_info = self._collect_failed_steps_info()
                logger.warning(
                    f"检测到 {self.current_plan.failed_steps} 个失败步骤，"
                    f"尝试重新规划 (第 {self._replan_count + 1}/{self.config.max_replan_attempts} 次)"
                )

                try:
                    new_plan = self.planner.replan(
                        self.current_plan, failed_info
                    )
                    # 保留已完成步骤的状态
                    for old_step in self.current_plan.steps:
                        if old_step.status == StepStatus.COMPLETED:
                            # 在新计划中标记对应步骤为已完成
                            for new_step in new_plan.steps:
                                if (old_step.description[:30] in new_step.description or
                                    new_step.description[:30] in old_step.description):
                                    new_step.status = StepStatus.COMPLETED
                                    new_step.result = old_step.result

                    self.current_plan = new_plan
                    self._replan_count += 1
                    self._emit("replanning", {
                        "attempt": self._replan_count,
                        "failed_count": failed_info.count("失败"),
                        "new_total_steps": new_plan.total_steps,
                    })
                    logger.info(
                        f"重新规划完成: {new_plan.total_steps} 个步骤 "
                        f"(已完成: {new_plan.completed_steps}, 失败: {new_plan.failed_steps})"
                    )

                    # 将重新规划信息注入短期记忆，帮助 LLM 理解上下文变化
                    replan_msg = (
                        f"[系统通知] 检测到计划执行中有步骤失败，已自动重新规划。\n"
                        f"新计划共 {new_plan.total_steps} 个步骤。\n"
                        f"已完成: {new_plan.completed_steps}, 待执行: {new_plan.total_steps - new_plan.completed_steps}"
                    )
                    self.short_term.add_assistant(replan_msg)

                except Exception as e:
                    logger.error(f"重新规划失败: {e}，继续使用当前计划")

            # 检查计划是否全部完成
            if self.current_plan and self.current_plan.get_next_step() is None:
                self.state = AgentState.DONE
                if self.current_plan.failed_steps > 0:
                    # 有失败步骤，生成包含失败信息的总结
                    summary_messages = self._build_summary_prompt()
                    summary_messages[0]["content"] += (
                        f"\n注意：有 {self.current_plan.failed_steps} 个步骤执行失败。"
                        f"请在总结中说明哪些步骤已完成、哪些失败，并给出建议。"
                    )
                else:
                    summary_messages = self._build_summary_prompt()
                summary_response = self._call_llm(summary_messages)
                final_answer = summary_response.content
                self.short_term.add_assistant(final_answer)
                self._emit("done", {"answer": final_answer})
                break

        # 如果达到最大迭代次数还未结束
        if self.state != AgentState.DONE:
            final_answer = self._force_summary()
            self.state = AgentState.DONE

        elapsed = time.time() - start_time

        # 保存重要内容到长期记忆
        self._save_to_long_term(user_input, final_answer)

        return AgentResult(
            success=True,
            answer=final_answer,
            steps=steps,
            plan=self.current_plan,
            iterations=len(steps),
            elapsed_seconds=elapsed,
        )

    def _emit(self, event: str, data: dict) -> None:
        """触发步骤回调"""
        if self.on_step:
            try:
                self.on_step(event, data)
            except Exception as e:
                logger.debug(f"回调异常: {e}")

    def add_tool(self, func: Callable) -> None:
        """添加自定义工具（被 @tool 装饰的函数）"""
        self.tool_registry.register_function(func)
        self.planner._tools_desc = self.tool_registry.describe_for_prompt()

    def add_tools(self, *functions: Callable) -> None:
        """批量添加工具"""
        for func in functions:
            self.add_tool(func)

    def clear_memory(self) -> None:
        """清除短期和工作记忆"""
        self.short_term.clear()
        self.working.reset()
        self.current_plan = None

    def reset(self) -> None:
        """完全重置 Agent"""
        self.clear_memory()
        self.state = AgentState.IDLE

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _build_messages(self) -> list[dict]:
        """构建发送给 LLM 的消息列表（含上下文窗口裁剪）"""
        messages = []

        # 系统提示词
        system_parts = [self.config.system_prompt]

        # 注入工具描述
        system_parts.append("\n" + self.tool_registry.describe_for_prompt())
        system_parts.append(
            "\n重要：当你需要调用工具时，请使用以下 JSON 格式：\n"
            '{"tool_call": {"name": "工具名", "arguments": {"参数": "值"}}}\n'
            "如果不需要调用工具，直接回复用户即可。"
        )

        # 注入计划信息
        if self.current_plan:
            plan_text = self.current_plan.format_for_prompt()
            system_parts.append(f"\n当前任务计划：\n{plan_text}")
            next_step = self.current_plan.get_next_step()
            if next_step:
                system_parts.append(
                    f"\n当前步骤：执行 Step {next_step.id}: {next_step.description}"
                )

        messages.append({
            "role": "system",
            "content": "\n".join(system_parts),
        })

        # 追加对话历史
        for msg in self.short_term.get_recent():
            messages.append(msg.to_dict())

        logger.debug(
            f"[构建消息] 共 {len(messages)} 条，"
            f"system_prompt 长度={len(messages[0]['content'])}, "
            f"plan={'有' if self.current_plan else '无'}"
        )

        # 上下文窗口裁剪
        if self.config.max_context_tokens:
            messages = self._trim_messages(messages)

        return messages

    def _trim_messages(self, messages: list[dict]) -> list[dict]:
        """
        裁剪消息列表以适应上下文窗口。

        策略：
        1. 始终保留 system prompt（第一条消息）
        2. 截断过长的工具结果
        3. 如果仍超出预算，从旧到新移除消息，但保留第一条用户消息
        4. 日志记录裁剪操作
        """
        budget = self.config.max_context_tokens
        total = estimate_messages_tokens(messages)

        if total <= budget:
            return messages

        logger.warning(f"⚠️ 上下文超出预算 ({total}/{budget} tokens)，开始裁剪...")
        original_count = len(messages)

        # 预留给 LLM 响应的空间（保守估计 30%）
        effective_budget = int(budget * 0.7)

        system_msg = messages[0]
        body = messages[1:]

        if not body:
            logger.warning("仅剩系统消息，无法进一步裁剪")
            return messages

        # Step 1: 截断过长的工具结果
        max_tool_chars = self.config.max_tool_result_chars
        truncated_count = 0
        for i, msg in enumerate(body):
            if msg.get("role") == "tool" and len(msg.get("content", "")) > max_tool_chars:
                body[i] = {
                    "role": "tool",
                    "content": truncate_text(msg["content"], max_tool_chars),
                }
                truncated_count += 1

        if truncated_count:
            total = estimate_messages_tokens([system_msg] + body)
            if total <= effective_budget:
                logger.info(
                    f"截断 {truncated_count} 条工具结果后 tokens: {total}/{effective_budget}，"
                    f"消息数: {len(body) + 1}"
                )
                return [system_msg] + body

        # Step 2: 从旧到新裁剪消息（保留第一条用户消息）
        # 找到第一条用户消息的位置
        first_user_idx = None
        for i, msg in enumerate(body):
            if msg.get("role") == "user":
                first_user_idx = i
                break

        # 从最新的消息开始，向前收集直到预算允许
        kept_body = []
        current_tokens = estimate_message_tokens(system_msg)

        # 始终保留第一条用户消息（如果存在）
        first_user_msg = None
        if first_user_idx is not None:
            first_user_msg = body[first_user_idx]
            current_tokens += estimate_message_tokens(first_user_msg)

        # 从后往前收集消息
        for i in range(len(body) - 1, -1, -1):
            # 跳过第一条用户消息（已经计入）
            if first_user_idx is not None and i == first_user_idx:
                continue

            msg_tokens = estimate_message_tokens(body[i])
            if current_tokens + msg_tokens <= effective_budget:
                kept_body.insert(0, body[i])
                current_tokens += msg_tokens
            else:
                break

        # 在开头插入第一条用户消息
        if first_user_msg is not None:
            kept_body.insert(0, first_user_msg)

        final_messages = [system_msg] + kept_body
        final_total = estimate_messages_tokens(final_messages)

        removed = original_count - len(final_messages)
        logger.warning(
            f"上下文裁剪完成: {original_count} → {len(final_messages)} 条消息 "
            f"({total} → {final_total} tokens), 移除了 {removed} 条早期消息"
        )

        return final_messages

    def _build_summary_prompt(self) -> list[dict]:
        """构建总结提示"""
        return [
            {
                "role": "system",
                "content": (
                    "所有计划步骤已完成。请根据工具执行结果，生成一个清晰、完整的总结回复。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户原始请求: {self.working.current_task}\n\n"
                    f"已完成步骤:\n" +
                    "\n".join(
                        f"- {sr['step']}: {sr['result'][:300]}"
                        for sr in self.working.get_step_results()
                    ) +
                    "\n\n请总结回复用户。"
                ),
            },
        ]

    def _force_summary(self) -> str:
        """达到最大迭代次数时强制生成总结"""
        logger.warning(f"达到最大迭代次数 {self.config.max_iterations}，强制结束")
        try:
            messages = [
                {
                    "role": "system",
                    "content": "请根据已获得的信息，给用户一个简洁的总结回复。",
                },
                {"role": "user", "content": "请总结当前任务的进展情况。"},
            ]
            response = self._call_llm(messages)
            return response.content or "任务已达到最大执行轮次，部分步骤可能未完成。"
        except Exception:
            return "任务已达到最大执行轮次，部分步骤可能未完成。请尝试简化请求后重试。"

    def _call_llm(self, messages: list[dict]) -> LLMResponse:
        """调用 LLM（非流式），带重试。如果设置了 on_token 则自动切换流式"""
        if self.on_token:
            return self._call_llm_stream(messages)

        # 安全检查：裁剪过长消息
        if self.config.max_context_tokens:
            messages = self._trim_messages(messages)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                tools = self.tool_registry.to_ollama_schemas()
                response = self.llm.chat(messages, tools=tools if tools else None)

                if response.content or response.has_tool_calls():
                    return response

                if attempt < max_retries - 1:
                    logger.warning(f"LLM 返回空响应，重试 {attempt + 2}/{max_retries}")
                    time.sleep(1)
                else:
                    return response
            except Exception as e:
                logger.error(f"LLM 调用失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return LLMResponse(content=f"抱歉，LLM 调用失败: {e}")

        return LLMResponse(content="抱歉，暂时无法处理你的请求。")

    def _call_llm_stream(self, messages: list[dict]) -> LLMResponse:
        """流式调用 LLM，逐 token 推送"""
        # 安全检查：裁剪过长消息
        if self.config.max_context_tokens:
            messages = self._trim_messages(messages)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                tools = self.tool_registry.to_ollama_schemas()
                full_content = ""
                full_thinking = ""
                final_tool_calls: list[dict] = []
                final_usage: dict = {}

                for event in self.llm.chat_stream(messages, tools=tools if tools else None):
                    if event.type == "thinking":
                        full_thinking += event.content
                        if self.on_thinking:
                            self.on_thinking(event.content)
                        self._emit("thinking_token", {"content": event.content})
                    elif event.type == "token":
                        full_content += event.content
                        if self.on_token:
                            self.on_token(event.content)
                        self._emit("token", {"content": event.content})
                    elif event.type == "done":
                        full_content = event.content or full_content
                        full_thinking = event.thinking or full_thinking
                        final_tool_calls = event.tool_calls
                        final_usage = event.usage

                if full_content or full_thinking or final_tool_calls:
                    return LLMResponse(
                        content=full_content,
                        thinking=full_thinking,
                        tool_calls=final_tool_calls,
                        usage=final_usage,
                    )

                if attempt < max_retries - 1:
                    logger.warning(f"LLM 流式返回空响应，重试 {attempt + 2}/{max_retries}")
                    time.sleep(1)
                else:
                    return LLMResponse(content=full_content, tool_calls=final_tool_calls)

            except Exception as e:
                logger.error(f"LLM 流式调用失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return LLMResponse(content=f"抱歉，LLM 调用失败: {e}")

        return LLMResponse(content="抱歉，暂时无法处理你的请求。")

    def _simple_chat(self, messages: list[dict]) -> str:
        """简化的聊天接口（给 Planner 用的）"""
        response = self.llm.chat(messages, tools=None)
        return response.content

    def _save_to_long_term(self, user_input: str, answer: str) -> None:
        """保存重要信息到长期记忆"""
        try:
            if len(user_input) > 30:
                self.long_term.remember(
                    f"用户曾问过: {user_input[:200]}",
                    importance=2,
                )
            for sr in self.working.get_step_results():
                result_text = sr.get("result", "")
                if len(result_text) > 50:
                    self.long_term.remember(
                        f"任务 '{self.working.current_task[:100]}' 的结果: {result_text[:200]}",
                        importance=1,
                    )
        except Exception as e:
            logger.debug(f"保存长期记忆失败: {e}")

    # ----------------------------------------------------------
    # 工具执行方法
    # ----------------------------------------------------------

    def _build_security_context(self) -> SecurityContext:
        """根据当前配置构建安全上下文"""
        from ai_agent.utils.security import get_allowed_directories
        return SecurityContext(
            allowed_directories=get_allowed_directories(
                self.config.workspace_directories
            ),
            allowed_commands=self.config.shell_allowed_commands,
            allow_all_commands=self.config.shell_allow_all_commands,
            enabled=True,
        )

    def _execute_tools_sequential(self, tool_infos: list[dict]) -> list[str]:
        """顺序执行工具，返回结果列表（顺序与输入一致）"""
        ctx = self._build_security_context()
        results = []
        for ti in tool_infos:
            set_security_context(ctx)
            try:
                result = self.tool_registry.execute(ti["name"], ti["arguments"])
            except Exception as e:
                result = f"工具执行异常: {e}"
                logger.error(f"工具 {ti['name']} 执行失败: {e}")
            results.append(result)
        clear_security_context()
        return results

    def _execute_tools_parallel(self, tool_infos: list[dict]) -> list[str]:
        """并发执行多个工具调用，返回结果列表（顺序与输入一致）

        使用 ThreadPoolExecutor 并发执行，每个工具在独立线程中运行。
        单个工具失败不影响其他工具的执行。
        """
        ctx = self._build_security_context()
        max_workers = min(len(tool_infos), self.config.max_parallel_tools)
        results: dict[int, str] = {}  # index -> result

        def _run_one(index: int, ti: dict) -> tuple[int, str]:
            """在独立线程中执行单个工具"""
            set_security_context(ctx)
            name = ti["name"]
            args = ti["arguments"]
            try:
                result = self.tool_registry.execute(name, args)
                return index, result
            except Exception as e:
                logger.error(f"工具 {name} 执行失败: {e}")
                return index, f"工具执行异常: {e}"

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_one, i, ti): i
                for i, ti in enumerate(tool_infos)
            }
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"工具 {tool_infos[idx]['name']} 线程异常: {e}")
                    results[idx] = f"工具执行异常: {e}"

        # 按原始顺序返回
        return [results.get(i, "执行结果缺失") for i in range(len(tool_infos))]

    # ----------------------------------------------------------
    # 错误检测与恢复
    # ----------------------------------------------------------

    @staticmethod
    def _is_error_result(result: str) -> bool:
        """判断工具执行结果是否为错误"""
        error_patterns = [
            "错误：", "错误:",
            "参数错误", "参数异常",
            "工具执行异常",
            "执行异常",
            "权限不足", "Permission denied",
            "未找到工具",
            "命令不存在",
            "文件不存在",
            "操作不允许",
        ]
        return any(pattern in result for pattern in error_patterns)

    @staticmethod
    def _categorize_error(result: str) -> str:
        """对错误进行分类，返回错误类别标签"""
        if "未找到工具" in result:
            return "tool_not_found"
        if "超时" in result or "timeout" in result.lower():
            return "timeout"
        if "参数错误" in result or "参数异常" in result:
            return "parameter_error"
        if "权限" in result or "Permission" in result or "操作不允许" in result:
            return "permission_error"
        if "文件不存在" in result:
            return "not_found"
        if "命令不存在" in result:
            return "command_not_found"
        if "工具执行异常" in result or "执行异常" in result:
            return "execution_error"
        return "unknown_error"

    def _collect_failed_steps_info(self) -> str:
        """收集失败步骤的详细信息，用于重新规划"""
        if not self.current_plan:
            return "无失败步骤"

        lines = ["以下步骤执行失败，需要调整计划："]
        for step in self.current_plan.steps:
            if step.status == StepStatus.FAILED:
                error_cat = self._categorize_error(step.result)
                lines.append(
                    f"- Step {step.id}: {step.description}\n"
                    f"  错误类型: {error_cat}\n"
                    f"  错误详情: {step.result[:300]}"
                )

        # 也包含已完成步骤的简要信息
        completed = [s for s in self.current_plan.steps
                     if s.status == StepStatus.COMPLETED]
        if completed:
            lines.append("\n已完成的步骤：")
            for step in completed:
                lines.append(
                    f"- Step {step.id}: {step.description} [已完成]"
                )

        return "\n".join(lines)
