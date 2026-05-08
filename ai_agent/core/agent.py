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
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from ai_agent.config import AgentConfig
from ai_agent.core.memory import LongTermMemory, ShortTermMemory, WorkingMemory
from ai_agent.core.planner import Plan, Planner, StepStatus
from ai_agent.llm.base import BaseLLM, LLMResponse
from ai_agent.tools.registry import ToolRegistry

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
        self.on_step = on_step
        self.on_token = on_token

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
                observation_parts = []

                for tc in tool_calls[:self.config.max_tool_calls_per_iteration]:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})

                    if not tool_name:
                        continue

                    step.action = f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})"
                    logger.info(f"  Act -> {tool_name}")

                    # --- 回调：行动 ---
                    self._emit("acting", {
                        "tool": tool_name,
                        "arguments": tool_args,
                    })

                    # 执行工具
                    result = self.tool_registry.execute(tool_name, tool_args)

                    # 2d. 观察阶段
                    self.state = AgentState.OBSERVING
                    observation_parts.append(
                        f"[工具: {tool_name}]\n输入: {json.dumps(tool_args, ensure_ascii=False)}\n输出: {result}"
                    )
                    step.observation = result

                    # --- 回调：观察 ---
                    self._emit("observing", {
                        "tool": tool_name,
                        "result": result,
                    })

                    # 记录到记忆
                    self.short_term.add_tool_result(tool_name, result)

                    # 更新计划步骤
                    if self.current_plan:
                        next_step = self.current_plan.get_next_step()
                        if next_step:
                            matched = (
                                tool_name == next_step.tool_hint or
                                next_step.description.lower() in result.lower()
                            )
                            if matched or not next_step.tool_hint:
                                self.current_plan.mark_step(
                                    next_step.id, StepStatus.COMPLETED, result
                                )

                    # 工作记忆中记录
                    self.working.set(f"_last_tool_{tool_name}", result)
                    self.working.add_step_result(
                        step_name=tool_name, result=result
                    )

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

            # 检查计划是否全部完成
            if self.current_plan and self.current_plan.get_next_step() is None:
                if self.current_plan.failed_steps == 0:
                    self.state = AgentState.DONE
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
        """构建发送给 LLM 的消息列表"""
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

        return messages

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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tools = self.tool_registry.to_ollama_schemas()
                full_content = ""
                final_tool_calls: list[dict] = []
                final_usage: dict = {}

                for event in self.llm.chat_stream(messages, tools=tools if tools else None):
                    if event.type == "token":
                        full_content += event.content
                        self.on_token(event.content)
                        # 同时触发 on_step 的 token 事件
                        self._emit("token", {"content": event.content})
                    elif event.type == "done":
                        full_content = event.content or full_content
                        final_tool_calls = event.tool_calls
                        final_usage = event.usage

                if full_content or final_tool_calls:
                    return LLMResponse(
                        content=full_content,
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
