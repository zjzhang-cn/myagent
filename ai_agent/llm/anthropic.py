"""
Anthropic Claude LLM 集成（基于 anthropic SDK）

支持 Anthropic Messages API，包括工具调用、流式传输和扩展思考（extended thinking）。
消息格式在内部 OpenAI 格式和 Anthropic 格式之间自动转换。
"""

import json
import logging
import os
import time
from typing import Any, Generator

from ai_agent.llm.base import BaseLLM, LLMResponse, StreamEvent

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic
    from anthropic import (
        APIError,
        APIConnectionError,
        APIStatusError,
        BadRequestError,
        RateLimitError,
    )
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    Anthropic = None  # type: ignore
    APIError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore


class AnthropicLLM(BaseLLM):
    """Anthropic Claude API 的 LLM 实现（基于 anthropic SDK）"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_log_path: str | None = None,
        enable_thinking: bool = False,
        thinking_budget_tokens: int = 16000,
        max_retries: int = 2,
    ):
        """
        Args:
            model: 模型名（如 claude-sonnet-4-20250514）
            api_key: API 密钥。不提供则从 LLM_API_KEY 环境变量读取。
            base_url: API 基础地址。不提供则从 LLM_BASE_URL 环境变量读取，默认 https://api.anthropic.com。
            temperature: 生成温度（扩展思考启用时忽略）
            max_tokens: 最大生成 token 数（Anthropic 要求必需）
            response_log_path: 原始响应 JSONL 文件路径
            enable_thinking: 启用扩展思考（需要模型支持）
            thinking_budget_tokens: 思考 token 预算（默认 16000）
            max_retries: 自动重试次数
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Claude 模型需要 'anthropic' 包。请执行: pip install anthropic"
            )

        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_log_path = response_log_path
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens

        # API 密钥：参数 > LLM_API_KEY
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "未设置 API Key，请通过 api_key 参数或 LLM_API_KEY 环境变量提供"
            )

        # 基础地址：参数 > LLM_BASE_URL > SDK 默认
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "") or None
        if self.base_url:
            logger.info(f"使用自定义 API 地址: {self.base_url}")

        # 创建 Anthropic 客户端
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "max_retries": max_retries,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = Anthropic(**client_kwargs)

        # 确保日志文件目录存在
        if response_log_path:
            log_dir = os.path.dirname(response_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self._model

    # ----------------------------------------------------------
    # 非流式请求
    # ----------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """发送聊天请求到 Anthropic API（非流式）"""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        params = self._build_params(system_prompt, anthropic_messages, anthropic_tools, stream=False)

        self._log_request(messages, tools)

        try:
            response = self._client.messages.create(**params)
            result = self._parse_response(response)
            self._log_response(result)
            return result
        except APIError as e:
            err_detail = self._log_sdk_error(e, params=params)
            return LLMResponse(
                content=f"调用 LLM 失败: {e}\n\n{err_detail}",
                tool_calls=[],
            )

    # ----------------------------------------------------------
    # 流式请求
    # ----------------------------------------------------------

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """流式聊天请求，逐 token 返回"""
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        params = self._build_params(system_prompt, anthropic_messages, anthropic_tools, stream=True)

        self._log_request(messages, tools)

        full_content = ""
        full_thinking = ""
        tool_use_blocks: dict[int, dict] = {}
        usage: dict = {}
        stop_reason = "end_turn"

        try:
            with self._client.messages.stream(**params) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_use_blocks[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "partial_json": "",
                            }
                        elif block.type == "thinking":
                            # thinking 块的签名在 delta 中会补充
                            pass

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            full_content += delta.text
                            yield StreamEvent(type="token", content=delta.text)
                        elif delta.type == "input_json_delta":
                            if event.index in tool_use_blocks:
                                tool_use_blocks[event.index]["partial_json"] += delta.partial_json
                        elif delta.type == "thinking_delta":
                            full_thinking += delta.thinking
                            yield StreamEvent(type="thinking", content=delta.thinking)

                    elif event.type == "content_block_stop":
                        pass

                    elif event.type == "message_start":
                        if event.message.usage:
                            usage = {
                                "prompt_tokens": event.message.usage.input_tokens or 0,
                                "completion_tokens": event.message.usage.output_tokens or 0,
                                "total_tokens": (event.message.usage.input_tokens or 0) + (event.message.usage.output_tokens or 0),
                            }

                    elif event.type == "message_delta":
                        stop_reason = event.delta.stop_reason or stop_reason
                        if event.usage:
                            output_tokens = event.usage.output_tokens or 0
                            usage["completion_tokens"] = usage.get("completion_tokens", 0) + output_tokens
                            usage["total_tokens"] = usage.get("total_tokens", 0) + output_tokens

                    elif event.type == "message_stop":
                        pass

        except APIError as e:
            err_detail = self._log_sdk_error(e, params=params)
            yield StreamEvent(
                type="done",
                content=f"调用 LLM 失败: {e}\n\n{err_detail}",
            )
            return

        # 解析累积的 tool_use
        tool_calls = []
        for idx in sorted(tool_use_blocks.keys()):
            tb = tool_use_blocks[idx]
            try:
                args = json.loads(tb["partial_json"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            if tb["name"]:
                tool_calls.append({
                    "id": tb["id"],
                    "name": tb["name"],
                    "arguments": args,
                })

        # Anthropic stop_reason → OpenAI finish_reason
        reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = reason_map.get(stop_reason, stop_reason)

        logger.debug(
            f"[LLM 流式响应] content={full_content[:300]!r}, "
            f"tool_calls={[tc['name'] for tc in tool_calls]}, "
            f"usage={usage}"
        )

        # 保存原始响应
        self._save_raw_response(LLMResponse(
            content=full_content.strip(),
            thinking=full_thinking.strip(),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        ))

        yield StreamEvent(
            type="done",
            content=full_content.strip(),
            thinking=full_thinking.strip(),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    # ----------------------------------------------------------
    # 模型列表
    # ----------------------------------------------------------

    def list_models(self) -> list[str]:
        """获取可用模型列表"""
        try:
            models_page = self._client.models.list()
            model_ids = [m.id for m in models_page]
            model_ids.sort()
            return model_ids
        except Exception as e:
            logger.warning(f"获取模型列表失败: {e}")
            return [self._model]

    # ----------------------------------------------------------
    # 消息格式转换
    # ----------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        """将内部 OpenAI 格式消息转换为 Anthropic 格式。

        Args:
            messages: 内部消息列表 [{"role": "system", "content": "..."}, ...]

        Returns:
            (system_prompt, anthropic_messages)
        """
        system_prompt = ""
        anthropic_messages: list[dict] = []

        # 提取第一条 system 消息
        start_idx = 0
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", "")
            start_idx = 1

        i = start_idx
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")

            if role == "user":
                anthropic_messages.append(AnthropicLLM._convert_user_message(msg))
                i += 1

            elif role == "assistant":
                anthropic_messages.append(AnthropicLLM._convert_assistant_message(msg))

                # 收集后续连续的 tool 消息，合并为一条 user(tool_result) 消息
                j = i + 1
                tool_results: list[dict] = []
                while j < len(messages) and messages[j].get("role") == "tool":
                    tc_id = messages[j].get("tool_call_id", "")
                    content = messages[j].get("content", "")
                    if tc_id:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc_id,
                            "content": content,
                        })
                    else:
                        logger.warning(f"跳过缺少 tool_call_id 的 tool 消息 (index={j})")
                    j += 1

                if tool_results:
                    anthropic_messages.append({
                        "role": "user",
                        "content": tool_results,
                    })

                i = j

            elif role == "tool":
                # 孤立的 tool 消息（没有前置 assistant），跳过
                logger.warning(
                    f"跳过孤立的 tool 消息 (index={i}, "
                    f"tool_call_id={msg.get('tool_call_id', '?')[:50]})"
                )
                i += 1

            elif role == "system":
                # 非首条 system 消息，追加到 system_prompt
                content = msg.get("content", "")
                if content:
                    system_prompt += "\n\n" + content
                i += 1

            else:
                i += 1

        # 空消息列表：至少需要一条 user 消息
        if not anthropic_messages:
            anthropic_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": ""}],
            })

        return system_prompt.strip(), anthropic_messages

    @staticmethod
    def _convert_user_message(msg: dict) -> dict:
        """转换 user 消息格式"""
        content = msg.get("content", "")
        if isinstance(content, str):
            return {"role": "user", "content": [{"type": "text", "text": content}]}
        elif isinstance(content, list):
            return {"role": "user", "content": content}
        return {"role": "user", "content": [{"type": "text", "text": str(content)}]}

    @staticmethod
    def _convert_assistant_message(msg: dict) -> dict:
        """转换 assistant 消息格式（含 tool_calls 和 reasoning_content）"""
        content_blocks: list[dict] = []
        text_content = msg.get("content", "")

        # 将 reasoning_content 前置到文本内容（Anthropic 不接受 thinking 块在历史消息中）
        reasoning = msg.get("reasoning_content", "") or msg.get("thinking", "")
        if reasoning:
            text_content = f"[推理]\n{reasoning}\n\n{text_content}" if text_content else f"[推理]\n{reasoning}"

        if text_content:
            content_blocks.append({"type": "text", "text": text_content})

        # 转换 tool_calls
        for tc in msg.get("tool_calls", []):
            # 支持两种格式：OpenAI 包装格式 和 扁平格式
            if "function" in tc:
                name = tc["function"].get("name", "")
                args = tc["function"].get("arguments", {})
            else:
                name = tc.get("name", "")
                args = tc.get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": name,
                "input": args,
            })

        # Anthropic 要求每条 assistant 消息至少有一个 content block
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        return {"role": "assistant", "content": content_blocks}

    # ----------------------------------------------------------
    # 工具格式转换
    # ----------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: list[dict] | None) -> list[dict] | None:
        """将内部工具格式转换为 Anthropic 格式。

        内部格式（OpenAI function calling）:
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        或扁平格式:
            {"name": "...", "description": "...", "parameters": {...}}

        Anthropic 格式:
            {"name": "...", "description": "...", "input_schema": {...}}
        """
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            # 已经是 Anthropic 格式
            if "input_schema" in tool:
                anthropic_tools.append(tool)
                continue

            # OpenAI 包装格式
            if "function" in tool:
                func = tool["function"]
                anthropic_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }),
                })
            else:
                # 扁平格式
                anthropic_tools.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("parameters", {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }),
                })
        return anthropic_tools

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def _build_params(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """构建 client.messages.create() 参数"""
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = tools

        if self.enable_thinking:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            # temperature 在扩展思考模式下不可用
        else:
            params["temperature"] = self.temperature

        return params

    @staticmethod
    def _parse_response(response) -> LLMResponse:
        """解析 Anthropic API 响应为 LLMResponse"""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input if isinstance(block.input, dict) else {},
                })
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)

        # 映射 Anthropic stop_reason → OpenAI finish_reason
        reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        finish_reason = reason_map.get(reason, reason)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens or 0,
                "completion_tokens": response.usage.output_tokens or 0,
                "total_tokens": (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0),
            }

        return LLMResponse(
            content="\n".join(content_parts).strip(),
            thinking="\n".join(thinking_parts).strip() if thinking_parts else "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    # ----------------------------------------------------------
    # 日志辅助
    # ----------------------------------------------------------

    def _log_request(self, messages: list[dict], tools: list[dict] | None) -> None:
        """记录 LLM 请求日志"""
        self._save_raw_request(messages, tools)

        if not logger.isEnabledFor(logging.DEBUG):
            return
        recent = messages[-4:] if len(messages) > 4 else messages
        parts = []
        for msg in recent:
            content = str(msg.get("content", ""))[:200]
            role = msg.get("role", "?")
            parts.append(f"[{role}] {content!r}")
        logger.debug(
            f"[LLM 请求] model={self._model}, provider=anthropic, "
            f"msgs={len(messages)}, "
            f"tools={[t.get('function', t).get('name', '?') for t in (tools or [])]}, "
            f"recent={' | '.join(parts)}"
        )

    def _log_response(self, response: LLMResponse) -> None:
        """记录 LLM 响应日志"""
        logger.debug(
            f"[LLM 响应] content={response.content[:300]!r}, "
            f"tool_calls={[tc['name'] for tc in response.tool_calls]}, "
            f"usage={response.usage}"
        )
        self._save_raw_response(response)

    def _log_sdk_error(self, error: APIError, params: dict | None = None) -> str:
        """记录 SDK 错误详情，返回详细描述字符串"""
        lines = []
        lines.append(f"异常类型: {type(error).__name__}")
        lines.append(f"异常信息: {error}")

        if params:
            lines.append(f"\n{'─' * 40}")
            lines.append("请求详情:")
            lines.append(f"  Model: {self._model}")
            lines.append(f"  Max tokens: {self.max_tokens}")
            lines.append(f"  Messages count: {len(params.get('messages', []))}")
            lines.append(f"  Thinking: {self.enable_thinking}")

            tools = params.get("tools", [])
            if tools:
                tool_names = [t.get("name", "?") for t in tools]
                lines.append(f"  Tools: {tool_names}")

        lines.append(f"\n{'─' * 40}")
        lines.append("错误详情:")
        if isinstance(error, APIStatusError):
            lines.append(f"  HTTP Status: {error.status_code}")
            try:
                body_preview = str(error.body)[:2000] if error.body else "(空)"
                lines.append(f"  Body: {body_preview}")
            except Exception:
                lines.append("  Body: (无法读取)")
        elif isinstance(error, APIConnectionError):
            lines.append("  类型: 连接错误（网络不可达或超时）")
            lines.append(f"  详情: {error}")

        detail = "\n".join(lines)
        logger.error(detail)

        # 保存错误日志到 JSONL
        if self.response_log_path and params:
            error_entry = {
                "type": "error",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": self._model,
                "provider": "anthropic",
                "error_type": type(error).__name__,
                "error": str(error),
                "status_code": error.status_code if isinstance(error, APIStatusError) else None,
                "response_body": str(error.body)[:3000] if isinstance(error, APIStatusError) and error.body else None,
                "request_payload": {
                    "model": self._model,
                    "messages": params.get("messages", []),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "tools": params.get("tools", []),
                    "thinking": self.enable_thinking,
                },
            }
            self._write_jsonl(error_entry)

        return detail

    def _save_raw_request(self, messages: list[dict], tools: list[dict] | None) -> None:
        """保存原始 LLM 请求到 JSONL 文件"""
        if not self.response_log_path:
            return
        tool_names = None
        if tools:
            tool_names = [t.get("function", t).get("name", str(t)) for t in tools]
        entry = {
            "type": "request",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self._model,
            "provider": "anthropic",
            "message_count": len(messages),
            "messages": messages,
            "tools": tool_names,
        }
        self._write_jsonl(entry)

    def _save_raw_response(self, response: LLMResponse) -> None:
        """保存原始 LLM 响应到 JSONL 文件"""
        if not self.response_log_path:
            return
        entry = {
            "type": "response",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self._model,
            "provider": "anthropic",
            "content": response.content,
            "thinking": response.thinking,
            "tool_calls": response.tool_calls,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }
        self._write_jsonl(entry)

    def _write_jsonl(self, entry: dict) -> None:
        """追加一行 JSON 到日志文件"""
        try:
            with open(self.response_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"保存原始数据失败: {e}")
