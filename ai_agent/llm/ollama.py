"""
Ollama LLM 集成

支持两种模式：
1. 原生 Ollama API (推荐)
2. OpenAI 兼容 API (/v1/chat/completions)

均支持流式 (stream) 和非流式两种调用方式。
支持将原始 LLM 响应保存到 JSONL 文件（response_log_path）。
"""

import json
import logging
import os
import time
from typing import Any, Generator

import requests

from ai_agent.llm.base import BaseLLM, LLMResponse, StreamEvent

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM 实现"""

    def __init__(
        self,
        model: str = "minimax-m2.5:cloud",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        use_openai_compat: bool = False,
        response_log_path: str | None = None,
        enable_thinking: bool = False,
    ):
        """
        Args:
            model: Ollama 模型名
            host: Ollama 服务地址
            temperature: 生成温度
            max_tokens: 最大 token 数
            use_openai_compat: 使用 OpenAI 兼容 API
            response_log_path: 原始响应 JSONL 文件路径（追加写入每条完整响应）
            enable_thinking: 启用模型推理（think 参数），支持推理的模型将返回 thinking 内容
        """
        self._model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_openai_compat = use_openai_compat
        self.response_log_path = response_log_path
        self.enable_thinking = enable_thinking
        # 确保日志文件目录存在
        if response_log_path:
            log_dir = os.path.dirname(response_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """发送聊天请求到 Ollama（非流式）"""
        if self.use_openai_compat:
            return self._chat_openai_compat(messages, tools)
        else:
            return self._chat_native(messages, tools)

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """流式聊天请求，逐 token 返回"""
        if self.use_openai_compat:
            yield from self._chat_stream_openai_compat(messages, tools)
        else:
            yield from self._chat_stream_native(messages, tools)

    def _chat_native(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """使用 Ollama 原生 API"""
        url = f"{self.host}/api/chat"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        if tools:
            payload["tools"] = tools
        if self.enable_thinking:
            payload["think"] = True

        self._log_request(messages, tools)

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            message = data.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking", "")

            # 解析 tool_calls（Ollama 原生格式）
            tool_calls_raw = message.get("tool_calls", [])
            tool_calls = self._parse_tool_calls(tool_calls_raw)

            result = LLMResponse(
                content=content.strip() if content else "",
                thinking=thinking.strip() if thinking else "",
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )
            self._log_response(result)
            return result
        except requests.RequestException as e:
            logger.error(f"Ollama API 请求失败: {e}")
            logger.debug(f"请求 URL: {url}, model: {self._model}")
            return LLMResponse(
                content=f"调用 LLM 失败: {e}",
                tool_calls=[],
            )

    def _chat_openai_compat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """使用 OpenAI 兼容 API"""
        url = f"{self.host}/v1/chat/completions"

        # 转换 tool 格式（OpenAI 风格的 function definitions）
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get("parameters", {}),
                    },
                })

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if openai_tools:
            payload["tools"] = openai_tools
        if self.enable_thinking:
            payload["think"] = True

        self._log_request(messages, tools)

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking", "")

            # 解析 tool_calls（OpenAI 格式）
            tc_list = []
            raw_tool_calls = message.get("tool_calls", [])
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tc_list.append({
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "arguments": args,
                })

            result = LLMResponse(
                content=content.strip() if content else "",
                thinking=thinking.strip() if thinking else "",
                tool_calls=tc_list,
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                },
            )
            self._log_response(result)
            return result
        except requests.RequestException as e:
            logger.error(f"Ollama OpenAI-compat API 请求失败: {e}")
            logger.debug(f"请求 URL: {url}, model: {self._model}")
            return LLMResponse(
                content=f"调用 LLM 失败: {e}",
                tool_calls=[],
            )

    # ----------------------------------------------------------
    # 流式实现
    # ----------------------------------------------------------

    def _chat_stream_native(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """Ollama 原生 API 流式请求（NDJSON）"""
        url = f"{self.host}/api/chat"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        if self.enable_thinking:
            payload["think"] = True

        full_content = ""
        full_thinking = ""
        last_message: dict = {}

        self._log_request(messages, tools)

        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message = chunk.get("message", {})
                    last_message = message

                    # 推理 token（thinking）
                    think_token = message.get("thinking", "")
                    if think_token:
                        full_thinking += think_token
                        yield StreamEvent(type="thinking", content=think_token)

                    # 普通 token
                    token = message.get("content", "")
                    if token:
                        full_content += token
                        yield StreamEvent(type="token", content=token)

                    if chunk.get("done", False):
                        tool_calls = self._parse_tool_calls(
                            last_message.get("tool_calls", [])
                        )
                        usage = {
                            "prompt_tokens": chunk.get("prompt_eval_count", 0),
                            "completion_tokens": chunk.get("eval_count", 0),
                        }
                        logger.debug(
                            f"[LLM 响应] thinking={full_thinking[:200]!r}, "
                            f"content={full_content[:300]!r}, "
                            f"tool_calls={[tc['name'] for tc in tool_calls]}, "
                            f"usage={usage}"
                        )
                        self._save_raw_response(LLMResponse(
                            content=full_content.strip(),
                            thinking=full_thinking.strip(),
                            tool_calls=tool_calls,
                            usage=usage,
                        ))
                        yield StreamEvent(
                            type="done",
                            content=full_content.strip(),
                            thinking=full_thinking.strip(),
                            tool_calls=tool_calls,
                            usage=usage,
                        )
                        return

        except requests.RequestException as e:
            logger.error(f"Ollama 流式请求失败: {e}")
            logger.debug(f"请求 URL: {url}, model: {self._model}")
            yield StreamEvent(
                type="done",
                content=f"调用 LLM 失败: {e}",
            )

    def _chat_stream_openai_compat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[StreamEvent, None, None]:
        """OpenAI 兼容 API 流式请求（SSE）"""
        url = f"{self.host}/v1/chat/completions"

        # 转换 tool 格式
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get("parameters", {}),
                    },
                })

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if openai_tools:
            payload["tools"] = openai_tools

        self._log_request(messages, tools)

        full_content = ""
        tool_call_chunks: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        usage = {}
        finish_reason = "stop"

        try:
            with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "usage" in chunk:
                        usage = chunk["usage"]

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]

                    delta = choices[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full_content += token
                        yield StreamEvent(type="token", content=token)

                    # 累积 tool_calls（流式模式下分多个 chunk 传输）
                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        if idx not in tool_call_chunks:
                            tool_call_chunks[idx] = {
                                "id": "",
                                "name": "",
                                "arguments_str": "",
                            }
                        entry = tool_call_chunks[idx]
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        func = tc.get("function", {})
                        if func.get("name"):
                            entry["name"] += func["name"]
                        if func.get("arguments"):
                            entry["arguments_str"] += func["arguments"]

        except requests.RequestException as e:
            logger.error(f"OpenAI-compat 流式请求失败: {e}")
            yield StreamEvent(
                type="done",
                content=f"调用 LLM 失败: {e}",
            )
            return

        # 解析累积的 tool_calls
        tool_calls = []
        for entry in tool_call_chunks.values():
            try:
                args = json.loads(entry["arguments_str"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            if entry["name"]:
                tool_calls.append({
                    "id": entry["id"],
                    "name": entry["name"],
                    "arguments": args,
                })

        final_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        logger.debug(
            f"[LLM 流式响应] content={full_content[:300]!r}, "
            f"tool_calls={[tc['name'] for tc in tool_calls]}, "
            f"usage={final_usage}"
        )
        # 保存原始响应
        self._save_raw_response(LLMResponse(
            content=full_content.strip(),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=final_usage,
        ))
        yield StreamEvent(
            type="done",
            content=full_content.strip(),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=final_usage,
        )

    # ----------------------------------------------------------
    # 日志辅助
    # ----------------------------------------------------------

    def _log_request(self, messages: list[dict], tools: list[dict] | None) -> None:
        """记录 LLM 请求日志"""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        # 取最后几条消息（用户输入 + 最近的上下文）
        recent = messages[-4:] if len(messages) > 4 else messages
        parts = []
        for msg in recent:
            content = msg.get("content", "")[:200]
            role = msg.get("role", "?")
            parts.append(f"[{role}] {content!r}")
        logger.debug(f"[LLM 请求] model={self._model}, msgs={len(messages)}, "
                     f"tools={[t['function']['name'] if 'function' in t else t.get('name','?') for t in (tools or [])]}, "
                     f"recent={' | '.join(parts)}")

    def _log_response(self, response: LLMResponse) -> None:
        """记录 LLM 响应日志"""
        logger.debug(
            f"[LLM 响应] content={response.content[:300]!r}, "
            f"tool_calls={[tc['name'] for tc in response.tool_calls]}, "
            f"usage={response.usage}"
        )
        self._save_raw_response(response)

    def _save_raw_response(self, response: LLMResponse) -> None:
        """保存原始 LLM 响应到 JSONL 文件"""
        if not self.response_log_path:
            return
        entry = {
            "type": "response",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self._model,
            "content": response.content,
            "thinking": response.thinking,
            "tool_calls": response.tool_calls,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
        }
        self._write_jsonl(entry)

    def list_models(self) -> list[str]:
        """获取 Ollama 可用模型列表"""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except requests.RequestException as e:
            logger.warning(f"获取模型列表失败: {e}")
            return [self._model]

    def _parse_tool_calls(self, raw_tool_calls: list) -> list[dict]:
        """解析 Ollama 原生 tool_calls 为标准格式"""
        parsed = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            parsed.append({
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "arguments": args,
            })
        return parsed

    @staticmethod
    def pull_model(model: str, host: str = "http://localhost:11434") -> bool:
        """拉取 Ollama 模型"""
        try:
            resp = requests.post(
                f"{host.rstrip('/')}/api/pull",
                json={"name": model},
                timeout=300,
            )
            resp.raise_for_status()
            logger.info(f"模型 {model} 拉取成功")
            return True
        except requests.RequestException as e:
            logger.error(f"拉取模型失败: {e}")
            return False
