"""
OpenAI LLM 集成

支持 OpenAI API 及任何 OpenAI 兼容的 API（如 Azure、DeepSeek、Moonshot 等）。
通过 /v1/chat/completions 接口进行聊天补全，支持流式和非流式两种调用方式。
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

# 常见的 OpenAI 兼容 API 基础地址
KNOWN_BASE_URLS = {
    "openai": "https://api.openai.com",
    "azure": None,  # Azure 需要用户指定完整 endpoint
    "deepseek": "https://api.deepseek.com",
    "moonshot": "https://api.moonshot.cn",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode",
    "siliconflow": "https://api.siliconflow.cn",
    "groq": "https://api.groq.com/openai",
    "together": "https://api.together.xyz",
    "fireworks": "https://api.fireworks.ai/inference",
    "xai": "https://api.x.ai",
    "custom": None,
}


class OpenAILLM(BaseLLM):
    """OpenAI API 及兼容 API 的 LLM 实现"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_log_path: str | None = None,
        enable_thinking: bool = False,
        extra_headers: dict | None = None,
    ):
        """
        Args:
            model: 模型名（如 gpt-4o, gpt-4o-mini, deepseek-chat 等）
            api_key: API 密钥。不提供则从 OPENAI_API_KEY 环境变量读取。
            base_url: API 基础地址（如 https://api.openai.com）。
                      不提供则根据 model 或 OPENAI_BASE_URL 环境变量自动推断。
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            response_log_path: 原始响应 JSONL 文件路径（追加写入每条完整响应）
            enable_thinking: 启用模型推理（部分模型支持，如 o1 系列）
            extra_headers: 额外的 HTTP 请求头
        """
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_log_path = response_log_path
        self.enable_thinking = enable_thinking
        self.extra_headers = extra_headers or {}

        # API 密钥：参数 > 环境变量
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning("未设置 OpenAI API Key，请通过 api_key 参数或 OPENAI_API_KEY 环境变量提供")

        # 基础地址：参数 > 环境变量 > 已知地址推断 > OpenAI 默认
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif os.environ.get("OPENAI_BASE_URL"):
            self.base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
        else:
            self.base_url = self._infer_base_url(model)

        # 构建认证头
        self._auth_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

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
        """发送聊天请求到 OpenAI API（非流式）"""
        url = f"{self.base_url}/v1/chat/completions"

        openai_tools = self._convert_tools(tools)
        payload = self._build_payload(messages, openai_tools, stream=False)

        self._log_request(messages, tools)

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._auth_headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            result = self._parse_response(data)
            self._log_response(result)
            return result
        except requests.RequestException as e:
            logger.error(f"OpenAI API 请求失败: {e}")
            self._log_api_error(resp if 'resp' in dir() and resp is not None else None, e)
            return LLMResponse(
                content=f"调用 LLM 失败: {e}",
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
        url = f"{self.base_url}/v1/chat/completions"

        openai_tools = self._convert_tools(tools)
        payload = self._build_payload(messages, openai_tools, stream=True)
        # 流式请求需要 stream_options 来获取 usage 信息
        payload["stream_options"] = {"include_usage": True}

        self._log_request(messages, tools)

        full_content = ""
        full_thinking = ""
        tool_call_chunks: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        usage = {}
        finish_reason = "stop"

        try:
            with requests.post(
                url,
                json=payload,
                headers=self._auth_headers,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str.strip() == "[DONE]":
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

                    # 推理 token（部分模型如 o1、DeepSeek-R1 支持 thinking/reasoning_content）
                    think_token = delta.get("reasoning_content", "") or delta.get("thinking", "")
                    if think_token:
                        full_thinking += think_token
                        yield StreamEvent(type="thinking", content=think_token)

                    # 普通 token
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
            logger.error(f"OpenAI 流式请求失败: {e}")
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
            "total_tokens": usage.get("total_tokens", 0),
        }

        logger.debug(
            f"[LLM 流式响应] content={full_content[:300]!r}, "
            f"tool_calls={[tc['name'] for tc in tool_calls]}, "
            f"usage={final_usage}"
        )

        # 保存原始响应
        self._save_raw_response(LLMResponse(
            content=full_content.strip(),
            thinking=full_thinking.strip(),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=final_usage,
        ))

        yield StreamEvent(
            type="done",
            content=full_content.strip(),
            thinking=full_thinking.strip(),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=final_usage,
        )

    # ----------------------------------------------------------
    # 模型列表
    # ----------------------------------------------------------

    def list_models(self) -> list[str]:
        """获取可用模型列表（从 API 获取，失败则返回当前模型）"""
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers=self._auth_headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            models.sort()
            return models
        except requests.RequestException as e:
            logger.warning(f"获取模型列表失败: {e}")
            return [self._model]

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """构建请求 payload"""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        # o1 系列模型不支持 temperature 和 system role 的一些特性
        if self._model.startswith("o1") or self._model.startswith("o3"):
            payload.pop("temperature", None)
            # o1 使用 max_completion_tokens 而非 max_tokens
            payload["max_completion_tokens"] = payload.pop("max_tokens", 4096)

        # 启用推理（部分模型需要通过额外参数）
        if self.enable_thinking:
            # DeepSeek-R1 等推理模型不需要特殊参数，thinking 会自动包含在响应中
            pass

        return payload

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """将内部工具格式转换为 OpenAI function calling 格式"""
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
            # 兼容两种格式：{name, description, parameters} 或 {function: {name, ...}}
            if "function" in tool:
                # 已经是 OpenAI 格式
                openai_tools.append(tool)
            else:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        }),
                    },
                })
        return openai_tools

    def _parse_response(self, data: dict) -> LLMResponse:
        """解析 OpenAI API 响应为 LLMResponse"""
        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""

        # 推理内容（部分模型在 reasoning_content 字段返回）
        thinking = message.get("reasoning_content", "") or message.get("thinking", "") or ""

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

        usage_data = data.get("usage", {})
        return LLMResponse(
            content=content.strip(),
            thinking=thinking.strip() if thinking else "",
            tool_calls=tc_list,
            finish_reason=choice.get("finish_reason", "stop"),
            usage={
                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                "completion_tokens": usage_data.get("completion_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            },
        )

    def _infer_base_url(self, model: str) -> str:
        """根据模型名推断 API 基础地址"""
        model_lower = model.lower()

        # 按优先级匹配
        provider_patterns = [
            ("deepseek", "deepseek"),
            ("moonshot", "moonshot"),
            ("kimi", "moonshot"),
            ("glm", "zhipu"),
            ("chatglm", "zhipu"),
            ("qwen", "qwen"),
            ("qwq", "qwen"),
            ("deepseek-r1", "deepseek"),
            ("deepseek-v3", "deepseek"),
            ("siliconflow", "siliconflow"),
            ("groq", "groq"),
            ("llama", "together"),
            ("mixtral", "together"),
            ("fireworks", "fireworks"),
            ("grok", "xai"),
        ]

        for pattern, provider in provider_patterns:
            if pattern in model_lower:
                base = KNOWN_BASE_URLS.get(provider)
                if base:
                    logger.info(f"根据模型 '{model}' 自动推断 API 地址: {base}")
                    return base

        # 默认 OpenAI
        logger.info(f"使用默认 OpenAI API 地址，模型: {model}")
        return "https://api.openai.com"

    # ----------------------------------------------------------
    # 日志辅助
    # ----------------------------------------------------------

    def _log_request(self, messages: list[dict], tools: list[dict] | None) -> None:
        """记录 LLM 请求日志（同时保存原始请求到 JSONL）"""
        self._save_raw_request(messages, tools)

        if not logger.isEnabledFor(logging.DEBUG):
            return
        recent = messages[-4:] if len(messages) > 4 else messages
        parts = []
        for msg in recent:
            content = msg.get("content", "")[:200]
            role = msg.get("role", "?")
            parts.append(f"[{role}] {content!r}")
        logger.debug(
            f"[LLM 请求] model={self._model}, base_url={self.base_url}, "
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

    def _log_api_error(
        self,
        resp: requests.Response | None,
        error: Exception,
    ) -> None:
        """记录 API 错误详情"""
        if resp is not None:
            try:
                body = resp.text[:500]
            except Exception:
                body = "(无法读取响应体)"
            logger.debug(f"API 错误响应 ({resp.status_code}): {body}")

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
            "provider": "openai",
            "base_url": self.base_url,
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
            "provider": "openai",
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
