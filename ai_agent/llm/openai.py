"""
OpenAI LLM 集成（基于官方 openai SDK）

支持 OpenAI API 及任何 OpenAI 兼容的 API（如 Azure、DeepSeek、Moonshot 等）。
使用 openai.OpenAI 客户端，自动处理认证、重试、流式 SSE 解析和错误格式化。
支持将原始 LLM 响应保存到 JSONL 文件（response_log_path）。
"""

import json
import logging
import os
import time
from typing import Any, Generator

from openai import (
    APIError,
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from ai_agent.llm.base import BaseLLM, LLMResponse, StreamEvent

logger = logging.getLogger(__name__)

# 常见的 OpenAI 兼容 API 基础地址（含 API 版本路径）
KNOWN_BASE_URLS: dict[str, str | None] = {
    "openai": "https://api.openai.com/v1",
    "azure": None,  # Azure 需要用户指定完整 endpoint
    "deepseek": "https://api.deepseek.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "xai": "https://api.x.ai/v1",
    "custom": None,
}


class OpenAILLM(BaseLLM):
    """OpenAI API 及兼容 API 的 LLM 实现（基于 openai SDK）"""

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
        max_retries: int = 2,
    ):
        """
        Args:
            model: 模型名（如 gpt-4o, deepseek-chat 等）
            api_key: API 密钥。不提供则从 OPENAI_API_KEY 环境变量读取。
            base_url: API 基础地址（含版本路径，如 https://api.openai.com/v1）。
                      不提供则根据 model 名或 OPENAI_BASE_URL 环境变量自动推断。
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            response_log_path: 原始响应 JSONL 文件路径
            enable_thinking: 启用模型推理（部分模型如 o1 系列支持）
            extra_headers: 额外的 HTTP 请求头
            max_retries: 自动重试次数（SDK 内置）
        """
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_log_path = response_log_path
        self.enable_thinking = enable_thinking
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries

        # API 密钥：参数 > 环境变量
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "未设置 OpenAI API Key，"
                "请通过 api_key 参数或 OPENAI_API_KEY 环境变量提供"
            )

        # 基础地址：参数 > 环境变量 > 已知地址推断 > OpenAI 默认
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif os.environ.get("OPENAI_BASE_URL"):
            self.base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
        else:
            self.base_url = self._infer_base_url(model)

        # 创建 OpenAI 客户端（SDK 自动处理认证、重试、超时等）
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_retries": max_retries,
            "timeout": 120.0,
        }
        if self.extra_headers:
            client_kwargs["default_headers"] = self.extra_headers
        self._client = OpenAI(**client_kwargs)

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
        openai_tools = self._convert_tools(tools)
        params = self._build_params(messages, openai_tools, stream=False)

        self._log_request(messages, tools)

        try:
            response = self._client.chat.completions.create(**params)
            result = self._parse_sdk_response(response)
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
        openai_tools = self._convert_tools(tools)
        params = self._build_params(messages, openai_tools, stream=True)
        params["stream_options"] = {"include_usage": True}

        self._log_request(messages, tools)

        full_content = ""
        full_thinking = ""
        tool_call_chunks: dict[int, dict] = {}
        usage: dict = {}
        finish_reason = "stop"

        try:
            stream = self._client.chat.completions.create(**params)
            for chunk in stream:
                # SDK 在流式模式下有时会在最后一个 chunk 附带 usage
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                        "total_tokens": chunk.usage.total_tokens or 0,
                    }

                choices = chunk.choices or []
                if not choices:
                    continue

                choice = choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                # 推理 token（reasoning_content / thinking）
                think_token = getattr(delta, "reasoning_content", "") or ""
                if think_token:
                    full_thinking += think_token
                    yield StreamEvent(type="thinking", content=think_token)

                # 普通文本 token
                token = delta.content or ""
                if token:
                    full_content += token
                    yield StreamEvent(type="token", content=token)

                # 累积 tool_calls（流式模式下分多个 chunk 传输）
                for tc in delta.tool_calls or []:
                    idx = tc.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments_str": "",
                        }
                    entry = tool_call_chunks[idx]
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function and tc.function.name:
                        entry["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        entry["arguments_str"] += tc.function.arguments

        except APIError as e:
            err_detail = self._log_sdk_error(e, params=params)
            yield StreamEvent(
                type="done",
                content=f"调用 LLM 失败: {e}\n\n{err_detail}",
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
        """获取可用模型列表（从 API 获取，失败则返回当前模型）"""
        try:
            models_page = self._client.models.list()
            # SDK 返回 SyncPage[Model]，可直接迭代
            model_ids = [m.id for m in models_page]
            model_ids.sort()
            return model_ids
        except APIError as e:
            logger.warning(f"获取模型列表失败: {e}")
            return [self._model]

    def create_embedding(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float] | None:
        """生成文本嵌入向量

        按顺序尝试：指定模型 → text-embedding-3-small → 当前 chat 模型。

        Args:
            text: 输入文本
            model: 嵌入模型名（默认为 text-embedding-3-small）

        Returns:
            嵌入向量列表，失败返回 None
        """
        candidates = [model] if model else []
        candidates += ["text-embedding-3-small", "text-embedding-ada-002", self._model]

        seen = set()
        for m in candidates:
            if m in seen:
                continue
            seen.add(m)
            try:
                resp = self._client.embeddings.create(model=m, input=text)
                return resp.data[0].embedding
            except Exception:
                continue

        logger.warning("所有嵌入模型均不可用，语义搜索将回退到关键词匹配")
        return None

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def _build_params(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """构建 SDK create() 参数"""
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }

        if tools:
            params["tools"] = tools

        # o1/o3 系列：不支持 temperature，用 max_completion_tokens
        if self._model.startswith(("o1", "o3")):
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["temperature"] = self.temperature
            params["max_tokens"] = self.max_tokens

        return params

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """将内部工具格式转换为 OpenAI function calling 格式"""
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
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

    def _parse_sdk_response(self, response) -> LLMResponse:
        """解析 SDK 响应对象为 LLMResponse"""
        choice = response.choices[0]
        message = choice.message

        content = message.content or ""

        # 推理内容（reasoning_content 是部分模型的扩展字段）
        thinking = getattr(message, "reasoning_content", "") or ""

        # 解析 tool_calls
        tc_list = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tc_list.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })

        usage_data = response.usage
        return LLMResponse(
            content=content.strip(),
            thinking=thinking.strip() if thinking else "",
            tool_calls=tc_list,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": usage_data.prompt_tokens if usage_data else 0,
                "completion_tokens": usage_data.completion_tokens if usage_data else 0,
                "total_tokens": usage_data.total_tokens if usage_data else 0,
            },
        )

    def _infer_base_url(self, model: str) -> str:
        """根据模型名推断 API 基础地址"""
        model_lower = model.lower()

        provider_patterns = [
            ("deepseek", "deepseek"),
            ("moonshot", "moonshot"),
            ("kimi", "moonshot"),
            ("glm", "zhipu"),
            ("chatglm", "zhipu"),
            ("qwen", "qwen"),
            ("qwq", "qwen"),
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

        logger.info(f"使用默认 OpenAI API 地址，模型: {model}")
        return "https://api.openai.com/v1"

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

    def _log_sdk_error(self, error: APIError, params: dict | None = None) -> str:
        """记录 SDK 错误详情，返回详细描述字符串"""
        lines = []
        lines.append(f"异常类型: {type(error).__name__}")
        lines.append(f"异常信息: {error}")

        # 请求信息
        if params:
            lines.append(f"\n{'─' * 40}")
            lines.append("📤 请求详情:")
            lines.append(f"  URL: {self.base_url}/chat/completions")
            lines.append(f"  Model: {self._model}")
            lines.append(f"  Temperature: {self.temperature}")
            lines.append(f"  Max tokens: {self.max_tokens}")
            lines.append(f"  Stream: {params.get('stream', False)}")
            lines.append(f"  Messages count: {len(params.get('messages', []))}")

            messages = params.get("messages", [])
            for i, msg in enumerate(messages):
                role = msg.get("role", "?")
                content = msg.get("content", "")[:200]
                extra = ""
                if msg.get("tool_calls"):
                    tc_names = [tc.get("function", {}).get("name", "?")
                                for tc in msg["tool_calls"]]
                    extra = f", tool_calls={tc_names}"
                if msg.get("tool_call_id"):
                    extra = f", tool_call_id={msg['tool_call_id']}"
                lines.append(f"  [{i}] role={role}{extra}: {content[:200]}")

            tools = params.get("tools", [])
            if tools:
                tool_names = [t.get("function", {}).get("name", "?") for t in tools]
                lines.append(f"  Tools: {tool_names}")

        # 响应信息（SDK 错误对象自带详细信息）
        lines.append(f"\n{'─' * 40}")
        lines.append("📥 错误详情:")
        if isinstance(error, APIStatusError):
            lines.append(f"  HTTP Status: {error.status_code}")
            lines.append(f"  Request ID: {error.request_id or 'N/A'}")
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
                "provider": "openai",
                "base_url": self.base_url,
                "error_type": type(error).__name__,
                "error": str(error),
                "status_code": error.status_code if isinstance(error, APIStatusError) else None,
                "response_body": str(error.body)[:3000] if isinstance(error, APIStatusError) and error.body else None,
                "request_payload": {
                    "model": self._model,
                    "messages": params.get("messages", []),
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": params.get("stream", False),
                    "tools": params.get("tools", []),
                },
            }
            if params.get("stream_options"):
                error_entry["request_payload"]["stream_options"] = params["stream_options"]
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
