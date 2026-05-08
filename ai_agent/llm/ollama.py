"""
Ollama LLM 集成

支持两种模式：
1. 原生 Ollama API (推荐)
2. OpenAI 兼容 API (/v1/chat/completions)
"""

import json
import logging
from typing import Any

import requests

from ai_agent.llm.base import BaseLLM, LLMResponse

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
    ):
        self._model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_openai_compat = use_openai_compat

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """发送聊天请求到 Ollama"""
        if self.use_openai_compat:
            return self._chat_openai_compat(messages, tools)
        else:
            return self._chat_native(messages, tools)

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

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            message = data.get("message", {})
            content = message.get("content", "")

            # 解析 tool_calls（Ollama 原生格式）
            tool_calls_raw = message.get("tool_calls", [])
            tool_calls = self._parse_tool_calls(tool_calls_raw)

            return LLMResponse(
                content=content.strip() if content else "",
                tool_calls=tool_calls,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )
        except requests.RequestException as e:
            logger.error(f"Ollama API 请求失败: {e}")
            # 返回错误响应，让 Agent 处理
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

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")

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

            return LLMResponse(
                content=content.strip() if content else "",
                tool_calls=tc_list,
                usage={
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                },
            )
        except requests.RequestException as e:
            logger.error(f"Ollama OpenAI-compat API 请求失败: {e}")
            return LLMResponse(
                content=f"调用 LLM 失败: {e}",
                tool_calls=[],
            )

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
