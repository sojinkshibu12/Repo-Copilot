import os
import json
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from core.tracer import get_tracer

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    model: str = ""


class BaseProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...


_SUPPORTED_MODELS: dict[str, tuple[str, str]] = {
    # (provider, model_name)
    "claude-sonnet-4-20250514": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-sonnet-4": ("anthropic", "claude-sonnet-4-20250514"),
    "claude-3-5-haiku": ("anthropic", "claude-3-5-haiku-20241022"),
    "claude-3-opus": ("anthropic", "claude-3-opus-20240229"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4-turbo": ("openai", "gpt-4-turbo"),
    "o3-mini": ("openai", "o3-mini"),
    "gemini-2.0-flash": ("google", "gemini-2.0-flash"),
    "gemini-2.0-pro": ("google", "gemini-2.0-pro-exp-02-05"),
    "openrouter/o1": ("openrouter", "o1"),
    "openrouter/claude-sonnet": ("openrouter", "anthropic/claude-sonnet-4"),
    "openrouter/gpt-4o": ("openrouter", "openai/gpt-4o"),
    "ollama/llama3": ("ollama", "llama3"),
    "ollama/mistral": ("ollama", "mistral"),
    "ollama/qwen2.5-coder": ("ollama", "qwen2.5-coder:7b"),
    "mock": ("mock", "mock"),
}

DEFAULT_MODEL = "claude-sonnet-4-20250514"


class LLMClient:
    def __init__(
        self,
        model: str = "",
        api_key: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        model = model or os.environ.get("LLM_MODEL", "") or DEFAULT_MODEL
        provider_name, resolved_model = _SUPPORTED_MODELS.get(
            model, ("openai", model)
        )
        provider_name = os.environ.get("LLM_PROVIDER", provider_name)

        self.model = resolved_model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._provider: BaseProvider = self._build_provider(provider_name)

        logger.info(
            "LLMClient initialized: provider=%s model=%s",
            provider_name, self.model
        )

    def _build_provider(self, provider: str) -> BaseProvider:
        factory: dict[str, type[BaseProvider]] = {
            "anthropic": _AnthropicProvider,
            "openai": _OpenAIProvider,
            "google": _GoogleProvider,
            "openrouter": _OpenRouterProvider,
            "ollama": _OllamaProvider,
            "mock": _MockProvider,
        }
        cls = factory.get(provider)
        if cls is None:
            logger.warning("Unknown provider '%s', falling back to mock", provider)
            return _MockProvider()
        try:
            return cls(api_key=self.api_key)
        except TypeError:
            return cls()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
    ) -> LLMResponse:
        tracer = get_tracer()
        tool_names = [t["name"] for t in tools] if tools else []
        with tracer.span("llm.chat", kind="llm", attributes={
            "model": self.model,
            "provider": self.provider_name,
            "num_tools": len(tool_names),
            "tools": tool_names,
            "system_len": len(system) if system else 0,
            "messages": len(messages),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }) as span:
            response = self._provider.chat(
                messages=messages,
                tools=tools,
                system=system,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            span.set_attribute("output_tokens", response.usage.get("output_tokens", 0))
            span.set_attribute("input_tokens", response.usage.get("input_tokens", 0))
            span.set_attribute("total_tokens",
                               response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0))
            span.set_attribute("num_tool_calls", len(response.tool_calls))
            span.set_attribute("finish_reason", response.finish_reason)
            span.set_attribute("content_length", len(response.content) if response.content else 0)
            span.set_attribute("has_content", response.content is not None)
            return response

    @property
    def provider_name(self) -> str:
        return type(self._provider).__name__.replace("Provider", "").lower()


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class _AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs = dict(model=model, max_tokens=max_tokens, temperature=temperature, messages=messages)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        content_parts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        return LLMResponse(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            finish_reason=response.stop_reason,
            model=model,
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class _OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)

        body = dict(model=model, max_tokens=max_tokens, temperature=temperature)

        if system:
            body["messages"] = [{"role": "system", "content": system}] + messages
        else:
            body["messages"] = messages

        if tools:
            body["tools"] = _openai_tools(tools)

        response = client.chat.completions.create(**body)
        choice = response.choices[0]
        msg = choice.message

        content_parts = [msg.content] if msg.content else []
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = {}
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})

        usage = response.usage
        return LLMResponse(
            content="\n".join(content_parts) if content_parts else None,
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "",
            model=model,
        )


def _openai_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool schemas to OpenAI tool format."""
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class _GoogleProvider(BaseProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)

        client = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=dict(max_output_tokens=max_tokens, temperature=temperature),
            tools=_gemini_tools(tools) if tools else None,
        )

        history = [_convert_gemini_msg(m) for m in messages]
        chat = client.start_chat(history=history[:-1]) if len(history) > 1 else client.start_chat()
        last = history[-1]

        response = chat.send_message(last["parts"] if isinstance(last, dict) else last)
        candidate = response.candidates[0]
        content = candidate.content

        text = ""
        tool_calls = []
        for part in content.parts:
            if part.text:
                text += part.text
            if part.function_call:
                tc = part.function_call
                tool_calls.append({
                    "id": tc.name,
                    "name": tc.name,
                    "input": dict(tc.args.items()),
                })

        usage = response.usage_metadata
        return LLMResponse(
            content=text or None,
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.prompt_token_count if usage else 0,
                "output_tokens": usage.candidates_token_count if usage else 0,
            },
            finish_reason=candidate.finish_reason.name if candidate.finish_reason else "",
            model=model,
        )


def _gemini_tools(tools: list[dict]) -> list[dict]:
    import google.ai.generativelanguage as glm
    func_decls = []
    for t in tools:
        func_decls.append(glm.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("input_schema", {"type": "object", "properties": {}}),
        ))
    return [glm.Tool(function_declarations=func_decls)]


def _convert_gemini_msg(msg: dict) -> dict:
    role = msg["role"]
    content = msg.get("content", "")

    # Assistant message with text + optional tool_calls
    if role == "assistant":
        if isinstance(content, dict) and "tool_calls" in content:
            parts = []
            if content.get("text"):
                parts.append({"text": content["text"]})
            for tc in content["tool_calls"]:
                parts.append({
                    "function_call": {
                        "name": tc["name"],
                        "args": tc["input"],
                    }
                })
            return {"role": "model", "parts": parts}
        return {"role": "model", "parts": [{"text": content}] if isinstance(content, str) else content}

    if role == "system":
        return {"role": "user", "parts": [{"text": content}] if isinstance(content, str) else content}

    # Handle tool_result blocks (list content type from tool responses)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append({
                    "function_response": {
                        "name": block.get("tool_use_id", ""),
                        "response": {"content": block.get("content", "")},
                    }
                })
            elif isinstance(block, dict):
                parts.append({"text": json.dumps(block)})
            else:
                parts.append({"text": str(block)})
        return {"role": "user", "parts": parts}

    return {"role": role, "parts": [{"text": content}] if isinstance(content, str) else content}


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------

class _OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        from openai import OpenAI
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        body = dict(model=model, max_tokens=max_tokens, temperature=temperature)

        if system:
            body["messages"] = [{"role": "system", "content": system}] + messages
        else:
            body["messages"] = messages

        if tools:
            body["tools"] = _openai_tools(tools)

        response = client.chat.completions.create(**body)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = {}
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "input": args})

        usage = response.usage
        return LLMResponse(
            content=msg.content or None,
            tool_calls=tool_calls,
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
            finish_reason=choice.finish_reason or "",
            model=model,
        )


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class _OllamaProvider(BaseProvider):
    def __init__(self, api_key: str | None = None):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import httpx

        body = dict(
            model=model,
            stream=False,
            options=dict(num_predict=max_tokens, temperature=temperature),
        )

        if system:
            body["system"] = system

        msgs = []
        for m in messages:
            msgs.append({"role": m["role"], "content": m.get("content", "")})
        body["messages"] = msgs

        if tools:
            body["tools"] = _ollama_tools(tools)

        response = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=120)
        response.raise_for_status()
        data = response.json()

        tool_calls = []
        if "tool_calls" in data.get("message", {}):
            for tc in data["message"]["tool_calls"]:
                tool_calls.append({
                    "id": tc["function"]["name"],
                    "name": tc["function"]["name"],
                    "input": tc["function"].get("arguments", {}),
                })

        return LLMResponse(
            content=data["message"].get("content") or None,
            tool_calls=tool_calls,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            finish_reason=data.get("done_reason", ""),
            model=model,
        )


def _ollama_tools(tools: list[dict]) -> list[dict]:
    converted = []
    for t in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return converted


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

class _MockProvider(BaseProvider):
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return LLMResponse(
            content=json.dumps({
                "classification": "unclear",
                "confidence": 0.5,
                "explanation": "Mock response — no LLM configured",
            }),
            usage={"input_tokens": 0, "output_tokens": 0},
            finish_reason="end_turn",
            model="mock",
        )


# ---------------------------------------------------------------------------
# Convenience: provider auto-detection from model name
# ---------------------------------------------------------------------------

def resolve_model(model: str) -> str:
    """Resolve a short model alias to its full model name."""
    if model in _SUPPORTED_MODELS:
        return _SUPPORTED_MODELS[model][1]
    return model


def list_supported_models() -> list[dict]:
    """Return all known model aliases with their provider."""
    return [
        {"alias": alias, "provider": prov, "model": model}
        for alias, (prov, model) in sorted(_SUPPORTED_MODELS.items())
    ]
