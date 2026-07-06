# Multi-LLM Provider Support

Repo Copilot supports six LLM providers via a unified interface.

## Quick Switch

Set two environment variables:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
```

Or pass directly:

```python
client = LLMClient(model="gpt-4o")
```

## Supported Providers & Models

| Provider   | `LLM_PROVIDER` | Example Models                                     |
|------------|----------------|----------------------------------------------------|
| Anthropic  | `anthropic`    | `claude-sonnet-4`, `claude-3-5-haiku`             |
| OpenAI     | `openai`       | `gpt-4o`, `gpt-4o-mini`, `o3-mini`               |
| Google     | `google`       | `gemini-2.0-flash`, `gemini-2.0-pro`              |
| OpenRouter | `openrouter`   | `openrouter/o1`, `openrouter/claude-sonnet`       |
| Ollama     | `ollama`       | `ollama/llama3`, `ollama/qwen2.5-coder`           |
| Mock       | `mock`         | `mock` (no API key needed)                         |

## Provider-Specific Setup

### Anthropic
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_PROVIDER=anthropic
export LLM_MODEL=claude-sonnet-4
```

### OpenAI
```bash
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o
```

### Google Gemini
```bash
export GOOGLE_API_KEY=...
export LLM_PROVIDER=google
export LLM_MODEL=gemini-2.0-flash
```

### OpenRouter
```bash
export OPENROUTER_API_KEY=sk-or-...
export LLM_PROVIDER=openrouter
export LLM_MODEL=openrouter/claude-sonnet
```

### Ollama (local)
```bash
export OLLAMA_BASE_URL=http://localhost:11434
export LLM_PROVIDER=ollama
export LLM_MODEL=ollama/qwen2.5-coder
```

### Mock (offline development)
```bash
export LLM_PROVIDER=mock
export LLM_MODEL=mock
```

## Architecture

```
LLMClient (unified entry point)
├── _AnthropicProvider   → anthropic SDK, native tool use
├── _OpenAIProvider      → openai SDK, function calling
├── _GoogleProvider      → google-generativeai SDK, function calling
├── _OpenRouterProvider  → openai-compatible SDK, custom base_url
├── _OllamaProvider      → httpx → local /api/chat endpoint
└── _MockProvider        → no network, deterministic mock responses
```

All providers share the same `BaseProvider` ABC and return `LLMResponse` dataclasses — the `Orchestrator` never touches provider-specific code.

## Aliases vs Raw Models

Short aliases (`gpt-4o`, `claude-sonnet-4`) auto-resolve via `_SUPPORTED_MODELS`.
Any unrecognized model string is passed verbatim to the provider — so you can use any model a provider supports even if it's not in the alias table.

## Best Practice: Cost-Aware Routing

Use cheaper models for classification, expensive models for code generation:

```python
classifier = LLMClient(model="gpt-4o-mini")   # cheap
coder = LLMClient(model="claude-sonnet-4")      # capable
```
