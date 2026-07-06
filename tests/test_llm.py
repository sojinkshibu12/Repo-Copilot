import pytest
from core.llm import LLMClient, list_supported_models, resolve_model


def test_mock_provider_works():
    client = LLMClient(model="mock")
    response = client.chat(messages=[{"role": "user", "content": "hello"}])
    assert response.content is not None
    assert response.finish_reason == "end_turn"
    assert response.model == "mock"


def test_provider_name():
    client = LLMClient(model="mock")
    assert "mock" in client.provider_name


def test_list_supported_models():
    models = list_supported_models()
    assert len(models) > 0
    assert any(m["alias"] == "mock" for m in models)
    assert any(m["alias"] == "gpt-4o" for m in models)
    assert any(m["alias"] == "claude-sonnet-4" for m in models)


def test_resolve_model():
    resolved = resolve_model("gpt-4o")
    assert resolved == "gpt-4o"

    resolved = resolve_model("claude-sonnet-4")
    assert "claude-sonnet" in resolved


def test_response_dataclass():
    from core.llm import LLMResponse
    r = LLMResponse(
        content="hello",
        tool_calls=[{"id": "1", "name": "test", "input": {}}],
        usage={"input_tokens": 10, "output_tokens": 20},
        finish_reason="end_turn",
        model="gpt-4o",
    )
    assert r.content == "hello"
    assert len(r.tool_calls) == 1
    assert r.usage["input_tokens"] == 10
