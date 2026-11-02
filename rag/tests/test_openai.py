import json
import traceback
from types import SimpleNamespace

import anyio
import httpx
import pytest
from openai import AsyncOpenAI

from app import models
from app.config import Settings, settings


def response(status="completed", text="测试回答"):
    return {"id": "response-test", "object": "response", "created_at": 1, "model": "gpt-6-luna",
            "status": status, "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
            "output": [{"id": "message-test", "type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}]}


def test_openai_environment_uses_selected_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test.invalid/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-key")
    monkeypatch.setenv("MODEL_ID", "gpt-6-luna")
    config = settings()
    assert (config.provider, config.api_key, config.base_url, config.model) == (
        "openai", "test-key", "http://test.invalid/v1", "gpt-6-luna")


@pytest.mark.parametrize("value,expected", [(None, "none"), ("  ", "none"), (" HIGH ", "high")])
def test_reasoning_effort_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("RAG_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("RAG_REASONING_EFFORT", value)
    assert settings().reasoning_effort == expected


@pytest.mark.parametrize("value,expected", [(None, 0), ("0.5", 0.5), ("2", 2), ("none", None), ("", None)])
def test_temperature_environment(monkeypatch, value, expected):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    if value is None:
        monkeypatch.delenv("RAG_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("RAG_TEMPERATURE", value)
    assert settings().temperature == expected


@pytest.mark.parametrize("provider,value", [("openai", "bad"), ("openai", "nan"), ("openai", "-0.1"),
                                            ("openai", "2.1"), ("anthropic", "2.1")])
def test_invalid_temperature_environment(monkeypatch, provider, value):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("RAG_TEMPERATURE", value)
    with pytest.raises(ValueError, match="RAG_TEMPERATURE"):
        settings()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "incomplete", "failed"])
@pytest.mark.parametrize("effort,temperature", [("none", "0"), ("none", "0.5"), ("max", "none")])
async def test_rag_openai_text_uses_responses(monkeypatch, status, effort, temperature):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("MODEL_ID", "gpt-6-luna")
    monkeypatch.setenv("RAG_REASONING_EFFORT", effort)
    monkeypatch.setenv("RAG_TEMPERATURE", temperature)
    config = settings()
    requests = []

    def handler(request):
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response(status))

    def client(**kwargs):
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(models, "AsyncOpenAI", client)
    if status == "completed":
        assert await models.Models(config).text("系统", "问题") == "测试回答"
    else:
        with pytest.raises(RuntimeError, match="未完成"):
            await models.Models(config).text("系统", "问题")
    if temperature == "none":
        assert "temperature" not in requests[0]
    else:
        assert requests[0]["temperature"] == float(temperature)
    assert requests[0]["reasoning"] == {"effort": effort} and requests[0]["store"] is False
    assert requests[0]["max_output_tokens"] == models.params.MAX_OUTPUT_TOKENS
    assert requests[0]["input"] == [{"role": "system", "content": "系统"}, {"role": "user", "content": "问题"}]


@pytest.mark.asyncio
async def test_rag_structured_response_and_empty_output(tmp_path, monkeypatch):
    config = Settings(tmp_path, tmp_path, "", "", "", "", "", "", "", "test-key",
                      "http://test.invalid/v1", "gpt-6-luna", "openai", "max", None)
    output = ['{"sufficient": true}', ""]
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=response(text=output.pop(0))))
    monkeypatch.setattr(models, "AsyncOpenAI", lambda **kwargs: AsyncOpenAI(
        **kwargs, http_client=httpx.AsyncClient(transport=transport)))
    client = models.Models(config)
    assert await client.structured("系统", "问题", {}) == {"sufficient": True}
    with pytest.raises(RuntimeError, match="没有返回内容"):
        await client.text("系统", "问题")


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", ["0.5", "1.5", "none"])
async def test_rag_anthropic_text_uses_configured_temperature(monkeypatch, temperature):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://test.invalid")
    monkeypatch.setenv("MODEL_ID", "test-model")
    monkeypatch.setenv("RAG_TEMPERATURE", temperature)
    requests = []

    class Client:
        def __init__(self, **kwargs):
            self.messages = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(stop_reason="end_turn", usage=None,
                                   content=[SimpleNamespace(type="text", text="测试回答")])

    monkeypatch.setattr(models, "AsyncAnthropic", Client)
    assert await models.Models(settings()).text("系统", "问题") == "测试回答"
    if temperature == "none":
        assert "temperature" not in requests[0]
    else:
        assert requests[0]["temperature"] == float(temperature)


@pytest.mark.asyncio
async def test_rag_api_errors_do_not_expose_key(tmp_path, monkeypatch):
    config = Settings(tmp_path, tmp_path, "", "", "", "", "", "", "", "test-key",
                      "http://test.invalid/v1", "gpt-6-luna", "openai")
    transport = httpx.MockTransport(lambda _request: httpx.Response(
        401, json={"error": {"message": "test-key", "code": "invalid_api_key"}}))
    monkeypatch.setattr(models, "AsyncOpenAI", lambda **kwargs: AsyncOpenAI(
        **kwargs, http_client=httpx.AsyncClient(transport=transport)))
    with pytest.raises(RuntimeError) as caught:
        await models.Models(config).text("系统", "问题")
    assert "test-key" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_rag_model_connection_retries_three_times_with_backoff(tmp_path, monkeypatch, recover):
    config = Settings(tmp_path, tmp_path, "", "", "", "", "", "", "", "test-key",
                      "http://test.invalid/v1", "test-model", "openai")
    attempts, delays = [], []

    def handler(request):
        attempts.append(request)
        if len(attempts) <= 3 or not recover:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(200, json=response())

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(models, "AsyncOpenAI", lambda **kwargs: AsyncOpenAI(
        **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    monkeypatch.setattr(anyio, "sleep", sleep)
    if recover:
        assert await models.Models(config).text("系统", "问题") == "测试回答"
    else:
        with pytest.raises(RuntimeError, match="连接错误"):
            await models.Models(config).text("系统", "问题")
    assert len(attempts) == 4
    assert len(delays) == 3 and 0 < delays[0] < delays[1] < delays[2]
