import json
import traceback

import httpx
import pytest
from openai import AsyncOpenAI

from app import models
from app.config import Settings, settings


def test_openai_environment_uses_selected_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://test.invalid/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "other-key")
    monkeypatch.setenv("MODEL_ID", "gpt-6-luna")
    config = settings()
    assert (config.provider, config.api_key, config.base_url, config.model) == (
        "openai", "test-key", "http://test.invalid/v1", "gpt-6-luna")


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["stop", "length"])
async def test_rag_openai_text_uses_chat_completions(tmp_path, monkeypatch, finish):
    config = Settings(tmp_path, tmp_path, "", "", "", "", "", "", "", "test-key",
                      "", "gpt-6-luna", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "chat-test", "object": "chat.completion", "created": 1,
            "model": "gpt-6-luna", "choices": [{"index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": "测试回答"}}]})

    def client(**kwargs):
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(models, "AsyncOpenAI", client)
    if finish == "stop":
        assert await models.Models(config).text("系统", "问题") == "测试回答"
    else:
        with pytest.raises(RuntimeError, match="未完成"):
            await models.Models(config).text("系统", "问题")
    assert requests[0]["reasoning_effort"] == "none" and requests[0]["store"] is False
    assert requests[0]["messages"] == [{"role": "system", "content": "系统"}, {"role": "user", "content": "问题"}]


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
