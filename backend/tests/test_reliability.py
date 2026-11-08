import asyncio
from types import SimpleNamespace

import pytest

from app.agent import compact
from app.agent.llm import AnthropicStreamClient


def collect(generator):
    async def run():
        return [event async for event in generator]
    return asyncio.run(run())


class FakeClient:
    model = "fake"

    def __init__(self, steps):
        self.steps = iter(steps)

    async def stream(self, **_kwargs):
        for event in next(self.steps):
            if isinstance(event, Exception):
                raise event
            yield event


def test_truncated_compaction_summary_is_rejected():
    client = FakeClient([[{"type": "text_delta", "text": "不完整摘要"},
                          {"type": "stop_reason", "reason": "max_tokens"}]])
    with pytest.raises(ValueError, match="摘要输出达到上限"):
        asyncio.run(compact.summarize_history(client, [{"role": "user", "content": "请求"}]))


def test_stream_emits_stop_reason():
    def obj(**kwargs):
        return SimpleNamespace(**kwargs)

    class Stream:
        def __init__(self, **kwargs):
            assert kwargs["temperature"] == 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def __aiter__(self):
            yield obj(type="message_start", message=obj(usage=obj(input_tokens=8, output_tokens=0)))
            yield obj(type="message_delta", usage=obj(output_tokens=5), delta=obj(stop_reason="max_tokens"))
            yield obj(type="message_stop")

    client = object.__new__(AnthropicStreamClient)
    client._client = obj(messages=obj(stream=Stream))
    client.model, client._max_tokens = "fake", 100
    events = collect(client.stream(system="", messages=[], tools=[]))
    snapshots = [event for event in events if event["type"] == "usage"]
    assert [(e["input_tokens"], e["output_tokens"]) for e in snapshots] == [(8, 5)]
    assert {"type": "stop_reason", "reason": "max_tokens"} in events
