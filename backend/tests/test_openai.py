import asyncio
import json
import os
import traceback
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI

from app import db
from app.agent.compact import is_context_limit_error
from app.agent.llm import OpenAIStreamClient, build_client, openai_messages
from app.agent.rag_client import RagClient
from app.agent.runtime import Runtime
from app.agent.worker import run_worker
from app.config import Settings, settings, validate_llm_config
from app.sandbox.lease import SandboxLeaseManager
from app.sessions import chat_store as store
from fake_sandbox import FakeProvider


READ_TOOL = {"name": "read_file", "description": "读取文件", "input_schema": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}


def chunk(delta=None, finish=None, usage=None):
    return {"id": "chat-test", "object": "chat.completion.chunk", "created": 1, "model": "gpt-6-luna",
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if usage is None else [],
            **({"usage": usage} if usage is not None else {})}


def response(chunks):
    content = "".join(f"data: {json.dumps(item, ensure_ascii=False)}\n\n" for item in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=content)


def make_client(handler):
    client = object.__new__(OpenAIStreamClient)
    client.model, client._max_tokens = "gpt-6-luna", 100
    client._client = AsyncOpenAI(api_key="test-key", base_url="http://test.invalid/v1", max_retries=0,
                                 http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return client


def test_provider_selection_and_configuration(monkeypatch):
    assert validate_llm_config(Settings(model_id="x")) == ["OPENAI_API_KEY"]
    assert validate_llm_config(Settings(llm_provider="openai", model_id="x")) == ["OPENAI_API_KEY"]
    assert validate_llm_config(Settings(llm_provider="other", model_id="x")) == ["LLM_PROVIDER"]
    config = Settings(llm_provider="openai", openai_api_key="test-key", model_id="gpt-6-luna")
    assert not validate_llm_config(config)
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    client = build_client(config)
    assert isinstance(client, OpenAIStreamClient)
    assert str(client._client.base_url) == "https://api.openai.com/v1/"
    asyncio.run(client._client.close())


def test_history_conversion_pairs_tool_results_and_preserves_errors():
    history = [
        {"role": "user", "content": "检查文件"},
        {"role": "assistant", "content": [{"type": "text", "text": "先读取。"},
            {"type": "tool_use", "id": "one", "name": "read_file", "input": {"path": "a.txt"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "one",
                                      "content": "用户拒绝", "is_error": True}]},
    ]
    converted = openai_messages("系统", history)
    assert converted[2]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": '{"path": "a.txt"}'}
    assert converted[3] == {"role": "tool", "tool_call_id": "one", "content": "Error: 用户拒绝"}
    with pytest.raises(ValueError, match="缺少结果"):
        openai_messages("系统", history[:-1])


def test_openai_stream_aggregates_and_validates_before_emitting():
    captured = []
    parts = [
        chunk({"content": "读取中。", "tool_calls": [{"index": 0, "id": "call-a", "type": "function",
              "function": {"name": "read_file", "arguments": '{"pa'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'th":"a.txt"}'}}]}),
        chunk(finish="tool_calls"),
        chunk(usage={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24,
                     "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2}}),
    ]

    def handler(request):
        captured.append(json.loads(request.content))
        return response(parts)

    async def run():
        client = make_client(handler)
        try:
            return [event async for event in client.stream(system="系统", messages=[{"role": "user", "content": "读取"}],
                    tools=[READ_TOOL], tool_choice={"type": "tool", "name": "read_file"})]
        finally:
            await client._client.close()

    events = asyncio.run(run())
    request = captured[0]
    assert request["temperature"] == 0
    assert request["reasoning_effort"] == "none" and request["store"] is False
    assert request["tool_choice"] == {"type": "function", "function": {"name": "read_file"}}
    assert request["tools"][0]["function"]["strict"] is False
    assert {"type": "tool_use", "id": "call-a", "name": "read_file", "input": {"path": "a.txt"}} in events
    assert {"type": "usage", "input_tokens": 20, "output_tokens": 4,
            "cache_read_tokens": 3, "cache_creation_tokens": 2} in events


@pytest.mark.parametrize("arguments,finish", [
    ('{"path":', "tool_calls"), ('{"other":1}', "tool_calls"),
    ('{"path":"a", "extra":1}', "tool_calls"), ('{"path":"a"}', None),
])
def test_incomplete_or_invalid_tools_never_emit(arguments, finish):
    parts = [chunk({"tool_calls": [{"index": 0, "id": "call-a", "type": "function",
                     "function": {"name": "read_file", "arguments": arguments}}]})]
    if finish:
        parts.append(chunk(finish=finish))

    async def run():
        client = make_client(lambda _request: response(parts))
        events = []
        try:
            with pytest.raises(ValueError):
                async for event in client.stream(system="", messages=[], tools=[READ_TOOL]):
                    events.append(event)
            assert not any(event["type"] == "tool_use" for event in events)
        finally:
            await client._client.close()
    asyncio.run(run())


def test_truncated_tool_is_not_emitted():
    parts = [chunk({"tool_calls": [{"index": 0, "id": "call-a", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"a"}'}}]}), chunk(finish="length")]

    async def run():
        client = make_client(lambda _request: response(parts))
        try:
            events = [event async for event in client.stream(system="", messages=[], tools=[READ_TOOL])]
            assert {"type": "stop_reason", "reason": "max_tokens"} in events
            assert not any(event["type"] == "tool_use" for event in events)
        finally:
            await client._client.close()
    asyncio.run(run())


@pytest.mark.parametrize("status,code", [(401, "invalid_api_key"), (400, "context_length_exceeded")])
def test_api_errors_do_not_expose_key_and_keep_context_limit(status, code):
    def handler(_request):
        return httpx.Response(status, json={"error": {"message": "test-key", "code": code}})

    async def run():
        client = make_client(handler)
        try:
            with pytest.raises(RuntimeError) as caught:
                async for _event in client.stream(system="", messages=[], tools=[]):
                    pass
            assert "test-key" not in "".join(traceback.format_exception(caught.value))
            assert is_context_limit_error(caught.value) == (code == "context_length_exceeded")
        finally:
            await client._client.close()
    asyncio.run(run())


def test_invalid_call_in_batch_does_not_emit_other_calls():
    parts = [chunk({"tool_calls": [
        {"index": 0, "id": "first", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":"a"}'}},
        {"index": 1, "id": "second", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":'}},
    ]}), chunk(finish="tool_calls")]

    async def run():
        client = make_client(lambda _request: response(parts))
        events = []
        try:
            with pytest.raises(ValueError):
                async for event in client.stream(system="", messages=[], tools=[READ_TOOL]):
                    events.append(event)
            assert not any(event["type"] == "tool_use" for event in events)
        finally:
            await client._client.close()
    asyncio.run(run())


def test_multiple_tool_calls_keep_index_order():
    parts = [chunk({"tool_calls": [
        {"index": 1, "id": "second", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":"b"}'}},
        {"index": 0, "id": "first", "type": "function",
         "function": {"name": "read_file", "arguments": '{"path":"a"}'}},
    ]}), chunk(finish="tool_calls")]

    async def run():
        client = make_client(lambda _request: response(parts))
        try:
            events = [event async for event in client.stream(system="", messages=[], tools=[READ_TOOL])]
            assert [event["id"] for event in events if event["type"] == "tool_use"] == ["first", "second"]
        finally:
            await client._client.close()
    asyncio.run(run())


@pytest.mark.parametrize("provider,prefix", [("openai", "OPENAI"), ("anthropic", "ANTHROPIC")])
def test_rag_process_receives_selected_provider(tmp_path, monkeypatch, provider, prefix):
    import mcp
    import mcp.client.stdio
    from app.agent import rag_client

    executable = tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(rag_client, "ROOT", tmp_path)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, f"{provider}_api_key", "test-key")
    monkeypatch.setattr(settings, f"{provider}_base_url", "http://test.invalid/v1")
    monkeypatch.setattr(settings, "model_id", "model-test")
    monkeypatch.setenv("RAG_REASONING_EFFORT", "high")
    monkeypatch.setenv("RAG_TEMPERATURE", "none")
    captured = []

    @asynccontextmanager
    async def fake_stdio(params):
        captured.append(params.env)
        yield None, None

    class Session:
        def __init__(self, *_args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def initialize(self):
            pass

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio)
    monkeypatch.setattr(mcp, "ClientSession", Session)

    async def run():
        client = RagClient()
        task = asyncio.create_task(client._run())
        await asyncio.wait_for(client.ready.wait(), 2)
        client.stop.set()
        await task
    asyncio.run(run())
    assert captured[0]["LLM_PROVIDER"] == provider
    assert captured[0][f"{prefix}_API_KEY"] == "test-key"
    assert captured[0][f"{prefix}_BASE_URL"] == "http://test.invalid/v1"
    assert captured[0]["MODEL_ID"] == "model-test"
    assert captured[0]["RAG_REASONING_EFFORT"] == "high"
    assert captured[0]["RAG_TEMPERATURE"] == "none"


def test_main_direct_action_with_mock_openai(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    (tmp_path / ("a" * 32)).mkdir()
    (tmp_path / ("a" * 32) / "note.txt").write_text("隔离测试内容", encoding="utf-8")
    db.dispose_engine()
    db.init_db(tmp_path)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            args = {"path": "/mnt/user-data/workspace/note.txt"}
            return response([chunk({"tool_calls": [{"index": 0, "id": "call-read", "type": "function",
                            "function": {"name": "read_file", "arguments": json.dumps(args)}}]}),
                            chunk(finish="tool_calls")])
        return response([chunk({"content": "已读取。"}), chunk(finish="stop")])

    async def run():
        client = make_client(handler)
        runtime = Runtime(client=client, sandbox_leases=SandboxLeaseManager(FakeProvider()))
        await runtime.start()
        try:
            await runtime.accept("openai-test", "读取 note.txt", "build",
                                 assistant_id=(await runtime.snapshot())["assistant_id"])
            async with asyncio.timeout(5):
                while len(requests) < 2:
                    await asyncio.sleep(0.02)
            assert "隔离测试内容" in json.dumps(requests[1]["messages"], ensure_ascii=False)
            assert any(item["role"] == "tool" for item in requests[1]["messages"])
        finally:
            await runtime.close()
            await client._client.close()
    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_worker_tool_round_trip_with_mock_openai(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    (tmp_path / ("a" * 32)).mkdir()
    (tmp_path / ("a" * 32) / "note.txt").write_text("后台测试内容", encoding="utf-8")
    db.dispose_engine()
    db.init_db(tmp_path)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return response([chunk({"tool_calls": [{"index": 0, "id": "call-read", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path":"/mnt/user-data/workspace/note.txt"}'}}]}),
                            chunk(finish="tool_calls")])
        return response([chunk({"content": "处理完成。"}), chunk(finish="stop")])

    async def run():
        client = make_client(handler)
        runtime = Runtime(client=client, sandbox_leases=SandboxLeaseManager(FakeProvider()))
        with db.new_session() as session:
            store.initialize(session, tmp_path)
            task = runtime.create_task(session, "读取 note.txt", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        try:
            await run_worker(runtime, task["id"])
            assert len(requests) == 2
            assert "后台测试内容" in json.dumps(requests[1]["messages"], ensure_ascii=False)
            assert any(item["role"] == "tool" for item in requests[1]["messages"])
            with db.new_session() as session:
                assert store.get(session, f"task:{task['id']}")["status"] == "reviewing"
        finally:
            await runtime.close()
            await client._client.close()
    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()
