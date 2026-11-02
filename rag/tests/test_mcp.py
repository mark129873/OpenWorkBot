import asyncio
import os
import json
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.asyncio
async def test_stdio_exposes_only_knowledge_tools(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.server"], cwd=os.getcwd(), env={
        "OPENWORKBOT_WORKSPACE": str(workspace), "RAG_DATA_DIR": str(tmp_path / "data"),
        "MILVUS_CLOUD_URI": "", "MILVUS_CLOUD_TOKEN": "",
    })
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {"rag_search", "rag_ingest", "rag_list", "rag_remove", "rag_cancel"}
            search = next(tool for tool in tools.tools if tool.name == "rag_search")
            assert search.inputSchema["properties"]["include_evidence"]["type"] == "boolean"
            result = await session.call_tool("rag_list", {})
            assert not result.isError
            data = result.structuredContent or json.loads(result.content[0].text)
            assert data["documents"] == []


@pytest.mark.asyncio
async def test_benchmark_server_exposes_internal_trace_flush(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.server"], cwd=os.getcwd(), env={
        "OPENWORKBOT_WORKSPACE": str(workspace), "RAG_DATA_DIR": str(tmp_path / "data"),
        "MILVUS_CLOUD_URI": "", "MILVUS_CLOUD_TOKEN": "", "LANGFUSE_ENABLED": "false",
        "OPENWORKBOT_BENCHMARK_FLUSH": "1",
    })
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert "rag_flush_tracing" in {tool.name for tool in (await session.list_tools()).tools}
            result = await session.call_tool("rag_flush_tracing", {})
            assert not result.isError and json.loads(result.content[0].text) == {"status": "disabled"}


@pytest.mark.asyncio
async def test_trace_flush_uses_server_process_client(monkeypatch):
    from app import server

    calls = []

    class Client:
        def flush(self):
            calls.append("flush")

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setattr(server.tracing, "configured_client", Client)
    assert await server.rag_flush_tracing() == {"status": "success"}
    assert calls == ["flush"]


@pytest.mark.asyncio
async def test_cancel_stops_matching_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    from app import server

    started = asyncio.Event()

    class SlowKnowledge:
        async def search(self, query):
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(server, "knowledge", SlowKnowledge())
    task = asyncio.create_task(server.rag_search("问题", "op-one"))
    await started.wait()
    assert await server.rag_cancel("op-one") == {"status": "cancelled"}
    assert task.cancelled() and "op-one" not in server.operations
    assert await server.rag_cancel("op-one") == {"status": "not_running"}
