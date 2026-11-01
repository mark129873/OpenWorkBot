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
            result = await session.call_tool("rag_list", {})
            assert not result.isError
            data = result.structuredContent or json.loads(result.content[0].text)
            assert data["documents"] == []


@pytest.mark.asyncio
async def test_cancel_stops_matching_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
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
