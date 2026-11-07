import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from app import db
from app.agent.permission_service import check_permission
from app.agent.rag_client import RagClient
from app.agent.runtime import Conflict, Runtime
from app.agent.task_tools import DIRECT_TOOLS, MAIN_TOOLS, RAG_TOOLS, WORKER_TOOLS
from app.agent.worker import Worker
from app.api.chat import router
from app.api.uploads import MAX_UPLOAD, upload
from app.config import settings
from app.sessions import chat_store as store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    (tmp_path / ("a" * 32)).mkdir()
    db.dispose_engine()
    db.init_db(tmp_path)
    yield tmp_path
    db.dispose_engine()


def test_upload_overwrite_replay_and_file_only_message(isolated):
    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        app = FastAPI()
        app.include_router(router)
        from conftest import install_runtime
        install_runtime(app, runtime)
        try:
            assistant = (await runtime.snapshot())["assistant_id"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def put(upload_id, content):
                    return await client.post("/api/chat/uploads", data={"assistant_id": assistant, "upload_id": upload_id},
                                             files={"file": ("a.txt", content, "text/plain")})
                first = await put("one", b"first")
                assert first.status_code == 200
                assert first.json()["path"] == "uploads/a.txt"
                assert (await put("two", b"second")).status_code == 200
                assert (await put("one", b"first")).status_code == 200
                assert (isolated / ("a" * 32) / "uploads/a.txt").read_bytes() == b"second"
                assert (await put("one", b"different")).status_code == 409
                stale = await client.post("/api/chat/messages", json={
                    "assistant_id": assistant, "client_message_id": "stale", "attachments": ["one"]})
                assert stale.status_code == 409
                payload = {"assistant_id": assistant, "client_message_id": "file-only", "attachments": ["two"]}
                accepted = await client.post("/api/chat/messages", json=payload)
                assert accepted.status_code == 202
                assert (await client.post("/api/chat/messages", json=payload)).json()["replayed"]
                async with asyncio.timeout(4):
                    while True:
                        with db.new_session() as session:
                            row = store.get(session, "input:file-only")
                        if row["status"] == "processed":
                            break
                        await asyncio.sleep(0.02)
                messages = (await runtime.snapshot())["messages"]
                assert len(messages) == 1 and messages[0]["attachments"][0]["name"] == "a.txt"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_upload_limit_and_invalid_filename(isolated, monkeypatch):
    from app.api import uploads
    assert MAX_UPLOAD == 100 * 1024 * 1024
    monkeypatch.setattr(uploads, "MAX_UPLOAD", 4)

    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        app = FastAPI()
        app.include_router(router)
        from conftest import install_runtime
        install_runtime(app, runtime)
        try:
            assistant = (await runtime.snapshot())["assistant_id"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                data = {"assistant_id": assistant, "upload_id": "one"}
                assert (await client.post("/api/chat/uploads", data=data, files={"file": ("a.txt", b"1234")})).status_code == 200
                data["upload_id"] = "two"
                assert (await client.post("/api/chat/uploads", data=data, files={"file": ("a.txt", b"12345")})).status_code == 400
                assert (isolated / ("a" * 32) / "uploads/a.txt").read_bytes() == b"1234"
                data["upload_id"] = "three"
                assert (await client.post("/api/chat/uploads", data=data, files={"file": ("../escape.txt", b"x")})).status_code == 400
        finally:
            await runtime.close()
    asyncio.run(run())


def test_actual_100_mib_upload_boundary(isolated):
    class StreamFile:
        filename = "large.bin"

        def __init__(self, size):
            self.remaining = size
            self.block = b"x" * (1024 * 1024)

        async def read(self, size):
            length = min(size, self.remaining)
            self.remaining -= length
            return self.block[:length]

        async def close(self):
            pass

    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        try:
            assistant = (await runtime.snapshot())["assistant_id"]
            accepted = await upload(runtime, assistant, "large", StreamFile(MAX_UPLOAD))
            assert accepted["size"] == MAX_UPLOAD
            with pytest.raises(ValueError, match="100 MiB"):
                await upload(runtime, assistant, "too-large", StreamFile(MAX_UPLOAD + 1))
            target = isolated / ("a" * 32) / "uploads/large.bin"
            assert target.stat().st_size == MAX_UPLOAD
        finally:
            await runtime.close()
    asyncio.run(run())


def test_reset_during_upload_does_not_replace_file(isolated):
    class SlowFile:
        filename = "a.txt"

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.count = 0

        async def read(self, _):
            self.count += 1
            if self.count == 1:
                self.started.set()
                await self.release.wait()
                return b"new"
            return b""

        async def close(self):
            pass

    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        target = isolated / ("a" * 32) / "uploads/a.txt"
        target.parent.mkdir()
        target.write_bytes(b"old")
        file = SlowFile()
        try:
            assistant = (await runtime.snapshot())["assistant_id"]
            pending = asyncio.create_task(upload(runtime, assistant, "one", file))
            await file.started.wait()
            await runtime.reset(assistant, "reset-one")
            file.release.set()
            with pytest.raises(Conflict):
                await pending
            assert target.read_bytes() == b"old"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_rag_tools_are_not_available_to_agents(isolated):
    names = {item["name"] for item in RAG_TOOLS}
    assert not names & {item["name"] for item in MAIN_TOOLS + DIRECT_TOOLS + WORKER_TOOLS}

    async def run():
        runtime = Runtime()
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "查询知识库", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        for name in names:
            result = await worker.execute({"id": name, "name": name, "input": {}})
            assert "暂不可用" in result
        await runtime.close()
    asyncio.run(run())

def test_new_user_message_blocks_old_rag_operation(isolated):
    async def run():
        runtime = Runtime()
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "将 a.txt 加入知识库", 1)
            store.put(session, "execution:old", {"type": "tool_execution", "id": "old", "status": "pending"})
            store.put(session, "input:new", {"type": "user_event", "event_type": "message", "user_seq": 2,
                                            "status": "pending", "user_text": "不要入库"})
            session.commit()
        worker = Worker(runtime, task["id"])
        assert "新的用户消息" in await worker.started({"id": "old"}, 1)
        with db.new_session() as session:
            assert store.get(session, "execution:old")["status"] == "pending"
        await runtime.close()
    asyncio.run(run())


def test_worker_mcp_client_uses_isolated_server(isolated, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(isolated / "rag"))
    monkeypatch.setenv("MILVUS_CLOUD_URI", "")
    monkeypatch.setenv("MILVUS_CLOUD_TOKEN", "")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "anthropic_base_url", "")
    monkeypatch.setattr(settings, "model_id", "")

    async def run():
        client = RagClient()
        try:
            first, second = await asyncio.wait_for(asyncio.gather(
                client.call("rag_list", {}), client.call("rag_list", {})), 60)
            assert '"documents": []' in first and '"documents": []' in second
            assert client.session is not None
        finally:
            await client.close()
    asyncio.run(run())


def test_worker_reports_mcp_start_timeout(monkeypatch):
    from app.agent import rag_client
    monkeypatch.setattr(rag_client, "START_TIMEOUT", 0.01)

    async def run():
        client = RagClient()

        async def stalled():
            await asyncio.Event().wait()

        monkeypatch.setattr(client, "_run", stalled)
        assert "启动超时" in await client.call("rag_list", {})
        await asyncio.gather(client.task, return_exceptions=True)
        await client.close()
    asyncio.run(run())
