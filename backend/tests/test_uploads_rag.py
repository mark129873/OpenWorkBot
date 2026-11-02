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
    (tmp_path / "workspace").mkdir()
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
        app.state.runtime = runtime
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
                assert (isolated / "workspace/uploads/a.txt").read_bytes() == b"second"
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
        app.state.runtime = runtime
        try:
            assistant = (await runtime.snapshot())["assistant_id"]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                data = {"assistant_id": assistant, "upload_id": "one"}
                assert (await client.post("/api/chat/uploads", data=data, files={"file": ("a.txt", b"1234")})).status_code == 200
                data["upload_id"] = "two"
                assert (await client.post("/api/chat/uploads", data=data, files={"file": ("a.txt", b"12345")})).status_code == 400
                assert (isolated / "workspace/uploads/a.txt").read_bytes() == b"1234"
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
            target = isolated / "workspace/uploads/large.bin"
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
        target = isolated / "workspace/uploads/a.txt"
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


def test_rag_tools_are_worker_only_and_respect_rules(isolated):
    names = {item["name"] for item in RAG_TOOLS}
    assert names <= {item["name"] for item in WORKER_TOOLS}
    assert not names & {item["name"] for item in MAIN_TOOLS + DIRECT_TOOLS}
    assert check_permission("build", {"deny": [{"toolName": "rag_remove"}]}, "rag_remove", {"document_id": "a"})["decision"] == "deny"
    assert check_permission("build", {"ask": [{"toolName": "rag_ingest"}]}, "rag_ingest", {"path": "a.txt"})["decision"] == "ask"
    assert check_permission("build", {}, "rag_ingest", {"path": "../outside.txt"})["decision"] == "deny"
    assert check_permission("build", {}, "rag_ingest", {"path": "a.txt"})["decision"] == "allow"

    with db.new_session() as session:
        store.initialize(session, isolated)
        runtime = Runtime()
        task = runtime.create_task(session, "将 a.txt 加入知识库", 1)
        store.put(session, "input:one", {"type": "user_event", "event_type": "message", "user_seq": 1,
                                         "user_text": "将 a.txt 加入知识库"})
        session.commit()
    worker = Worker(runtime, task["id"])
    assert worker.knowledge_authorized("rag_ingest")
    assert not worker.knowledge_authorized("rag_remove")
    with db.new_session() as session:
        store.put(session, "input:one", {"type": "user_event", "event_type": "message", "user_seq": 1,
                                         "user_text": "查询知识库，不要将 a.txt 入库"})
        session.commit()
    assert not worker.knowledge_authorized("rag_ingest")


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


def test_simple_rag_answer_is_delivered_without_main_model(isolated):
    async def run():
        runtime = Runtime()
        runtime.client = None
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "查询知识库问题", 1, knowledge_question=True)
            task.update(status="running", run_id="run-one")
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:rag", {"type": "tool_execution", "id": "rag", "task_id": task["id"],
                      "context_id": task["context_id"], "run_id": "run-one", "name": "rag_search",
                      "status": "completed", "output": json.dumps({"status": "success",
                      "answer": "事实。"}, ensure_ascii=False)})
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run-one"
        await worker.finish("result", "完成")
        with db.new_session() as session:
            report = store.records(session, "agent_message", context_id="main")[0]
            assert report["kind"] == "rag_answer"
        await runtime._report(report)
        snapshot = await runtime.snapshot()
        assert snapshot["messages"][-1]["text"] == "事实。"
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}")["status"] == "completed"
        await runtime.close()
    asyncio.run(run())


def test_unknown_rag_write_is_not_replayed_without_new_user_request(isolated):
    class FakeRag:
        async def call(self, name, args):
            return '{"status":"success"}'

        async def close(self):
            pass

    async def run():
        runtime = Runtime()
        runtime.rag = FakeRag()
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "将 a.txt 加入知识库", 1)
            task.update(status="running", run_id="run-one")
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "input:one", {"type": "user_event", "event_type": "message", "user_seq": 1,
                                             "user_text": "将 a.txt 加入知识库"})
            prior = {"type": "tool_execution", "id": "prior", "task_id": "previous-task",
                     "context_id": task["context_id"], "run_id": "run-one", "name": "rag_ingest",
                     "input": {"path": "a.txt"}, "status": "completed", "result_unknown": True, "user_seq": 1}
            current = {**prior, "id": "current", "task_id": task["id"], "status": "pending",
                       "result_unknown": False, "call_id": "call"}
            store.put(session, "execution:prior", prior)
            store.put(session, "execution:current", current)
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run-one"
        assert "等待用户明确要求重试" in await worker.execute(current)
        with db.new_session() as session:
            task = store.get(session, f"task:{task['id']}")
            store.put(session, "input:two", {"type": "user_event", "event_type": "message", "user_seq": 2,
                                             "user_text": "继续任务"})
            runtime._send_update(session, task, "继续任务", "update-two", 2)
            for message in store.records(session, "agent_message", context_id=task["context_id"]):
                message["delivery"] = "processed"
                store.put(session, f"bus:{message['id']}", message)
            session.commit()
        assert "等待用户明确要求重试" in await worker.execute(current)
        with db.new_session() as session:
            task = store.get(session, f"task:{task['id']}")
            store.put(session, "input:three", {"type": "user_event", "event_type": "message", "user_seq": 3,
                                               "user_text": "再次将 a.txt 加入知识库"})
            runtime._send_update(session, task, "再次将 a.txt 加入知识库", "update-three", 3)
            for message in store.records(session, "agent_message", context_id=task["context_id"]):
                message["delivery"] = "processed"
                store.put(session, f"bus:{message['id']}", message)
            session.commit()
        assert '"success"' in await worker.execute(current)
        await runtime.close()
    asyncio.run(run())


def test_stopping_worker_cancels_inflight_rag_operation(isolated):
    class FakeRag:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = None

        async def call(self, name, args):
            self.started.set()
            await asyncio.Event().wait()

        async def cancel(self, operation_id):
            self.cancelled = operation_id

        async def close(self):
            pass

    async def run():
        runtime = Runtime()
        runtime.rag = FakeRag()
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "查询知识库", 1)
            task.update(status="running", run_id="run-one")
            store.put(session, f"task:{task['id']}", task)
            execution = {"type": "tool_execution", "id": "current", "task_id": task["id"],
                         "context_id": task["context_id"], "run_id": "run-one", "name": "rag_search",
                         "input": {"query": "问题"}, "status": "pending", "call_id": "call"}
            store.put(session, "execution:current", execution)
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run-one"
        pending = asyncio.create_task(worker.execute(execution))
        await runtime.rag.started.wait()
        worker.stop.set()
        assert "结果可能未知" in await asyncio.wait_for(pending, 2)
        assert runtime.rag.cancelled == "current"
        await runtime.close()
    asyncio.run(run())
