import asyncio
import json
import sqlite3
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import db
from app.agent.rag_client import RagClient
from app.agent.runtime import Runtime
from app.agent.worker import Worker
from app.config import settings
from app.gateway import Gateway
from app.models import User
from app.sessions import chat_store as store
from app.user_context import bind_user


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    yield tmp_path
    db.dispose_engine()


async def allowed():
    return None


def test_gateway_shares_rag_between_users(isolated, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver

    monkeypatch.setattr(settings, "model_id", "")
    with db.new_session() as session:
        session.add_all([User(id=key * 32, username=key, password=key) for key in ("a", "b")])
        session.commit()

    async def run():
        gateway = Gateway(InMemorySaver())
        try:
            first = await gateway.runtime("a" * 32)
            second = await gateway.runtime("b" * 32)
            assert first.rag is second.rag is gateway.rag
            with bind_user("a" * 32):
                await first.close()
            assert second.rag is gateway.rag
        finally:
            await gateway.close()

    asyncio.run(run())


def test_real_mcp_replaces_existing_document_without_changing_rag(isolated, monkeypatch):
    import mcp.client.stdio

    data_dir = isolated / "rag-data"
    data_dir.mkdir()
    document_id, old_version = "1" * 32, "2" * 32
    with sqlite3.connect(data_dir / "catalog.sqlite3") as connection:
        connection.executescript("""
            CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, filename TEXT NOT NULL,
                                    version TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL);
            CREATE TABLE staged (version TEXT PRIMARY KEY, path TEXT NOT NULL);
        """)
        connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                           (document_id, "old/folder/Report.txt", "Report.txt", old_version, 3, "ready"))
    directory = data_dir / "documents" / document_id
    directory.mkdir(parents=True)
    (directory / f"{old_version}.json").write_text('{}', encoding="utf-8")
    folder = settings.workspaces_dir / ("a" * 32)
    folder.mkdir(parents=True)
    (folder / "report.txt").write_text("新知识库内容", encoding="utf-8")
    monkeypatch.setenv("RAG_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MILVUS_CLOUD_URI", "")
    monkeypatch.setenv("MILVUS_CLOUD_TOKEN", "")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "model_id", "")
    original = mcp.client.stdio.stdio_client
    fixture = Path(__file__).parent / "fixtures" / "fake_rag_mcp.py"
    fail = [True]

    @asynccontextmanager
    async def fake_stdio(params):
        params.args = [str(fixture)]
        params.env["LANGFUSE_ENABLED"] = "false"
        params.env["FAKE_RAG_FAIL_ADD"] = "1" if fail[0] else "0"
        async with original(params) as streams:
            yield streams

    monkeypatch.setattr(mcp.client.stdio, "stdio_client", fake_stdio)

    async def run():
        rag = RagClient()
        runtime = SimpleNamespace(file_locks={})
        try:
            result = await asyncio.wait_for(rag.perform(runtime, "rag_ingest", {"path": "report.txt"},
                                                         "failure", allowed, asyncio.Event()), 30)
            assert "结果可能未知" in result
            with sqlite3.connect(data_dir / "catalog.sqlite3") as connection:
                assert connection.execute("SELECT version FROM documents").fetchone()[0] == old_version
            assert (directory / f"{old_version}.json").exists()
        finally:
            await rag.close()
        fail[0] = False
        rag = RagClient()
        try:
            result = await asyncio.wait_for(rag.perform(runtime, "rag_ingest", {"path": "report.txt"},
                                                         "operation", allowed, asyncio.Event()), 30)
            assert json.loads(result)["document_id"] == document_id
            with sqlite3.connect(data_dir / "catalog.sqlite3") as connection:
                identity, version = connection.execute("SELECT id, version FROM documents").fetchone()
            assert identity == document_id and version != old_version
            assert (directory / f"{version}.json").exists()
            assert not (rag.workspace / "old/folder/Report.txt").exists()
            assert (folder / "report.txt").read_text(encoding="utf-8") == "新知识库内容"
            answer = await rag.perform(runtime, "rag_search", {"query": "问题"}, "search", allowed, asyncio.Event())
            assert json.loads(answer)["answer"] == "知识库回答：问题"
            removed = await rag.perform(runtime, "rag_remove", {"document_id": document_id},
                                        "remove", allowed, asyncio.Event())
            assert json.loads(removed)["document_id"] == document_id
            with sqlite3.connect(data_dir / "catalog.sqlite3") as connection:
                assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
            assert (folder / "report.txt").exists()
        finally:
            await rag.close()

    asyncio.run(run())


def test_same_name_updates_legacy_document_and_serializes_users(isolated, monkeypatch):
    async def run():
        rag = RagClient()
        document = {"id": "legacy-id", "path": "old/folder/Report.PDF", "filename": "Report.PDF", "status": "ready"}
        calls, active, peak = [], 0, 0

        async def fake_call(name, args, *, sent=None):
            nonlocal active, peak
            if name == "rag_list":
                return json.dumps({"status": "success", "documents": [document]})
            assert name == "rag_ingest"
            sent.set()
            active += 1
            peak = max(peak, active)
            try:
                calls.append((args["path"], (rag.workspace / args["path"]).read_bytes()))
                await asyncio.sleep(0.02)
                return json.dumps({"status": "success", "document_id": document["id"]})
            finally:
                active -= 1

        monkeypatch.setattr(rag, "call", fake_call)
        for user, content in (("a" * 32, b"first"), ("b" * 32, b"second")):
            folder = settings.workspaces_dir / user
            folder.mkdir(parents=True)
            (folder / "report.pdf").write_bytes(content)

        async def ingest(user):
            with bind_user(user):
                runtime = SimpleNamespace(file_locks={})
                return await rag.perform(runtime, "rag_ingest", {"path": "report.pdf"}, user, allowed, asyncio.Event())

        first, second = await asyncio.gather(ingest("a" * 32), ingest("b" * 32))
        assert json.loads(first)["document_id"] == json.loads(second)["document_id"] == "legacy-id"
        assert calls == [("old/folder/Report.PDF", b"first"), ("old/folder/Report.PDF", b"second")]
        assert peak == 1 and not (rag.workspace / document["path"]).exists()
        assert (settings.workspaces_dir / ("a" * 32) / "report.pdf").read_bytes() == b"first"
        assert (settings.workspaces_dir / ("b" * 32) / "report.pdf").read_bytes() == b"second"

    asyncio.run(run())


def test_ingest_rejects_other_users_file_and_unsafe_catalog_path(isolated, monkeypatch):
    async def run():
        rag = RagClient()
        user_a, user_b = "a" * 32, "b" * 32
        for user in (user_a, user_b):
            (settings.workspaces_dir / user).mkdir(parents=True)
        (settings.workspaces_dir / user_b / "secret.txt").write_text("private", encoding="utf-8")
        (settings.workspaces_dir / user_a / "safe.txt").write_text("safe", encoding="utf-8")
        calls = []

        async def fake_call(name, args, *, sent=None):
            calls.append(name)
            return json.dumps({"status": "success", "documents": [
                {"id": "old", "path": "../../safe.txt", "filename": "safe.txt", "status": "ready"}]})

        monkeypatch.setattr(rag, "call", fake_call)
        runtime = SimpleNamespace(file_locks={})
        with bind_user(user_a):
            outside = await rag.perform(runtime, "rag_ingest", {"path": f"../{user_b}/secret.txt"},
                                        "outside", allowed, asyncio.Event())
            unsafe = await rag.perform(runtime, "rag_ingest", {"path": "safe.txt"},
                                       "unsafe", allowed, asyncio.Event())
        assert outside.startswith("Error:") and unsafe.startswith("Error:")
        assert calls == ["rag_list"]
        assert not (isolated / "safe.txt").exists()

    asyncio.run(run())


def test_unknown_write_keeps_staged_file_and_does_not_overwrite(isolated, monkeypatch):
    async def run():
        rag = RagClient()
        folder = settings.workspaces_dir / ("a" * 32)
        folder.mkdir(parents=True)
        (folder / "note.txt").write_text("new", encoding="utf-8")
        calls = []

        async def fake_call(name, args, *, sent=None):
            calls.append(name)
            if name == "rag_list":
                return json.dumps({"status": "success", "documents": [
                    {"id": "old", "path": "old/note.txt", "filename": "note.txt", "status": "ready"}]})
            sent.set()
            return "Error: 操作结果可能未知，请先检查"

        monkeypatch.setattr(rag, "call", fake_call)
        runtime = SimpleNamespace(file_locks={})
        first = await rag.perform(runtime, "rag_ingest", {"path": "note.txt"}, "one", allowed, asyncio.Event())
        second = await rag.perform(runtime, "rag_ingest", {"path": "note.txt"}, "two", allowed, asyncio.Event())
        assert "结果可能未知" in first and "临时文件" in second
        assert calls.count("rag_ingest") == 1
        assert (rag.workspace / "old/note.txt").read_text(encoding="utf-8") == "new"

    asyncio.run(run())


def test_inconsistent_catalog_does_not_report_replacement_success(isolated, monkeypatch):
    async def run():
        rag = RagClient()
        folder = settings.workspaces_dir / ("a" * 32)
        folder.mkdir(parents=True)
        (folder / "note.txt").write_text("new", encoding="utf-8")
        document = {"id": "old", "path": "old/note.txt", "filename": "note.txt", "status": "deleting"}
        calls = []

        async def fake_call(name, args, *, sent=None):
            calls.append(name)
            if name == "rag_list":
                return json.dumps({"status": "success", "documents": [document]})
            sent.set()
            return json.dumps({"status": "success", "document_id": "unexpected"})

        monkeypatch.setattr(rag, "call", fake_call)
        runtime = SimpleNamespace(file_locks={})
        first = await rag.perform(runtime, "rag_ingest", {"path": "note.txt"}, "one", allowed, asyncio.Event())
        assert "状态异常" in first and calls == ["rag_list"]
        document["status"] = "ready"
        second = await rag.perform(runtime, "rag_ingest", {"path": "note.txt"}, "two", allowed, asyncio.Event())
        assert "结果未知" in second and (rag.workspace / "old/note.txt").exists()

    asyncio.run(run())


def test_remove_without_matching_success_is_unknown(isolated, monkeypatch):
    async def run():
        rag = RagClient()

        async def fake_call(name, args, *, sent=None):
            sent.set()
            return '{"status":"success","document_id":"other"}'

        monkeypatch.setattr(rag, "call", fake_call)
        result = await rag.perform(SimpleNamespace(), "rag_remove", {"document_id": "expected"},
                                   "operation", allowed, asyncio.Event())
        assert "结果未知" in result

    asyncio.run(run())


def setup_worker(runtime, user_text: str, name: str = "rag_ingest", args=None):
    args = args or {"path": "note.txt"}
    with db.new_session() as session:
        store.initialize(session, settings.data_dir)
        task = runtime.create_task(session, user_text, 1)
        task.update(status="running", run_id="run")
        store.put(session, f"task:{task['id']}", task)
        store.put(session, "input:one", {"type": "user_event", "event_type": "message", "user_seq": 1,
                                         "status": "processed", "user_text": user_text})
        execution = {"type": "tool_execution", "id": "one", "call_id": "call", "task_id": task["id"],
                     "context_id": task["context_id"], "run_id": "run", "name": name, "input": args,
                     "status": "pending", "user_seq": 1}
        store.put(session, "execution:one", execution)
        session.commit()
    worker = Worker(runtime, task["id"])
    worker.context_id, worker.run_id = task["context_id"], "run"
    return worker, execution


def test_worker_requires_explicit_instruction_for_shared_write(isolated):
    class FakeRag:
        async def perform(self, *_args):
            raise AssertionError("未授权的写操作不应执行")

    async def run():
        runtime = Runtime(client=object(), rag=FakeRag())
        (settings.workspaces_dir / ("a" * 32)).mkdir(parents=True)
        worker, execution = setup_worker(runtime, "查询知识库")
        result = await worker.execute(execution)
        assert "明确指定" in result
        await runtime.close()

    asyncio.run(run())


def test_unknown_write_is_not_replayed_without_new_user_request(isolated):
    class FakeRag:
        def __init__(self):
            self.calls = 0

        async def perform(self, runtime, name, args, operation_id, before_send, sent):
            self.calls += 1
            assert await before_send() is None
            sent.set()
            return '{"status":"success"}'

    async def run():
        rag = FakeRag()
        runtime = Runtime(client=object(), rag=rag)
        (settings.workspaces_dir / ("a" * 32)).mkdir(parents=True)
        worker, execution = setup_worker(runtime, "将 note.txt 加入知识库")
        with db.new_session() as session:
            store.put(session, "execution:old", {**execution, "id": "old", "status": "completed",
                                                 "result_unknown": True})
            session.commit()
        result = await worker.execute(execution)
        assert "等待用户明确要求重试" in result and rag.calls == 0
        with db.new_session() as session:
            task = store.get(session, f"task:{worker.task_id}")
            task["latest_user_seq"] = 2
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "input:two", {"type": "user_event", "event_type": "message", "user_seq": 2,
                                             "status": "processed", "user_text": "请重新将 note.txt 加入知识库"})
            store.put(session, "bus:update", {"type": "agent_message", "id": "update", "task_id": task["id"],
                      "source": "main", "context_id": task["context_id"], "user_seq": 2, "delivery": "processed"})
            session.commit()
        assert '"success"' in await worker.execute(execution) and rag.calls == 1
        await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("message_type", ["user_event", "agent_message"])
def test_new_user_message_stops_write_during_staging(isolated, monkeypatch, message_type):
    async def run():
        rag = RagClient()
        runtime = Runtime(client=object(), rag=rag)
        folder = settings.workspaces_dir / ("a" * 32)
        folder.mkdir(parents=True)
        (folder / "note.txt").write_text("内容", encoding="utf-8")
        worker, execution = setup_worker(runtime, "将 note.txt 加入知识库")
        started, release = threading.Event(), threading.Event()
        calls = []
        original_copy = rag._copy

        def slow_copy(source, target, created):
            original_copy(source, target, created)
            started.set()
            release.wait(5)

        async def fake_call(name, args, *, sent=None):
            calls.append(name)
            return json.dumps({"status": "success", "documents": []})

        monkeypatch.setattr(rag, "_copy", slow_copy)
        monkeypatch.setattr(rag, "call", fake_call)
        pending = asyncio.create_task(worker.execute(execution))
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(0.02)
            if message_type == "user_event":
                message = {"type": "user_event", "event_type": "message", "user_seq": 2,
                           "status": "pending", "user_text": "不要入库"}
            else:
                message = {"type": "agent_message", "id": "new", "task_id": worker.task_id,
                           "source": "main", "context_id": worker.context_id, "delivery": "pending"}
            await runtime.change(lambda session: store.put(session, "input:new", message))
            release.set()
            assert "新的用户" in await asyncio.wait_for(pending, 3)
            assert calls == ["rag_list"] and not (rag.workspace / "new/note.txt").exists()
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("writing", [False, True])
def test_stop_waits_for_sent_write_and_cancels_unsent_call(isolated, writing):
    class FakeRag:
        def __init__(self):
            self.ready = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = False

        async def perform(self, runtime, name, args, operation_id, before_send, sent):
            if writing:
                assert await before_send() is None
                sent.set()
            self.ready.set()
            try:
                await self.release.wait()
                return '{"status":"success"}'
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def cancel(self, operation_id):
            raise AssertionError("不得取消已发送的写操作")

    async def run():
        rag = FakeRag()
        runtime = Runtime(client=object(), rag=rag)
        (settings.workspaces_dir / ("a" * 32)).mkdir(parents=True)
        worker, execution = setup_worker(runtime, "将 note.txt 加入知识库")
        pending = asyncio.create_task(worker.execute(execution))
        await asyncio.wait_for(rag.ready.wait(), 2)
        worker.stop.set()
        await asyncio.sleep(0.05)
        if writing:
            assert not pending.done() and not rag.cancelled
            rag.release.set()
            assert '"success"' in await asyncio.wait_for(pending, 2)
            with db.new_session() as session:
                assert any("知识库操作结果" in item["content"] for item in store.records(session, "agent_message"))
        else:
            assert "停止" in await asyncio.wait_for(pending, 2) and rag.cancelled
        await runtime.close()

    asyncio.run(run())


def test_shutdown_waits_for_sent_write_and_records_result(isolated):
    class FakeRag:
        def __init__(self):
            self.started, self.release = asyncio.Event(), asyncio.Event()

        async def perform(self, runtime, name, args, operation_id, before_send, sent):
            assert await before_send() is None
            sent.set()
            self.started.set()
            await self.release.wait()
            return '{"status":"success"}'

    async def run():
        rag = FakeRag()
        runtime = Runtime(client=object(), rag=rag)
        (settings.workspaces_dir / ("a" * 32)).mkdir(parents=True)
        worker, execution = setup_worker(runtime, "将 note.txt 加入知识库")
        pending = asyncio.create_task(worker.execute(execution))
        runtime.workers[worker.task_id] = pending
        await asyncio.wait_for(rag.started.wait(), 2)
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        rag.release.set()
        await asyncio.wait_for(closing, 3)
        with db.new_session() as session:
            saved = store.get(session, "execution:one")
            assert saved["status"] == "completed" and '"success"' in saved["output"]

    asyncio.run(run())


def test_shutdown_records_unsent_write_as_stopped(isolated):
    class FakeRag:
        def __init__(self):
            self.started = asyncio.Event()

        async def perform(self, runtime, name, args, operation_id, before_send, sent):
            assert await before_send() is None
            self.started.set()
            await asyncio.Event().wait()

    async def run():
        rag = FakeRag()
        runtime = Runtime(client=object(), rag=rag)
        (settings.workspaces_dir / ("a" * 32)).mkdir(parents=True)
        worker, execution = setup_worker(runtime, "将 note.txt 加入知识库")
        pending = asyncio.create_task(worker.execute(execution))
        runtime.workers[worker.task_id] = pending
        await asyncio.wait_for(rag.started.wait(), 2)
        await asyncio.wait_for(runtime.close(), 3)
        with db.new_session() as session:
            saved = store.get(session, "execution:one")
            assert saved["status"] == "completed" and "工具未执行" in saved["output"]
            assert not saved.get("result_unknown")

    asyncio.run(run())


def test_rag_answer_is_bounded_delivered_and_task_cleaned(isolated):
    async def run():
        runtime = Runtime(client=object())
        (settings.workspaces_dir / ("a" * 32)).mkdir(parents=True)
        with db.new_session() as session:
            store.initialize(session, settings.data_dir)
            task = runtime.create_task(session, "知识库问答", 1, knowledge_question=True, key="query")
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:rag", {"type": "tool_execution", "id": "rag", "task_id": task["id"],
                      "context_id": task["context_id"], "run_id": "run", "name": "rag_search", "status": "completed",
                      "output": json.dumps({"status": "success", "answer": "回答" * 10000}, ensure_ascii=False)})
            store.put(session, "execution:list", {"type": "tool_execution", "id": "list", "task_id": task["id"],
                      "context_id": task["context_id"], "run_id": "run", "name": "rag_list", "status": "completed"})
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        await worker.finish("result", "完成")
        with db.new_session() as session:
            report = store.records(session, "agent_message", context_id="main")[0]
            assert len(report["content"].encode("utf-8")) <= 16 * 1024 and "已截断" in report["content"]
        await runtime._report(report)
        assert "已截断" in (await runtime.snapshot())["messages"][-1]["text"]
        await runtime._cleanup_finished()
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}") is None
            assert store.get(session, "task-key:query")["processed"]
        await runtime.close()

    asyncio.run(run())


def test_main_worker_and_rag_answer_end_to_end(isolated):
    class Model:
        model = "fake"

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if tool_choice and tool_choice["name"] == "route_input":
                request = json.loads(messages[-1]["content"])
                yield {"type": "tool_use", "id": "route", "name": "route_input", "input": {
                    "message_id": request["message_id"], "user_seq": request["user_seq"],
                    "intents": [{"type": "create", "instruction": "查询知识库", "knowledge_question": True}],
                    "needs_clarification": False}}
            elif system.startswith("你是后台工作助手"):
                calls = [part for message in messages if isinstance(message["content"], list)
                         for part in message["content"] if part.get("name") == "rag_search"]
                if not calls:
                    yield {"type": "tool_use", "id": "search", "name": "rag_search", "input": {"query": "问题"}}
                else:
                    yield {"type": "text_delta", "text": "已完成查询"}
            else:
                yield {"type": "text_delta", "text": "已安排查询"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    class FakeRag:
        async def perform(self, runtime, name, args, operation_id, before_send, sent):
            assert name == "rag_search" and await before_send() is None
            sent.set()
            return json.dumps({"status": "success", "answer": "知识库原答案", "sources": []}, ensure_ascii=False)

    with db.new_session() as session:
        session.add(User(id="a" * 32, username="a", password="a"))
        session.commit()

    async def run():
        runtime = Runtime(client=Model(), rag=FakeRag())
        await runtime.start()
        try:
            assistant = (await runtime.snapshot())["assistant_id"]
            await runtime.accept("message", "查询知识库里的问题", "build", assistant_id=assistant)
            async with asyncio.timeout(12):
                while True:
                    with db.new_session() as session:
                        marker = store.get(session, "task-key:route:message:0")
                        messages = store.visible(session)
                    if marker and marker.get("processed"):
                        break
                    await asyncio.sleep(0.05)
            assert any(message["text"] == "知识库原答案" for message in messages)
        finally:
            await runtime.close()

    asyncio.run(run())


def test_reset_waits_for_own_rag_write_without_stopping_other_user(isolated, monkeypatch):
    monkeypatch.setattr(settings, "bash_timeout", -4)
    class FakeRag:
        def __init__(self):
            self.started = {user: asyncio.Event() for user in ("a" * 32, "b" * 32)}
            self.release = {user: asyncio.Event() for user in self.started}
            self.cancelled = []

        async def perform(self, runtime, name, args, operation_id, before_send, sent):
            assert await before_send() is None
            sent.set()
            self.started[runtime.user_id].set()
            await self.release[runtime.user_id].wait()
            return json.dumps({"status": "success", "answer": "回答"}, ensure_ascii=False)

        async def cancel(self, operation_id):
            self.cancelled.append(operation_id)

    async def run():
        rag = FakeRag()
        runtimes, calls = {}, {}
        try:
            for user in ("a" * 32, "b" * 32):
                with bind_user(user):
                    runtime = Runtime(client=object(), rag=rag)
                    runtime.client = None
                    await runtime.start()
                    name, args = (("rag_ingest", {"path": "note.txt"}) if user.startswith("a")
                                  else ("rag_search", {"query": "问题"}))
                    worker, execution = setup_worker(runtime, "将 note.txt 加入知识库" if user.startswith("a") else "查询知识库",
                                                     name=name, args=args)

                    async def execute(current=runtime, current_worker=worker, item=execution):
                        result = await current_worker.execute(item)
                        def save(session):
                            row = store.get(session, f"execution:{item['id']}")
                            row.update(status="completed", output=result)
                            store.put(session, f"execution:{item['id']}", row)
                        await current.change(save)
                        return result

                    runtimes[user] = runtime
                    calls[user] = asyncio.create_task(execute())
                    runtime.workers[worker.task_id] = calls[user]
                    await asyncio.wait_for(rag.started[user].wait(), 3)
            with bind_user("a" * 32):
                runtime = runtimes["a" * 32]
                old = (await runtime.snapshot())["assistant_id"]
                await runtime.reset(old, "reset-a")
                await asyncio.sleep(1.3)
                assert not calls["a" * 32].done() and not runtime.reset_task.done()
                assert (await runtime.snapshot())["reset"]["status"] == "stopping"
                assert not calls["b" * 32].done() and not rag.cancelled
                rag.release["a" * 32].set()
                await asyncio.wait_for(runtime.reset_task, 5)
                assert (await runtime.snapshot())["assistant_id"] != old
            assert not calls["b" * 32].done() and not rag.cancelled
            with bind_user("b" * 32):
                rag.release["b" * 32].set()
                assert '"success"' in await asyncio.wait_for(calls["b" * 32], 3)
        finally:
            for event in rag.release.values():
                event.set()
            for user, runtime in runtimes.items():
                with bind_user(user):
                    await runtime.close()

    asyncio.run(run())
