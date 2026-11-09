import asyncio
import json
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from sqlalchemy import make_url, select
from sqlalchemy.exc import OperationalError

from app import db
from app.agent import agent_bus, shared_state
from app.agent.checkpoint import _open_manager, _run_io, open_saver
from app.agent.runtime import Runtime, _task_context
from app.agent.worker import Worker
from app.config import load_settings, settings
from app.models import Message, Part, Session, User
from app.sessions import chat_store as store, execution_state
from app.user_context import bind_user, scoped


@pytest.fixture(params=["sqlite", "postgresql"])
def database(request, tmp_path, monkeypatch):
    database_url = os.environ.get("OPENWORKBOT_TEST_DATABASE_URL")
    if request.param == "postgresql" and not database_url:
        pytest.skip("需要独立的 OPENWORKBOT_TEST_DATABASE_URL")
    admin = schema = None
    if request.param == "postgresql":
        admin = psycopg.connect(database_url, autocommit=True)
        schema = "owb_test_" + uuid.uuid4().hex
        admin.execute(sql.SQL("CREATE SCHEMA {} ").format(sql.Identifier(schema)))
        url = make_url(database_url).update_query_dict({"options": f"-csearch_path={schema}"})
        monkeypatch.setattr(settings, "database_url", url.render_as_string(hide_password=False))
    monkeypatch.setattr(settings, "database_backend", request.param)
    db.dispose_engine()
    try:
        db.init_db(settings.data_dir)
        with db.new_session() as session:
            session.add_all([User(id=char * 32, username=f"test-{char}", password="test") for char in "ab"])
            session.commit()
        yield request.param
    finally:
        db.dispose_engine()
        if admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
            admin.close()


def test_business_storage_order_and_user_isolation(database):
    with db.new_session() as session:
        store.initialize(session, settings.data_dir)
        for _ in range(10):
            store.bump(session)
        store.add_visible(session, "user", "第一条", message_id="same")
        store.put(session, "approval:one", {"type": "approval", "order": store.next_display_order(session)})
        store.add_visible(session, "assistant", "第二条")
        store.put(session, "input:same", {"type": "user_event", "context_id": "main", "text": "第一条"})
        store.stage_context(session, "main", "user", "第一条", key="same")
        store.clear_pending(session, "main", ["same"])
        session.commit()
        assert [item["order"] for item in store.visible(session)] == [1, 3]
        assert store.get(session, "approval:one")["order"] == 2
        assert store.records(session, "user_event", context_id="main")[0]["text"] == "第一条"
        assert store.records(session, "context_pending") == []
        row = session.get(Part, scoped("input:same"))
        assert row.type == "user_event" and row.context_id == "main" and "type" not in row.data
        assert session.scalar(select(Message).where(Message.id == scoped("same"))).text == "第一条"
        assert store.visible_page(session, 1)[0][0]["text"] == "第二条"
        store.add_visible(session, "user", "回滚")
        session.rollback()
        store.add_visible(session, "user", "提交")
        session.commit()
        assert store.visible(session)[-1]["order"] == 4

    execution_state.save_execution_state(settings.data_dir, "edit")
    execution_state.save_execution_state(settings.data_dir, "build")
    assert execution_state.load_execution_state(settings.data_dir)["mode"] == "build"

    with bind_user("b" * 32), db.new_session() as session:
        store.initialize(session, settings.data_dir)
        store.add_visible(session, "user", "另一用户", message_id="same")
        session.commit()
        assert [item["text"] for item in store.visible(session)] == ["另一用户"]
        assert store.get(session, "input:same") is None


def test_concurrent_message_sequences_are_unique(database):
    with db.new_session() as session:
        store.initialize(session, settings.data_dir)

    def send(index):
        with bind_user("a" * 32), db.new_session() as session:
            store.add_visible(session, "user", str(index))
            session.commit()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(send, range(12)))
    with db.new_session() as session:
        rows = list(session.scalars(select(Message).order_by(Message.sequence)))
        assert [row.sequence for row in rows] == list(range(1, 13))
        assert [row.display_order for row in rows] == list(range(1, 13))


def test_nested_json_updates_survive_with_loaded_rows(database):
    with db.new_session() as session:
        store.initialize(session, settings.data_dir)
        meta = store.get(session, "meta")
        meta["reset"] = {"id": "test", "status": "stopping"}
        store.put(session, "meta", meta)
        store.put(session, "nested", {"type": "example", "details": {"status": "old"}})
        session.commit()
        current_session = session.get(Session, scoped("main"))
        current_part = session.get(Part, scoped("nested"))
        meta = store.get(session, "meta")
        record = store.get(session, "nested")
        meta["reset"]["status"] = "deleting"
        record["details"]["status"] = "new"
        store.put(session, "meta", meta)
        store.put(session, "nested", record)
        session.commit()
        session.expire_all()
        assert current_session.reset_state["status"] == "deleting"
        assert current_part.data["details"]["status"] == "new"


def test_approval_and_task_queries(database):
    async def run():
        with db.new_session() as session:
            store.initialize(session, settings.data_dir)
            assistant_id = store.get(session, "meta")["assistant_id"]
        runtime = Runtime(client=Model(), memory=Memory())
        await runtime.accept("first", "第一条", None, assistant_id=assistant_id)
        task = await runtime.change(lambda session: runtime.create_task(session, "测试任务", 1))
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run-one"
        approval = asyncio.create_task(worker.approve(
            {"id": "call", "call_id": "call", "name": "bash", "input": {"command": "echo test"}}, "需要审批"))
        try:
            await wait_for(lambda: bool(runtime.approval_events))
            request_id = next(iter(runtime.approval_events))
            await runtime.accept("second", "第二条", None, assistant_id=assistant_id)
            with db.new_session() as session:
                assert [item["order"] for item in store.visible(session)] == [1, 3]
                assert store.get(session, f"approval:{request_id}")["order"] == 2
                result = runtime._main_tool(session, {"id": "search", "name": "task_search",
                                                     "input": {"keyword": "测试"}}, None)
                assert json.loads(result)["tasks"][0]["id"] == task["id"]
                assert _task_context(session)[0][0]["id"] == task["id"]
                agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                               content="结果", kind="result", message_id="report")
                result = runtime._main_tool(session, {"id": "inspect", "name": "task_inspect",
                                                     "input": {"task_id": task["id"]}}, None)
                assert json.loads(result)["reports"][0]["content"] == "结果"
                shared_state.replace(session, "main", None, "test-run", [{"content": "共享", "status": "in_progress"}])
                assert shared_state.snapshot(session)[0]["todo"] == "共享"
                session.commit()
            await runtime.decide(request_id, "deny", assistant_id=assistant_id)
            assert not (await approval)["approved"]
            with db.new_session() as session:
                events = store.records(session, "user_event")
                assert [(event["event_type"], event["user_seq"]) for event in events] == [
                    ("message", 1), ("message", 2), ("approval", 3)]
        finally:
            if not approval.done():
                approval.cancel()
                await asyncio.gather(approval, return_exceptions=True)
            await runtime.close()
    asyncio.run(run())


class Model:
    def __init__(self):
        self.requests = []

    async def stream(self, *, system, messages, tools, **kwargs):
        self.requests.append(messages)
        yield {"type": "text_delta", "text": "收到。"}
        yield {"type": "stop_reason", "reason": "end_turn"}


class Memory:
    async def snapshot(self, query):
        return ""

    async def clear(self):
        pass

    def enqueue(self, *args, **kwargs):
        pass

    def discard_pending(self, user_id):
        pass


async def wait_for(check):
    async with asyncio.timeout(20):
        while not check():
            await asyncio.sleep(0.02)


async def state(runtime):
    async with runtime.graph.write_lock:
        thread_id = runtime.graph.thread_id("main:loop", parent_context="main")
    return thread_id, (await runtime.agent_loop.graph.aget_state(
        {"configurable": {"thread_id": thread_id}})).values


def test_checkpoint_restart_and_reset(database):
    async def run():
        model = Model()
        old_thread = None
        for index in range(2):
            async with open_saver(settings.data_dir) as saver:
                runtime = Runtime(client=model, checkpointer=saver, memory=Memory())
                await runtime.start()
                try:
                    snapshot = await runtime.snapshot()
                    await runtime.accept(f"input-{index}", f"问题{index}", None,
                                         assistant_id=snapshot["assistant_id"])

                    def completed():
                        with db.new_session() as session:
                            reply = store.get(session, f"reply:input-{index}")
                            return bool(reply and reply["status"] == "completed" and runtime.main_call is None)

                    await wait_for(completed)
                    old_thread, values = await state(runtime)
                    assert [message.content for message in values["messages"] if message.type == "human"] == [
                        f"问题{n}" for n in range(index + 1)]
                    worker_messages = [{"id": "same", "content": "工作一"}, {"id": "same", "content": "工作二"}]
                    if index == 0:
                        for number, message in enumerate(worker_messages):
                            await runtime.graph.invoke(f"worker-{number}", incoming=message)
                    if index == 1:
                        assert any(item.get("content") == "问题0" for item in model.requests[-1])
                        history = [item async for item in saver.alist({"configurable": {"thread_id": old_thread}})]
                        assert len(history) > 1
                        worker_threads = []
                        for number, message in enumerate(worker_messages):
                            assert await runtime.graph.has_delivery(f"worker-{number}", message)
                            assert not await runtime.graph.has_delivery(f"worker-{number}", worker_messages[1 - number])
                            worker_threads.append(runtime.graph.thread_id(f"worker-{number}"))
                        before = await runtime.snapshot()
                        await runtime.reset(before["assistant_id"], "reset-test")

                        def reset_done():
                            with db.new_session() as session:
                                return store.get(session, "meta")["assistant_id"] != before["assistant_id"]

                        await wait_for(reset_done)
                        after = await runtime.snapshot()
                        assert after["revision"] > before["revision"] and after["messages"] == []
                        assert await saver.aget_tuple({"configurable": {"thread_id": old_thread}}) is None
                        for thread_id in worker_threads:
                            assert await saver.aget_tuple({"configurable": {"thread_id": thread_id}}) is None
                        with db.new_session() as session:
                            row = session.get(Session, scoped("main"))
                            assert row.last_reset["status"] == "completed" and row.reset_state is None
                            assert row.user_seq == 2
                        accepted = await runtime.accept("after-reset", "新会话", None,
                                                        assistant_id=after["assistant_id"])
                        assert accepted["user_seq"] == 3
                        current = await runtime.snapshot()
                        assert current["messages"][0]["order"] > max(item["order"] for item in before["messages"])
                finally:
                    await runtime.close()
            db.dispose_engine()
            db.init_db(settings.data_dir)
    asyncio.run(run())


@pytest.mark.parametrize("journal", [False, True])
def test_waiting_board_recovers_completed_todo_before_checkpoint(database, journal):
    async def run():
        async with open_saver(settings.data_dir) as saver:
            runtime = Runtime(client=Model(), checkpointer=saver, memory=Memory())
            runtime.client = None
            await runtime.start()
            try:
                def create(session):
                    task = runtime.create_task(session, "等待中的工作", 1)
                    task.update(status="waiting_agent", run_id="run-one")
                    store.put(session, f"task:{task['id']}", task)
                    return task
                task = await runtime.change(create)
                thread = runtime.graph.thread_id(f"{task['context_id']}:loop", parent_context=task["context_id"])
                await runtime.agent_loop.graph.aupdate_state({"configurable": {"thread_id": thread}}, {
                    "todos": [{"content": "旧检查点", "status": "in_progress"}]})
                items = [{"content": "等待期间的工作", "status": "in_progress"}]
                def update(session):
                    store.put(session, "execution:todo", {"type": "tool_execution", "id": "todo",
                              "name": "write_todos", "status": "completed", "output": "已更新",
                              "input": {"todos": items}, "task_id": task["id"], "context_id": task["context_id"],
                              "run_id": "run-one"})
                    if journal:
                        shared_state.replace(session, task["context_id"], task["id"], "run-one", items,
                                             thread_id=thread, execution_id="todo")
                    else:
                        store.put(session, "group:todo", {"type": "tool_group", "id": "todo",
                                  "context_id": task["context_id"], "calls": ["todo"], "status": "pending"})
                await runtime.change(update)
            finally:
                await runtime.close()
            resumed = Runtime(client=Model(), checkpointer=saver, memory=Memory())
            resumed.client = None
            await resumed.start()
            try:
                with db.new_session() as session:
                    assert [(item["todo"], item["status"]) for item in shared_state.snapshot(session)] == [
                        ("等待期间的工作", "waiting_agent")]
                    assert not store.records(session, "todo_sync")
            finally:
                await resumed.close()
    asyncio.run(run())


def test_main_todo_checkpoint_recovers_interrupted_tool_group(database):
    async def run():
        items = [{"content": "主助手待恢复工作", "status": "in_progress"}]
        async with open_saver(settings.data_dir) as saver:
            runtime = Runtime(client=Model(), checkpointer=saver, memory=Memory())
            runtime.client = None
            await runtime.start()
            try:
                thread = runtime.graph.thread_id("main:loop", parent_context="main")
                await runtime.agent_loop.graph.aupdate_state({"configurable": {"thread_id": thread}}, {"todos": []})
                await runtime.change(lambda session: store.put(session, "group:main:old", {
                    "type": "main_tool_group", "id": "main:old", "context_id": "main", "status": "pending",
                    "calls": [{"id": "todo", "name": "write_todos", "input": {"todos": items}}],
                    "outputs": {"todo": "已更新 TODO。"}}))
            finally:
                await runtime.close()
            resumed = Runtime(client=Model(), checkpointer=saver, memory=Memory())
            resumed.client = None
            await resumed.start()
            try:
                state = await resumed.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
                assert state.values["todos"] == items
            finally:
                await resumed.close()
    asyncio.run(run())


def test_task_cleanup_waits_for_delivery_before_deleting_checkpoint(database):
    async def run():
        async with open_saver(settings.data_dir) as saver:
            runtime = Runtime(client=Model(), checkpointer=saver, memory=Memory())
            with db.new_session() as session:
                store.initialize(session, settings.data_dir)
                store.add_visible(session, "user", "保留聊天")
                task = runtime.create_task(session, "待清理任务", 1, key="cleanup-test")
                task.update(status="completed", result_sent=True)
                store.put(session, f"task:{task['id']}", task)
                agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                               content="待确认结果", message_id="pending")
                session.commit()
            await runtime.graph.invoke(task["context_id"], incoming={"id": "input", "content": "任务"})
            thread_id = runtime.graph.thread_id(task["context_id"])
            config = {"configurable": {"thread_id": thread_id}}
            await runtime._cleanup_task(task["id"], {task["context_id"]})
            assert await saver.aget_tuple(config) is not None
            with db.new_session() as session:
                message = store.get(session, "bus:pending")
                message["delivery"] = "processed"
                store.put(session, "bus:pending", message)
                session.commit()
            await runtime._cleanup_task(task["id"], {task["context_id"]})
            assert await saver.aget_tuple(config) is None
            with db.new_session() as session:
                assert store.get(session, f"task:{task['id']}") is None
                assert store.get(session, "task-key:cleanup-test")["processed"] is True
                assert store.visible(session)[0]["text"] == "保留聊天"
            await runtime.close()
    asyncio.run(run())


def test_gateway_auth_chat_and_restart(database, monkeypatch):
    from app.main import create_app

    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "model_id", "")
    with TestClient(create_app(), base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        csrf = client.get("/api/auth/session").json()["csrf"]
        response = client.post("/api/auth/register", headers={"x-csrf-token": csrf},
                               json={"username": "gateway-user", "password": "test"})
        assert response.status_code == 201
        csrf = response.json()["csrf"]
        chat = client.get("/api/chat").json()
        response = client.post("/api/chat/messages", headers={"x-csrf-token": csrf}, json={
            "assistant_id": chat["assistant_id"], "client_message_id": "gateway-message", "text": "真实存储"})
        assert response.status_code == 202
        assert client.get("/api/chat").json()["messages"][0]["text"] == "真实存储"
        cookies = dict(client.cookies)

    with TestClient(create_app(), base_url="http://127.0.0.1", client=("127.0.0.1", 12345),
                    cookies=cookies) as client:
        assert client.get("/api/chat").json()["messages"][0]["text"] == "真实存储"


def test_invalid_database_configuration_fails_without_fallback(tmp_path, monkeypatch):
    db.dispose_engine()
    monkeypatch.setattr(settings, "database_backend", "postgresql")
    monkeypatch.setattr(settings, "database_url", "")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        db.init_db(tmp_path)
    assert not (tmp_path / "app.db").exists()

    monkeypatch.setattr(settings, "database_url", "postgresql://test:test@127.0.0.1:1/test?connect_timeout=1")
    with pytest.raises(OperationalError):
        db.init_db(tmp_path)
    assert not (tmp_path / "app.db").exists()


def test_database_environment_and_url_round_trip(monkeypatch):
    monkeypatch.setenv("DATABASE_BACKEND", "postgresql")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:p%40ss%25word@localhost/test?sslmode=require")
    configured = load_settings()
    assert configured.database_backend == "postgresql"
    monkeypatch.setattr(settings, "database_url", configured.database_url)
    for url in (db.postgres_url(), db.postgres_url(sqlalchemy=True)):
        parsed = make_url(url)
        assert parsed.password == "p@ss%word" and parsed.query["sslmode"] == "require"


def test_old_schema_is_rejected_without_deleting_data(tmp_path):
    db.dispose_engine()
    path = tmp_path / "app.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE session (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO session VALUES ('old-data')")
    with pytest.raises(RuntimeError, match="旧数据库结构"):
        db.init_db(tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT id FROM session").fetchone() == ("old-data",)


def test_old_part_layout_is_rejected_without_creating_new_tables(tmp_path):
    db.dispose_engine()
    path = tmp_path / "app.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE part (session_id TEXT, key TEXT, sequence INTEGER, type TEXT, context_id TEXT, data TEXT)")
        connection.execute("INSERT INTO part VALUES ('user', 'task', 1, 'task_state', NULL, '{}')")
    with pytest.raises(RuntimeError, match="旧数据库结构"):
        db.init_db(tmp_path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT key FROM part").fetchone() == ("task",)
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not {"shared_state", "agent_message"}.intersection(names)


def test_checkpoint_cancellation_waits_for_io_and_closes_connection():
    async def run():
        started, release, closed = threading.Event(), threading.Event(), threading.Event()

        class Manager:
            def __enter__(self):
                started.set()
                release.wait(2)
                return self

            def __exit__(self, *args):
                closed.set()

        task = asyncio.create_task(_open_manager(Manager()))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

        started.clear()
        release.clear()

        def write():
            started.set()
            release.wait(2)

        task = asyncio.create_task(_run_io(write))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(run())
