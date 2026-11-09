import asyncio
import threading

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select

from app import db
from app.agent import shared_state
from app.agent.runtime import Conflict, Runtime
from app.agent.worker import Worker
from app.api.chat import router
from app.config import settings
from app.models import AgentMessage, Message, Part, Session, SharedState, TurnUsage
from app.sessions import chat_store as store
from app.sessions import execution_state, session_permissions
from app.user_context import scoped


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "log")
    (tmp_path / ("a" * 32)).mkdir()
    db.dispose_engine()
    db.init_db(tmp_path)
    yield tmp_path
    session_permissions.forget_session(scoped("main"))
    db.dispose_engine()


async def wait_for(check, timeout=4):
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.02)


def snapshot_now(runtime):
    with db.new_session() as session:
        return runtime._snapshot(session)


def test_existing_chat_gets_identity_without_losing_data(isolated):
    with db.new_session() as session:
        store.initialize(session, isolated)
        store.put(session, "meta", {"type": "meta", "revision": 12, "user_seq": 3})
        store.add_visible(session, "user", "已有对话")
        session.commit()
        store.initialize(session, isolated)
        meta = store.get(session, "meta")
        assert meta["assistant_id"] and meta["revision"] == 12 and meta["user_seq"] == 3
        assert store.visible(session)[0]["text"] == "已有对话"


def test_reset_waits_for_tool_and_deletes_old_context(isolated):
    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        release = threading.Event()
        started = threading.Event()
        tool = None
        try:
            workspace_file = isolated / ("a" * 32) / "result.txt"
            workspace_file.write_text("任务结果", encoding="utf-8")
            log_file = isolated / "log" / "model-io-main.jsonl"
            log_file.parent.mkdir()
            log_file.write_text("旧模型记录", encoding="utf-8")
            config_file = isolated / ".env"
            config_file.write_text("model=kept", encoding="utf-8")
            rules = {"version": 1, "allow": [{"toolName": "bash"}], "deny": [], "ask": []}
            execution_state.save_permission_rules(isolated, rules)
            execution_state.save_execution_state(isolated, "yolo")
            scope = session_permissions.scope_for("bash", {"command": "pwd"})
            session_permissions.grant(scoped("main"), scope)

            def seed(session):
                store.bump(session, user_event=True)
                store.add_visible(session, "user", "旧对话", message_id="user:old")
                task = runtime.create_task(session, "旧任务", 1)
                task.update(status="running", run_id="old-run")
                store.put(session, f"task:{task['id']}", task)
                shared_state.replace(session, task["context_id"], task["id"], "old-run",
                                     [{"content": "旧工作", "status": "in_progress"}])
                store.put(session, "helper:old", {"type": "agent_run", "context_id": "old", "task_id": task["id"], "status": "running"})
                store.put(session, "bus:old", {"type": "agent_message", "id": "old", "context_id": "main", "task_id": task["id"],
                                                   "content": "旧报告", "delivery": "processed"})
                store.put(session, "output:old", {"type": "tool_output", "id": "old", "task_id": task["id"], "content": "旧输出"})
                store.put(session, "summary:old", {"type": "agent_summary", "content": "旧摘要"})
                session.add(TurnUsage(session_id=scoped("main"), turn_id="old", status="completed", started_at=1))
                return task
            task = await runtime.change(seed)
            worker = Worker(runtime, task["id"])
            worker.context_id, worker.run_id = task["context_id"], "old-run"
            approval = asyncio.create_task(worker.approve({"id": "x", "call_id": "x", "name": "write_file",
                                                            "input": {"path": "x", "content": "x"}}, "需要确认"))
            runtime.workers["test-approval"] = approval
            await wait_for(lambda: bool(runtime.approval_events))
            request_id = next(iter(runtime.approval_events))

            def blocked_tool():
                started.set()
                release.wait(5)
                return "done"
            tool = asyncio.create_task(runtime.run_tool(blocked_tool))
            await wait_for(started.is_set)
            before = await runtime.snapshot()
            old_id = before["assistant_id"]
            result = await runtime.reset(old_id, "reset-one")
            assert result["status"] == "stopping"
            assert not (await approval)["approved"]
            runtime.workers.pop("test-approval")
            await asyncio.sleep(0.15)
            pending = await runtime.snapshot()
            assert pending["assistant_id"] == old_id and pending["reset"]["status"] == "stopping"
            assert pending["approvals"] == []
            with pytest.raises(Conflict):
                await runtime.accept("late", "旧页面消息", None, assistant_id=old_id)
            with pytest.raises(Conflict):
                await runtime.decide(request_id, "allowOnce", assistant_id=old_id)
            release.set()
            await tool
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
            after = await runtime.snapshot()
            assert after["reset"] is None and after["messages"] == [] and after["approvals"] == []
            assert after["revision"] > before["revision"] and after["permission_mode"] == "build"
            assert not session_permissions.allows(scoped("main"), "bash", {"command": "pwd"})
            assert execution_state.load_permission_rules(isolated) == rules
            assert workspace_file.read_text(encoding="utf-8") == "任务结果"
            assert log_file.read_text(encoding="utf-8") == "旧模型记录"
            assert config_file.read_text(encoding="utf-8") == "model=kept"
            replay = await runtime.reset(old_id, "reset-one")
            assert replay["status"] == "completed" and replay["new_assistant_id"] == after["assistant_id"]
            with pytest.raises(Conflict):
                await runtime.reset(after["assistant_id"], "reset-one")
            with pytest.raises(Conflict):
                await runtime.accept("late", "旧页面消息", None, assistant_id=old_id)
            with db.new_session() as session:
                assert [row.id for row in session.scalars(select(Session))] == [scoped("main")]
                assert session.get(Message, scoped("user:old")) is None
                assert session.get(TurnUsage, (scoped("main"), "old")) is None
                assert not list(session.scalars(select(AgentMessage).where(AgentMessage.session_id == scoped("main"))))
                assert not list(session.scalars(select(SharedState).where(SharedState.session_id == scoped("main"))))
                assert store.records(session, "context_pending", context_id="main") == []
                assert store.records(session, "task_state") == []
                assert store.get(session, "meta")["user_seq"] >= 1
            accepted = await runtime.accept("old", "新助手消息", "build", assistant_id=after["assistant_id"])
            assert accepted["user_seq"] > 1
        finally:
            release.set()
            if tool:
                await asyncio.gather(tool, return_exceptions=True)
            await runtime.close()
    asyncio.run(run())


def test_restart_unknown_requires_explicit_confirmation(isolated):
    async def run():
        with db.new_session() as session:
            store.initialize(session, isolated)
            meta = store.get(session, "meta")
            old_id = meta["assistant_id"]
            meta["reset"] = {"id": "restart", "assistant_id": old_id, "status": "stopping"}
            store.put(session, "meta", meta)
            store.put(session, "execution:old", {"type": "tool_execution", "id": "old", "status": "started"})
            session.commit()
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        try:
            snapshot = await runtime.snapshot()
            assert snapshot["reset"]["status"] == "blocked" and snapshot["reset"]["reason"] == "unknown"
            with db.new_session() as session:
                assert "结果未知" in store.get(session, "execution:old")["output"]
            assert (await runtime.reset(old_id, "restart"))["status"] == "blocked"
            await asyncio.sleep(0.2)
            assert (await runtime.snapshot())["assistant_id"] == old_id
            await runtime.reset(old_id, "restart", confirm_stopped=True)
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_previously_recovered_unknown_operation_requires_confirmation_on_later_reset(isolated):
    async def run():
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.put(session, "execution:old", {"type": "tool_execution", "id": "old", "status": "started"})
            session.commit()
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        try:
            old_id = (await runtime.snapshot())["assistant_id"]
            result = await runtime.reset(old_id, "later", confirm_stopped=True)
            assert result["status"] == "blocked" and result["reason"] == "unknown"
            await asyncio.sleep(0.2)
            assert (await runtime.snapshot())["assistant_id"] == old_id
            await runtime.reset(old_id, "later", confirm_stopped=True)
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_reset_waits_for_unknown_result_created_while_stopping(isolated):
    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        release = asyncio.Event()
        active = asyncio.create_task(release.wait())
        runtime.active_tools.add(active)
        try:
            old_id = (await runtime.snapshot())["assistant_id"]
            assert (await runtime.reset(old_id, "during-stop"))["status"] == "stopping"
            await runtime.change(lambda session: store.put(session, "execution:later", {
                "type": "tool_execution", "id": "later", "status": "completed", "result_unknown": True}))
            release.set()
            await active
            await wait_for(lambda: (snapshot_now(runtime)["reset"] or {}).get("status") == "blocked")
            assert (await runtime.snapshot())["assistant_id"] == old_id
            assert (await runtime.snapshot())["reset"]["reason"] == "unknown"
            await runtime.reset(old_id, "during-stop", confirm_stopped=True)
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
        finally:
            release.set()
            await asyncio.gather(active, return_exceptions=True)
            await runtime.close()
    asyncio.run(run())


def test_timeout_can_retry_or_cancel_without_resuming_stopped_tasks(isolated, monkeypatch):
    monkeypatch.setattr(settings, "bash_timeout", -4)
    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        release = threading.Event()
        started = threading.Event()
        tool = None
        try:
            def seed(session):
                task = runtime.create_task(session, "停止的任务", 1)
                task["status"] = "running"
                store.put(session, f"task:{task['id']}", task)
                return task
            task = await runtime.change(seed)
            def blocked_tool():
                started.set()
                release.wait(5)
            tool = asyncio.create_task(runtime.run_tool(blocked_tool))
            await wait_for(started.is_set)
            old_id = (await runtime.snapshot())["assistant_id"]
            await runtime.reset(old_id, "timeout")
            await wait_for(lambda: (snapshot_now(runtime)["reset"] or {}).get("status") == "blocked")
            assert (await runtime.snapshot())["reset"]["reason"] == "timeout"
            assert (await runtime.reset(old_id, "timeout"))["status"] == "stopping"
            await wait_for(lambda: (snapshot_now(runtime)["reset"] or {}).get("status") == "blocked")
            assert (await runtime.cancel_reset(old_id, "timeout"))["status"] == "cancelled"
            assert (await runtime.reset(old_id, "timeout"))["status"] == "cancelled"
            with db.new_session() as session:
                assert store.get(session, f"task:{task['id']}")["stop_requested"]
            await runtime.accept("after-cancel", "继续旧对话", None, assistant_id=old_id)
            await runtime.reset(old_id, "retry")
            release.set()
            await tool
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
        finally:
            release.set()
            if tool:
                await asyncio.gather(tool, return_exceptions=True)
            await runtime.close()
    asyncio.run(run())


def test_cancel_pending_then_immediately_restart_reset(isolated):
    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        release, started = threading.Event(), threading.Event()
        tool = None
        try:
            def blocked_tool():
                started.set()
                release.wait(5)
            tool = asyncio.create_task(runtime.run_tool(blocked_tool))
            await wait_for(started.is_set)
            old_id = (await runtime.snapshot())["assistant_id"]
            await runtime.reset(old_id, "first")
            assert (await runtime.cancel_reset(old_id, "first"))["status"] == "cancelled"
            await runtime.reset(old_id, "second")
            assert (await runtime.cancel_reset(old_id, "first"))["status"] == "cancelled"
            release.set()
            await tool
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
            assert (await runtime.reset(old_id, "second"))["status"] == "completed"
        finally:
            release.set()
            if tool:
                await asyncio.gather(tool, return_exceptions=True)
            await runtime.close()
    asyncio.run(run())


def test_restart_resumes_reset_without_unknown_operation(isolated):
    async def run():
        with db.new_session() as session:
            store.initialize(session, isolated)
            meta = store.get(session, "meta")
            old_id = meta["assistant_id"]
            meta["reset"] = {"id": "resume", "assistant_id": old_id, "status": "stopping"}
            store.put(session, "meta", meta)
            session.commit()
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        try:
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
            assert (await runtime.reset(old_id, "resume"))["status"] == "completed"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_reset_waits_for_cancelled_main_call_to_finish(isolated):
    class LateModel:
        model = "fake"

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, **_kwargs):
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
            yield {"type": "text_delta", "text": "旧助手的迟到回复"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = LateModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            old_id = (await runtime.snapshot())["assistant_id"]
            await runtime.accept("old", "等待旧助手", None, assistant_id=old_id)
            await asyncio.wait_for(model.started.wait(), 3)
            await runtime.reset(old_id, "late-model")
            await asyncio.sleep(0.2)
            assert (await runtime.snapshot())["assistant_id"] == old_id
            model.release.set()
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
            assert (await runtime.snapshot())["messages"] == []
        finally:
            model.release.set()
            await runtime.close()
    asyncio.run(run())


def test_duplicate_reset_does_not_interrupt_main_cleanup(isolated):
    async def run():
        runtime = Runtime(client=object())
        runtime.client = None
        await runtime.start()
        started, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))
        async def main():
            try:
                started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release.wait()
                await runtime.change(lambda session: store.add_visible(session, "assistant", "已保存的中断结果"))
                cleaned.set()
                raise
        runtime.main_call = asyncio.create_task(main())
        try:
            await asyncio.wait_for(started.wait(), 3)
            old_id = (await runtime.snapshot())["assistant_id"]
            await runtime.reset(old_id, "duplicate")
            await asyncio.wait_for(cleaning.wait(), 3)
            await runtime.reset(old_id, "duplicate")
            assert runtime.main_call.cancelling() == 1
            release.set()
            await asyncio.wait_for(cleaned.wait(), 3)
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
            await runtime.reset(old_id, "duplicate")
            with db.new_session() as session:
                parts = session.scalars(select(Part).where(Part.session_id == scoped("main")))
                assert not any(store.read(part).get("text") == "已保存的中断结果" for part in parts)
        finally:
            release.set()
            await runtime.close()
    asyncio.run(run())


def test_cancel_reset_does_not_replay_interrupted_input(isolated):
    class BlockingModel:
        model = "fake"

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def stream(self, **_kwargs):
            self.calls += 1
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
            yield {"type": "text_delta", "text": "迟到结果"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = BlockingModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            old_id = (await runtime.snapshot())["assistant_id"]
            await runtime.accept("pending", "尚未路由的任务", None, assistant_id=old_id)
            await asyncio.wait_for(model.started.wait(), 3)
            await runtime.reset(old_id, "cancel-pending")
            await runtime.cancel_reset(old_id, "cancel-pending")
            model.release.set()
            await wait_for(lambda: runtime.main_call is None)
            await asyncio.sleep(0.3)
            with db.new_session() as session:
                assert store.get(session, "input:pending")["status"] == "cancelled"
                assert store.records(session, "task_state") == []
            assert model.calls == 1
            with pytest.raises(Conflict, match="中断"):
                await runtime.accept("pending", "尚未路由的任务", None, assistant_id=old_id)
            assert not (await runtime.accept("new", "尚未路由的任务", None, assistant_id=old_id))["replayed"]
        finally:
            model.release.set()
            await runtime.close()
    asyncio.run(run())


def test_reset_ignores_late_main_tool_call(isolated):
    class LateReconcile:
        model = "fake"

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.task_id = ""
            self.calls = 0

        async def stream(self, **_kwargs):
            self.calls += 1
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
            yield {"type": "tool_use", "id": "late", "name": "task_update",
                   "input": {"task_id": self.task_id, "instruction": "继续旧任务"}}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = LateReconcile()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.bump(session, user_event=True)
            task = runtime.create_task(session, "旧任务", 1)
            task.update(status="running", run_id="old")
            store.put(session, f"task:{task['id']}", task)
            event = {"type": "user_event", "event_type": "message", "id": "event", "message_id": "user:event",
                     "user_seq": 1, "status": "processed", "text": "旧要求"}
            store.put(session, "input:event", event)
            session.commit()
        model.task_id = task["id"]
        runtime.main_call = asyncio.create_task(runtime._dialog("处理旧要求", source_seq=1))
        try:
            await asyncio.wait_for(model.started.wait(), 3)
            old_id = (await runtime.snapshot())["assistant_id"]
            await runtime.reset(old_id, "reconcile-reset")
            model.release.set()
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
            await runtime.reset(old_id, "reconcile-reset")
            with db.new_session() as session:
                parts = session.scalars(select(Part).where(Part.session_id == scoped("main")))
                assert not any("继续旧任务" in str(store.read(part)) for part in parts)
            assert model.calls == 1
        finally:
            model.release.set()
            await runtime.close()
    asyncio.run(run())


def test_deletion_failure_cannot_cancel_and_can_retry(isolated, monkeypatch):
    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        try:
            await runtime.change(lambda session: store.add_visible(session, "user", "保留的对话"))
            execution_state.save_execution_state(isolated, "edit")
            old_id = (await runtime.snapshot())["assistant_id"]
            clear = runtime.memory.clear
            async def fail_clear():
                raise RuntimeError("test failure")
            monkeypatch.setattr(runtime.memory, "clear", fail_clear)
            await runtime.reset(old_id, "archive-error")
            await wait_for(lambda: (snapshot_now(runtime)["reset"] or {}).get("status") == "blocked")
            snapshot = await runtime.snapshot()
            assert snapshot["assistant_id"] == old_id and snapshot["messages"][0]["text"] == "保留的对话"
            assert snapshot["permission_mode"] == "edit" and snapshot["reset"]["reason"] == "error"
            assert snapshot["reset"]["deletion_started"]
            with pytest.raises(Conflict, match="不能取消"):
                await runtime.cancel_reset(old_id, "archive-error")
            with db.new_session() as session:
                assert [row.id for row in session.scalars(select(Session))] == [scoped("main")]
            monkeypatch.setattr(runtime.memory, "clear", clear)
            await runtime.reset(old_id, "archive-error")
            await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old_id)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_http_requires_assistant_identity_and_rejects_stale_tab(isolated):
    async def run():
        runtime = Runtime()
        runtime.client = None
        await runtime.start()
        app = FastAPI()
        from conftest import install_runtime
        install_runtime(app, runtime)
        app.include_router(router)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                old = (await client.get("/api/chat")).json()
                assert (await client.post("/api/chat/messages", json={"client_message_id": "1", "text": "hello"})).status_code == 422
                assert (await client.post("/api/chat/approvals/old", json={"option_id": "deny"})).status_code == 422
                body = {"assistant_id": old["assistant_id"], "reset_id": "http-reset"}
                assert (await client.post("/api/chat/reset", json=body)).status_code == 202
                await wait_for(lambda: snapshot_now(runtime)["assistant_id"] != old["assistant_id"])
                new = (await client.get("/api/chat")).json()
                stale = {"assistant_id": old["assistant_id"], "client_message_id": "late", "text": "stale"}
                assert (await client.post("/api/chat/messages", json=stale)).status_code == 409
                assert (await client.post("/api/chat/approvals/old", json={"assistant_id": old["assistant_id"], "option_id": "deny"})).status_code == 409
                assert new["messages"] == [] and new["revision"] > old["revision"]
                assert (await client.post("/api/chat/reset", json=body)).json()["status"] == "completed"
        finally:
            await runtime.close()
    asyncio.run(run())
