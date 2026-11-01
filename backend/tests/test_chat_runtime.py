import asyncio
import json
import socket

import pytest

from app import db
from app.agent.runtime import Conflict, Runtime
from app.config import settings
from app.models import new_id
from app.sessions import chat_store as store


class FakeModel:
    model = "fake"

    def __init__(self):
        self.worker_started = asyncio.Event()
        self.release_worker = asyncio.Event()
        self.max_workers = 0
        self.workers = 0

    async def stream(self, *, system, messages, tools, tool_choice=None):
        if tool_choice and tool_choice["name"] == "route_input":
            request = json.loads(messages[-1]["content"])
            yield {"type": "tool_use", "id": new_id(), "name": "route_input", "input": {
                "message_id": request["message_id"], "user_seq": request["user_seq"],
                "intents": [{"type": "create" if request["text"].startswith("任务") else "chat",
                             "instruction": request["text"]}], "needs_clarification": False}}
        elif system.startswith("你是后台工作助手"):
            self.workers += 1
            self.max_workers = max(self.max_workers, self.workers)
            self.worker_started.set()
            try:
                await self.release_worker.wait()
                yield {"type": "text_delta", "text": "工作已完成"}
            finally:
                self.workers -= 1
        elif tool_choice and tool_choice["name"] == "task_review":
            tasks = json.loads(messages[-1]["content"].split("当前任务：", 1)[1])
            task = next(t for t in tasks if t["status"] == "reviewing")
            yield {"type": "tool_use", "id": new_id(), "name": "task_review",
                   "input": {"task_id": task["id"], "decision": "complete"}}
        else:
            report_seen = "一轮结束" in json.dumps(messages, ensure_ascii=False)
            yield {"type": "text_delta", "text": "任务已完成。" if report_seen else "收到。"}
        yield {"type": "stop_reason", "reason": "end_turn"}


async def wait_for(check, timeout=4):
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.02)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    (tmp_path / "workspace").mkdir()
    db.dispose_engine()
    db.init_db(tmp_path)
    yield tmp_path
    db.dispose_engine()


def test_background_chat_and_idempotent_acceptance(isolated):
    async def run():
        model = FakeModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            first = await runtime.accept("one", "任务：整理文件", "build")
            await asyncio.wait_for(model.worker_started.wait(), 3)
            queue, before = await runtime.subscribe()
            runtime.unsubscribe(queue)
            assert before["messages"][0]["text"] == "任务：整理文件"
            second = await runtime.accept("two", "顺便问个问题", "build")
            assert first["user_seq"] < second["user_seq"]
            assert (await runtime.accept("one", "任务：整理文件", "build"))["replayed"]
            with pytest.raises(Conflict):
                await runtime.accept("one", "不同内容", "build")
            model.release_worker.set()
            def complete():
                with db.new_session() as session:
                    return any(t["status"] == "completed" for t in store.records(session, "task_state"))
            await wait_for(complete)
            snapshot = await runtime.snapshot()
            assert [m["text"] for m in snapshot["messages"] if m["role"] == "user"] == ["任务：整理文件", "顺便问个问题"]
            assert model.max_workers <= 3
        finally:
            await runtime.close()
    asyncio.run(run())


def test_restart_keeps_unknown_tool_result_and_invalidates_old_approval(isolated):
    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "恢复任务", 1)
            task.update(status="waiting_user", run_id="run1")
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:one", {"type": "tool_execution", "id": "one", "status": "started"})
            store.put(session, "approval:old", {"type": "approval", "request_id": "old", "status": "requested"})
            session.commit()
        runtime.client = None
        await runtime.start()
        try:
            with db.new_session() as session:
                assert "结果未知" in store.get(session, "execution:one")["output"]
                assert store.get(session, "approval:old")["status"] == "expired"
                assert store.get(session, f"task:{task['id']}")["status"] == "queued"
            with pytest.raises(Conflict):
                await runtime.decide("old", "allowOnce")
        finally:
            await runtime.close()
    asyncio.run(run())


def test_newer_approval_is_not_overridden_by_older_message(isolated):
    with db.new_session() as session:
        store.initialize(session, isolated)
        runtime = Runtime(client=FakeModel())
        task = runtime.create_task(session, "删除旧文件", 1)
        event = {"type": "user_event", "event_type": "message", "id": "old", "message_id": "user:old",
                 "text": "不要删了，同时整理清单", "user_seq": 2, "status": "pending"}
        store.put(session, "input:old", event)
        store.put(session, "decision:new", {"type": "user_event", "event_type": "approval",
                  "id": "new", "task_id": task["id"], "user_seq": 3, "status": "processed"})
        route = {"needs_clarification": False, "intents": [
            {"type": "stop", "task_id": task["id"], "instruction": "不要删除"},
            {"type": "create", "instruction": "整理清单"}]}
        notes = runtime._apply_route(session, event, route)
        session.commit()
        assert "更新的审批决定" in notes
        assert not store.get(session, f"task:{task['id']}")["stop_requested"]
        assert len(store.records(session, "task_state")) == 2


@pytest.mark.parametrize("conflict", [False, True])
def test_newer_message_rejudges_only_older_task_action(isolated, conflict):
    class ReconcileModel(FakeModel):
        async def stream(self, *, system, messages, tools, tool_choice=None):
            if system.startswith("重新判断"):
                data = json.loads(messages[-1]["content"])
                original = data["original_intent"]
                intent = {"type": "chat", "instruction": "较新的消息取代了旧要求"} if conflict else original
                yield {"type": "tool_use", "id": new_id(), "name": "route_input", "input": {
                    "message_id": data["message_id"], "user_seq": data["user_seq"],
                    "intents": [intent], "needs_clarification": False}}
                yield {"type": "stop_reason", "reason": "tool_use"}
                return
            async for item in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield item

    async def run():
        runtime = Runtime(client=ReconcileModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "任务 A", 1)
            event = {"type": "user_event", "event_type": "message", "id": "old", "message_id": "user:old",
                     "text": "停止 A，并整理 B", "user_seq": 1, "status": "pending"}
            store.put(session, "input:old", event)
            store.put(session, "input:new", {"type": "user_event", "event_type": "message", "id": "new",
                      "message_id": "user:new", "text": "继续 A" if conflict else "另外查一下天气",
                      "user_seq": 2, "status": "processed"})
            meta = store.get(session, "meta")
            meta["user_seq"] = 2
            store.put(session, "meta", meta)
            route = {"needs_clarification": False, "intents": [
                {"type": "stop", "task_id": task["id"], "instruction": "停止 A"},
                {"type": "create", "instruction": "整理 B"}]}
            runtime._apply_route(session, event, route)
            session.commit()
            assert not store.get(session, f"task:{task['id']}")["stop_requested"]
            assert len(store.records(session, "task_state")) == 2
            report = store.records(session, "agent_message", context_id="main")[0]
        await runtime._report(report)
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}")["stop_requested"] is (not conflict)
            assert store.get(session, f"bus:{report['id']}")["delivery"] == "processed"
    asyncio.run(run())


def test_review_requires_delivered_result(isolated):
    from app.agent import agent_bus
    with db.new_session() as session:
        store.initialize(session, isolated)
        runtime = Runtime(client=FakeModel())
        task = runtime.create_task(session, "读取文件", 1)
        task["status"] = "reviewing"
        store.put(session, f"task:{task['id']}", task)
        agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                       content="结果：371", kind="review", message_id="review")
        args = {"task_id": task["id"], "decision": "complete"}
        assert "尚未投递" in runtime._main_action(session, "task_review", args, None, "early")
        assert store.get(session, f"task:{task['id']}")["status"] == "reviewing"
        report = store.get(session, "bus:review")
        agent_bus.deliver(session, report)
        assert "completed" in runtime._main_action(session, "task_review", args, None, "proper", "review")
        session.commit()


def test_old_run_report_cannot_review_new_run(isolated):
    from app.agent import agent_bus
    with db.new_session() as session:
        store.initialize(session, isolated)
        runtime = Runtime(client=FakeModel())
        task = runtime.create_task(session, "处理任务", 1)
        task.update(status="reviewing", run_id="new-run")
        store.put(session, f"task:{task['id']}", task)
        agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                       content="旧轮结果", kind="review", message_id="old-report", run_id="old-run")
        agent_bus.deliver(session, store.get(session, "bus:old-report"))
        result = runtime._main_action(session, "task_review", {"task_id": task["id"], "decision": "complete"},
                                      None, "stale", "old-report")
        assert result.startswith("Error:")
        assert store.get(session, f"task:{task['id']}")["status"] == "reviewing"


def test_http_stream_disconnect_keeps_worker_running(isolated):
    import httpx
    import uvicorn
    from fastapi import FastAPI
    from app.api.chat import router

    async def run():
        model = FakeModel()
        runtime = Runtime(client=model)
        await runtime.start()
        app = FastAPI()
        app.state.runtime = runtime
        app.include_router(router)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            await wait_for(lambda: server.started)
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=4) as client:
                async with client.stream("GET", "/api/chat/events") as response:
                    assert response.status_code == 200
                    assert (await response.aiter_lines().__anext__()).startswith("data:")
                sent = await client.post("/api/chat/messages", json={"client_message_id": "http-one", "text": "任务：后台执行"})
                assert sent.status_code == 202
                await asyncio.wait_for(model.worker_started.wait(), 3)
                model.release_worker.set()
                def complete():
                    with db.new_session() as session:
                        return any(t["status"] == "completed" for t in store.records(session, "task_state"))
                await wait_for(complete)
                await wait_for(lambda: any(m["text"] == "任务已完成。" for m in asyncio_snapshot(runtime)))
                snapshot = (await client.get("/api/chat")).json()
                assert snapshot["messages"][0]["text"] == "任务：后台执行"
                async with client.stream("GET", "/api/chat/events") as response:
                    line = await response.aiter_lines().__anext__()
                    assert json.loads(line.removeprefix("data: "))["data"]["revision"] >= snapshot["revision"]
        finally:
            server.should_exit = True
            await server_task
            listener.close()
            await runtime.close()
    asyncio.run(run())


def asyncio_snapshot(runtime):
    with db.new_session() as session:
        return store.visible(session)


def test_process_lock_rejects_second_instance(isolated):
    from app.process_lock import ProcessLock
    first, second = ProcessLock(isolated), ProcessLock(isolated)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="已有运行中的后端"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_approval_stays_available_and_session_grant_is_shared(isolated):
    from app.agent.worker import Worker
    from app.sessions import session_permissions

    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "写文件", 1)
            task.update(status="running", run_id="run1")
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run1"
        execution = {"id": "call1", "call_id": "c1", "name": "write_file", "input": {"path": "x", "content": "a"}}
        waiting = asyncio.create_task(worker.approve(execution, "需要确认"))
        try:
            await wait_for(lambda: bool(runtime.approval_events))
            request_id = next(iter(runtime.approval_events))
            message = await runtime.accept("before-approval", "不要写入", None)
            assert (await runtime.snapshot())["approvals"][0]["request_id"] == request_id
            assert (await runtime.decide(request_id, "allowSession"))["ok"]
            assert (await waiting)["approved"]
            assert session_permissions.allows("main", "write_file", {"path": "x", "content": "b"})
            with db.new_session() as session:
                approval = store.get(session, f"approval:{request_id}")
                assert approval["user_seq"] > message["user_seq"]
            with pytest.raises(Conflict):
                await runtime.decide(request_id, "deny")
        finally:
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
            session_permissions.forget_session("main")
    asyncio.run(run())


def test_file_owner_requires_reread_before_other_task_writes(isolated):
    from app.agent.worker import Worker
    from app.sessions import execution_state

    async def run():
        runtime = Runtime(client=FakeModel())
        execution_state.save_execution_state(isolated, "edit")
        with db.new_session() as session:
            store.initialize(session, isolated)
            tasks = [runtime.create_task(session, f"任务 {i}", i) for i in (1, 2)]
            for task in tasks:
                task.update(status="running", run_id=new_id())
                store.put(session, f"task:{task['id']}", task)
            session.commit()
        workers = [Worker(runtime, t["id"]) for t in tasks]
        for worker, task in zip(workers, tasks):
            worker.context_id, worker.run_id = task["context_id"], task["run_id"]

        async def call(worker, name, inputs):
            execution = {"id": new_id(), "call_id": new_id(), "name": name, "input": inputs, "status": "pending"}
            await runtime.change(lambda session: store.put(session, f"execution:{execution['id']}", {
                **execution, "type": "tool_execution"}))
            return await worker.file_tool(execution)

        assert "已写入" in await call(workers[0], "write_file", {"path": "x.txt", "content": "原始"})
        assert "原始" in await call(workers[1], "read_file", {"path": "x.txt"})
        pending = asyncio.create_task(call(workers[1], "write_file", {"path": "x.txt", "content": "覆盖"}))
        def is_waiting():
            with db.new_session() as session:
                return any(w["status"] == "waiting" for w in store.records(session, "file_wait"))
        await wait_for(is_waiting)
        def close_owner(session):
            task = store.get(session, f"task:{tasks[0]['id']}")
            task["status"] = "completed"
            store.put(session, f"task:{task['id']}", task)
        await runtime.change(close_owner)
        assert "重新读取" in await asyncio.wait_for(pending, 3)
        assert "原始" in await call(workers[1], "read_file", {"path": "x.txt"})
        assert "已写入" in await call(workers[1], "write_file", {"path": "x.txt", "content": "覆盖"})
        assert (isolated / "workspace" / "x.txt").read_text(encoding="utf-8") == "覆盖"
    asyncio.run(run())


def test_thirtieth_tool_step_requires_review(isolated, monkeypatch):
    from app.agent.worker import run_worker
    from app.agent import tools

    class ToolModel:
        model = "fake"
        async def stream(self, **_kwargs):
            yield {"type": "tool_use", "id": "last", "name": "glob", "input": {"pattern": "*.txt"}}
            yield {"type": "stop_reason", "reason": "tool_use"}

    async def run():
        runtime = Runtime(client=ToolModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "搜索文件", 1)
            task.update(status="running", run_id="run", steps=29)
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        monkeypatch.setattr(tools, "execute_tool", lambda *_args: "无匹配")
        await run_worker(runtime, task["id"])
        with db.new_session() as session:
            current = store.get(session, f"task:{task['id']}")
            assert current["steps"] == 30 and current["status"] == "reviewing"
            assert "30 步上限" in store.records(session, "agent_message", context_id="main")[0]["content"]
    asyncio.run(run())


def test_scheduler_caps_parallel_workers_at_three(isolated):
    async def run():
        model = FakeModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            def create(session):
                for number in range(4):
                    runtime.create_task(session, f"任务 {number}", number + 1)
            await runtime.change(create)
            await wait_for(lambda: model.workers == 3)
            await asyncio.sleep(0.4)
            with db.new_session() as session:
                states = [t["status"] for t in store.records(session, "task_state")]
            assert states.count("running") == 3 and states.count("queued") == 1
            assert model.max_workers == 3
            model.release_worker.set()
            def all_done():
                with db.new_session() as session:
                    return all(t["status"] == "completed" for t in store.records(session, "task_state"))
            await wait_for(all_done, timeout=8)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_helper_allocation_and_busy_message_delivery(isolated):
    from app.agent.worker import Worker

    async def run():
        model = FakeModel()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "协作任务", 1)
            task.update(status="running", run_id="root-run")
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        owner = Worker(runtime, task["id"])
        owner.context_id, owner.run_id = task["context_id"], task["run_id"]
        hold = asyncio.create_task(asyncio.Event().wait())
        runtime.workers[task["id"]] = hold
        try:
            accepted = await owner.communicate("agent_help", {"goal": "检查资料"}, "help-call")
            assert "协助已受理" in accepted
            await runtime._schedule_once()
            await asyncio.wait_for(model.worker_started.wait(), 3)
            with db.new_session() as session:
                helper = store.records(session, "agent_run")[0]
            sent = await owner.communicate("agent_send", {"context_id": helper["context_id"], "content": "补充说明"}, "message-call")
            assert "消息已受理" in sent
            with db.new_session() as session:
                pending = store.records(session, "agent_message", context_id=helper["context_id"], delivery="pending")
            assert pending[0]["content"].endswith("补充说明")
            model.release_worker.set()
            await wait_for(lambda: f"helper:{helper['context_id']}" not in runtime.workers)
            with db.new_session() as session:
                result = store.records(session, "agent_message", context_id=owner.context_id, delivery="pending")
            assert "协助助手结果" in result[0]["content"]
        finally:
            hold.cancel()
            await asyncio.gather(hold, return_exceptions=True)
            await runtime.close()
    asyncio.run(run())


def test_compaction_preserves_raw_history_and_isolates_contexts(isolated, monkeypatch):
    from app.agent import compact

    async def run():
        monkeypatch.setattr(settings, "context_window", 700)
        monkeypatch.setattr(settings, "compact_output_reserve", 50)
        monkeypatch.setattr(settings, "compact_buffer", 50)
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            for i in range(5):
                store.append_history(session, "ctx-a", "user", f"请求{i}" + "文" * 500)
                store.append_history(session, "ctx-a", "assistant", f"答复{i}" + "字" * 500)
            store.append_history(session, "ctx-a", "user", "最新请求")
            store.append_history(session, "ctx-b", "user", "独立任务")
            session.commit()
        projected = await compact.context_history(runtime, "ctx-a")
        assert "此前对话摘要" in projected[0]["content"]
        assert projected[-1]["content"] == "最新请求"
        with db.new_session() as session:
            assert len(store.history(session, "ctx-a")) == 11
            assert store.history(session, "ctx-b") == [{"role": "user", "content": "独立任务"}]
            assert store.get(session, "summary:ctx-a")["boundary_id"]
            assert store.get(session, "summary:ctx-b") is None
    asyncio.run(run())


def test_slow_subscriber_disconnects_and_can_resnapshot(isolated):
    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
        queue, initial = await runtime.subscribe()
        for i in range(130):
            await runtime.change(lambda session, n=i: store.put(session, f"test:{n}", {"type": "test"}))
        assert await queue.get() is None
        fresh_queue, fresh = await runtime.subscribe()
        assert fresh["revision"] > initial["revision"]
        runtime.unsubscribe(fresh_queue)
    asyncio.run(run())


def test_restart_reissues_pending_approval(isolated):
    async def run():
        model = FakeModel()
        model.release_worker.set()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "写入文件", 1)
            task.update(status="waiting_user", run_id="run", steps=1)
            store.put(session, f"task:{task['id']}", task)
            execution_id = f"{task['id']}:run:c1"
            store.append_history(session, task["context_id"], "assistant", [{
                "type": "tool_use", "id": "c1", "name": "write_file", "input": {"path": "x", "content": "x"}}])
            store.put(session, f"execution:{execution_id}", {"type": "tool_execution", "id": execution_id,
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run", "call_id": "c1",
                      "name": "write_file", "input": {"path": "x", "content": "x"}, "status": "pending"})
            group_id = f"{task['id']}:run:1"
            store.put(session, f"group:{group_id}", {"type": "tool_group", "id": group_id,
                      "context_id": task["context_id"], "calls": [execution_id], "status": "pending"})
            store.put(session, "approval:old", {"type": "approval", "request_id": "old", "status": "requested"})
            session.commit()
        await runtime.start()
        try:
            await wait_for(lambda: bool(runtime.approval_events))
            new_request = next(iter(runtime.approval_events))
            assert new_request != "old"
            with db.new_session() as session:
                assert store.get(session, "approval:old")["status"] == "expired"
            await runtime.decide(new_request, "deny")
            def execution_finished():
                with db.new_session() as session:
                    return store.get(session, f"execution:{execution_id}")["status"] == "completed"
            await wait_for(execution_finished)
            assert not (isolated / "workspace" / "x").exists()
        finally:
            await runtime.close()
    asyncio.run(run())


def test_started_external_tool_is_not_replayed_on_restart(isolated):
    async def run():
        model = FakeModel()
        model.release_worker.set()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "执行操作", 1)
            task.update(status="running", run_id="run", steps=1)
            store.put(session, f"task:{task['id']}", task)
            identity = f"{task['id']}:run:c1"
            store.append_history(session, task["context_id"], "assistant", [{
                "type": "tool_use", "id": "c1", "name": "write_file", "input": {"path": "x", "content": "x"}}])
            store.put(session, f"execution:{identity}", {"type": "tool_execution", "id": identity,
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run", "call_id": "c1",
                      "name": "write_file", "input": {"path": "x", "content": "x"}, "status": "started"})
            group_id = f"{task['id']}:run:1"
            store.put(session, f"group:{group_id}", {"type": "tool_group", "id": group_id,
                      "context_id": task["context_id"], "calls": [identity], "status": "pending"})
            session.commit()
        await runtime.start()
        try:
            def paired():
                with db.new_session() as session:
                    return store.get(session, f"group:{group_id}")["status"] == "completed"
            await wait_for(paired)
            with db.new_session() as session:
                blocks = [block for message in store.history(session, task["context_id"])
                          if isinstance(message["content"], list) for block in message["content"]]
                result = next(block for block in blocks if block["type"] == "tool_result")
                assert result["is_error"] and "结果未知" in result["content"]
            assert not (isolated / "workspace" / "x").exists()
        finally:
            await runtime.close()
    asyncio.run(run())


def test_new_message_interrupts_only_main_reply(isolated):
    class InterruptedModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.first = True

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if tool_choice:
                async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                    yield event
            elif self.first:
                self.first = False
                yield {"type": "text_delta", "text": "已经输出"}
                self.started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    yield {"type": "text_delta", "text": "迟到的旧回复"}
            else:
                yield {"type": "text_delta", "text": "收到新问题"}
                yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = InterruptedModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await runtime.accept("one", "第一个问题", None)
            await asyncio.wait_for(model.started.wait(), 3)
            await runtime.accept("two", "第二个问题", None)
            def replied():
                with db.new_session() as session:
                    return any(m["text"] == "收到新问题" for m in store.visible(session))
            await wait_for(replied)
            messages = (await runtime.snapshot())["messages"]
            partial = next(m for m in messages if m["text"] == "已经输出")
            assert partial["status"] == "interrupted"
            assert all("迟到的旧回复" not in m["text"] for m in messages)
            assert [m["text"] for m in messages if m["role"] == "user"] == ["第一个问题", "第二个问题"]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_dependency_yields_and_known_wait_cycle_is_reported(isolated):
    from app.agent.worker import run_worker

    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            first = runtime.create_task(session, "先做 A", 1)
            later = runtime.create_task(session, "后做 B", 2)
            first.update(status="running", run_id="same-run", steps=4, files=["held-file"])
            store.put(session, f"task:{first['id']}", first)
            store.put(session, "file-wait:test", {"type": "file_wait", "task_id": later["id"],
                      "owner": first["id"], "status": "waiting", "path": "held-file"})
            assert "依赖已登记" in runtime._main_action(session, "task_wait_for", {
                "task_id": first["id"], "dependency_id": later["id"]}, None, "dependency")
            assert store.records(session, "agent_message", context_id="main")[0]["kind"] == "cycle"
            session.commit()
        await run_worker(runtime, first["id"])
        with db.new_session() as session:
            paused = store.get(session, f"task:{first['id']}")
            inherited = store.get(session, f"task:{later['id']}")
            assert (paused["status"], paused["run_id"], paused["steps"]) == ("waiting_dependency", "same-run", 4)
            assert inherited["order"] == 1
            inherited["status"] = "completed"
            runtime._release_dependents(session, inherited)
            session.commit()
            resumed = store.get(session, f"task:{first['id']}")
            assert (resumed["status"], resumed["run_id"], resumed["steps"]) == ("queued", "same-run", 4)
    asyncio.run(run())


def test_cross_task_messages_and_mixed_wait_graph(isolated):
    from app.agent import agent_bus
    from app.agent.worker import Worker

    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            first = runtime.create_task(session, "A", 1)
            second = runtime.create_task(session, "B", 2)
            first["status"] = second["status"] = "running"
            first["dependency"] = second["id"]
            store.put(session, f"task:{first['id']}", first)
            store.put(session, f"task:{second['id']}", second)
            store.put(session, "file-wait:b", {"type": "file_wait", "task_id": second["id"],
                      "context_id": second["context_id"], "owner": first["id"], "status": "waiting"})
            assert agent_bus.find_cycles(agent_bus.wait_graph(session))
            session.commit()
        sender = Worker(runtime, first["id"])
        sender.context_id, sender.run_id = first["context_id"], "run-a"
        result = await sender.communicate("agent_send", {"context_id": second["context_id"], "content": "补充资料"}, "cross")
        assert "消息已受理" in result
        with db.new_session() as session:
            message = store.records(session, "agent_message", context_id=second["context_id"])[0]
            assert message["source"] == first["context_id"] and "补充资料" in message["content"]
    asyncio.run(run())


def test_review_blocks_newer_tasks_and_continue_starts_new_round(isolated):
    from app.agent import agent_bus

    async def run():
        model = FakeModel()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            first = runtime.create_task(session, "A", 1)
            later = runtime.create_task(session, "B", 2)
            first.update(status="reviewing", run_id="previous", steps=30)
            store.put(session, f"task:{first['id']}", first)
            agent_bus.send(session, source=first["context_id"], target="main", task_id=first["id"],
                           content="达到 30 步", kind="review", message_id="review-a", run_id="previous")
            agent_bus.deliver(session, store.get(session, "bus:review-a"))
            session.commit()
        try:
            await runtime._schedule_once()
            assert not runtime.workers
            def continue_task(session):
                return runtime._main_action(session, "task_review", {"task_id": first["id"], "decision": "continue"},
                                            None, "continue", "review-a")
            assert "queued" in await runtime.change(continue_task)
            await runtime._schedule_once()
            await asyncio.wait_for(model.worker_started.wait(), 3)
            with db.new_session() as session:
                resumed = store.get(session, f"task:{first['id']}")
                waiting = store.get(session, f"task:{later['id']}")
                assert resumed["run_id"] != "previous" and resumed["steps"] == 1
                assert waiting["status"] == "queued"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_agent_wait_delivers_messages_before_and_after_registration(isolated):
    from app.agent import agent_bus
    from app.agent.worker import Worker

    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            a = runtime.create_task(session, "A", 1)
            b = runtime.create_task(session, "B", 2)
            a["status"] = b["status"] = "running"
            store.put(session, f"task:{a['id']}", a)
            store.put(session, f"task:{b['id']}", b)
            agent_bus.send(session, source=b["context_id"], target=a["context_id"], task_id=b["id"], content="提前回复")
            session.commit()
        worker = Worker(runtime, a["id"])
        worker.context_id, worker.run_id = a["context_id"], "run"
        assert await worker.wait_for_agent(b["context_id"], "wait-one") == "提前回复"
        assert await worker.wait_for_agent(b["context_id"], "wait-one") == "提前回复"
        waiting = asyncio.create_task(worker.wait_for_agent(b["context_id"], "wait-two"))
        try:
            def registered():
                with db.new_session() as session:
                    record = store.get(session, "agent-wait:wait-two")
                    return bool(record and record["status"] == "waiting")
            await wait_for(registered)
            await runtime.change(lambda session: agent_bus.send(session, source=b["context_id"], target=a["context_id"],
                                 task_id=b["id"], content="随后回复"))
            assert await asyncio.wait_for(waiting, 3) == "随后回复"
        finally:
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
    asyncio.run(run())


def test_report_delivery_failure_retries_without_duplicate_context(isolated):
    from app.agent import agent_bus

    class FlakyModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.failed = False

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if tool_choice and tool_choice["name"] == "task_review" and not self.failed:
                self.failed = True
                raise RuntimeError("临时模型错误")
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run():
        runtime = Runtime(client=FlakyModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "检查工作", 1)
            task["status"] = "reviewing"
            store.put(session, f"task:{task['id']}", task)
            agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                           content="任务报告：已完成", kind="review", message_id="retry")
            session.commit()
        await runtime.start()
        try:
            def processed():
                with db.new_session() as session:
                    return store.get(session, "bus:retry")["delivery"] == "processed"
            await wait_for(processed, timeout=8)
            with db.new_session() as session:
                records = store.records(session, "agent_context", context_id="main")
                assert sum(1 for r in records if r["id"] == "bus:retry") == 1
                assert store.get(session, f"task:{task['id']}")["status"] == "completed"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_invalid_routing_keeps_pending_approval_and_does_not_act(isolated):
    class InvalidModel(FakeModel):
        async def stream(self, *, system, messages, tools, tool_choice=None):
            if tool_choice:
                yield {"type": "tool_use", "id": "bad", "name": "route_input", "input": {"user_seq": -1}}
                return
            async for item in super().stream(system=system, messages=messages, tools=tools):
                yield item

    async def run():
        runtime = Runtime(client=InvalidModel())
        await runtime.start()
        try:
            def setup(session):
                task = runtime.create_task(session, "等待审批", 1)
                task["status"] = "waiting_user"
                store.put(session, f"task:{task['id']}", task)
                store.put(session, "approval:pending", {"type": "approval", "request_id": "pending",
                          "status": "requested", "task_id": task["id"]})
                return task
            task = await runtime.change(setup)
            await runtime.accept("invalid-route", "停止这个任务", None)
            def failed():
                with db.new_session() as session:
                    return store.get(session, "input:invalid-route")["status"] == "failed"
            await wait_for(failed)
            with db.new_session() as session:
                assert not store.get(session, f"task:{task['id']}")["stop_requested"]
                assert store.get(session, "approval:pending")["status"] == "requested"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_model_log_failure_does_not_block_chat(isolated, monkeypatch):
    async def run():
        not_directory = isolated / "not-a-directory"
        not_directory.write_text("x", encoding="utf-8")
        monkeypatch.setattr(settings, "modelio_dir", not_directory)
        runtime = Runtime(client=FakeModel())
        await runtime.start()
        try:
            await runtime.accept("logging", "简单问题", None)
            def answered():
                with db.new_session() as session:
                    return any(m["role"] == "assistant" and m["text"] == "收到。" for m in store.visible(session))
            await wait_for(answered)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_saved_output_pagination_and_task_scope(isolated):
    import re
    from app.agent.runtime import read_output

    with db.new_session() as session:
        store.initialize(session, isolated)
        content = "首" * 35000
        store.put(session, "output:one", {"type": "tool_output", "id": "one", "task_id": "A", "content": content})
        first = read_output(session, {"output_id": "one"}, task_id="A")
        assert len(first) <= 30000
        offset = int(re.search(r"offset=(\d+)", first).group(1))
        second = read_output(session, {"output_id": "one", "offset": offset}, task_id="A")
        assert first.split("\n[继续读取", 1)[0] + second == content
        assert read_output(session, {"output_id": "one"}, task_id="B").startswith("Error:")
