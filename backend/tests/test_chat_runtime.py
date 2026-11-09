import asyncio
import json
import socket
from copy import deepcopy

import pytest

from app import db
from app.agent.runtime import Conflict, Runtime
from app.config import settings
from app.models import new_id
from app.sessions import chat_store as store
from app.user_context import scoped


class FakeModel:
    model = "fake"

    def __init__(self):
        self.worker_started = asyncio.Event()
        self.release_worker = asyncio.Event()
        self.max_workers = 0
        self.workers = 0

    async def stream(self, *, system, messages, tools, tool_choice=None):
        if system.startswith("你是后台工作助手"):
            self.workers += 1
            self.max_workers = max(self.max_workers, self.workers)
            self.worker_started.set()
            try:
                await self.release_worker.wait()
                yield {"type": "text_delta", "text": "工作已完成"}
            finally:
                self.workers -= 1
        elif tool_choice and tool_choice["name"] == "task_review":
            note = next(m["content"] for m in messages if isinstance(m.get("content"), str)
                        and m["content"].startswith("运行状态（由系统提供，仅供参考）："))
            tasks = json.loads(note.split("当前任务：", 1)[1])
            task = next(t for t in tasks if t["status"] == "reviewing")
            yield {"type": "tool_use", "id": new_id(), "name": "task_review",
                   "input": {"task_id": task["id"], "decision": "complete"}}
        else:
            user_text = next((message["content"] for message in reversed(messages)
                              if message["role"] == "user" and isinstance(message["content"], str)
                              and not message["content"].startswith(("以下是用户长期记忆", "<此前对话摘要>",
                                                                      "运行状态（由系统提供，仅供参考）："))), "")
            created = any(block.get("name") == "task_create" and block.get("input", {}).get("goal") == user_text
                          for message in messages if isinstance(message.get("content"), list)
                          for block in message["content"] if isinstance(block, dict))
            if ("直接处理用户消息" in system and user_text.startswith("任务")
                    and not created):
                yield {"type": "tool_use", "id": new_id(), "name": "task_create",
                       "input": {"title": user_text[:40], "goal": user_text}}
            else:
                report_seen = "一轮结束" in json.dumps(messages, ensure_ascii=False)
                yield {"type": "text_delta", "text": "任务已完成。" if report_seen else "收到。"}
        yield {"type": "stop_reason", "reason": "end_turn"}


async def wait_for(check, timeout=4):
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.02)


async def accept(runtime, *args, **kwargs):
    return await runtime.accept(*args, assistant_id=(await runtime.snapshot())["assistant_id"], **kwargs)


async def decide(runtime, *args, **kwargs):
    return await runtime.decide(*args, assistant_id=(await runtime.snapshot())["assistant_id"], **kwargs)


async def checkpoint_history(runtime, context_id):
    from app.agent.context import records_from_messages
    async with runtime.graph.write_lock:
        thread = runtime.graph.thread_id(f"{context_id}:loop", parent_context=context_id)
    state = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
    return records_from_messages(state.values.get("messages", []))


def test_memory_is_available_only_to_main_agent(isolated):
    from app.agent.memory_store import _read_files, _store, empty_memory

    class Model(FakeModel):
        def __init__(self):
            super().__init__()
            self.main_memory = None
            self.worker_memory = None

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if not system.startswith("你是后台工作助手"):
                self.main_memory = (system, messages)
            elif system.startswith("你是后台工作助手"):
                self.worker_memory = (system, messages)
            async for item in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield item

    async def run():
        model = Model()
        runtime = Runtime(model)
        await runtime.start()
        try:
            summary = empty_memory()
            summary["user"]["personalContext"]["summary"] = "个人偏好标记"
            _store("a" * 32, summary, [])
            await accept(runtime, "memory-worker", "任务：执行测试", None)
            await asyncio.wait_for(model.worker_started.wait(), 4)
            assert "个人偏好标记" not in model.main_memory[0]
            assert "个人偏好标记" in json.dumps(model.main_memory[1], ensure_ascii=False)
            assert "个人偏好标记" not in json.dumps(model.worker_memory, ensure_ascii=False)
            summary, facts = _read_files("a" * 32)
            summary["user"]["personalContext"]["summary"] = "更新偏好标记"
            _store("a" * 32, summary, facts, summary["revision"])
            await accept(runtime, "memory-refresh", "简短问答", None)
            await wait_for(lambda: "更新偏好标记" in json.dumps(model.main_memory[1], ensure_ascii=False))
        finally:
            model.release_worker.set()
            await runtime.close()

    asyncio.run(run())


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from app.sandbox.lease import SandboxLeaseManager
    from fake_sandbox import FakeProvider

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    original_init = Runtime.__init__
    saver = InMemorySaver()
    def init(runtime, *args, **kwargs):
        if "sandbox_leases" not in kwargs:
            kwargs["sandbox_leases"] = SandboxLeaseManager(FakeProvider())
        if "checkpointer" not in kwargs:
            kwargs["checkpointer"] = saver
        original_init(runtime, *args, **kwargs)
    monkeypatch.setattr(Runtime, "__init__", init)
    (tmp_path / ("a" * 32)).mkdir()
    db.dispose_engine()
    db.init_db(tmp_path)
    yield tmp_path
    db.dispose_engine()


def test_skill_catalog_and_loading_in_main_and_worker(isolated, monkeypatch):
    from app.agent import skills

    directory = skills.user_skills_path("a" * 32) / "review-guide"
    (directory / "references").mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: review-guide\ndescription: 用于代码审查\n---\n说明正文标记", encoding="utf-8")
    (directory / "references" / "guide.md").write_text("参考资料标记", encoding="utf-8")

    class SkillModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.worker_calls = 0
            self.main_calls = 0

        async def stream(self, *, system, messages, tools, tool_choice=None):
            assert "review-guide" in system and "用于代码审查" in system
            assert "说明正文标记" not in system and "参考资料标记" not in system
            if system.startswith("你是后台工作助手"):
                self.worker_calls += 1
                history = json.dumps(messages, ensure_ascii=False)
                if self.worker_calls == 1:
                    assert "说明正文标记" not in history
                    yield {"type": "tool_use", "id": "skill", "name": "load_skill",
                           "input": {"name": "review-guide"}}
                elif self.worker_calls == 2:
                    assert "说明正文标记" in history and "参考资料标记" not in history
                    yield {"type": "tool_use", "id": "reference", "name": "load_skill",
                           "input": {"name": "review-guide", "path": "references/guide.md"}}
                else:
                    assert "参考资料标记" in history
                    yield {"type": "text_delta", "text": "已完成审查"}
                yield {"type": "stop_reason", "reason": "end_turn"}
                return
            self.main_calls += 1
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run():
        model = SkillModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "skill-test", "任务：审查代码", "build")
            def completed():
                with db.new_session() as session:
                    return any(task["status"] == "completed" for task in store.records(session, "task_state"))
            await wait_for(completed)
            assert model.worker_calls == 3 and model.main_calls >= 2
        finally:
            await runtime.close()
    asyncio.run(run())


def test_background_chat_and_idempotent_acceptance(isolated):
    async def run():
        model = FakeModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            first = await accept(runtime, "one", "任务：整理文件", "build")
            await asyncio.wait_for(model.worker_started.wait(), 3)
            queue, before = await runtime.subscribe()
            runtime.unsubscribe(queue)
            assert before["messages"][0]["text"] == "任务：整理文件"
            second = await accept(runtime, "two", "顺便问个问题", "build")
            assert first["user_seq"] < second["user_seq"]
            assert (await accept(runtime, "one", "任务：整理文件", "build"))["replayed"]
            with pytest.raises(Conflict):
                await accept(runtime, "one", "不同内容", "build")
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
                await decide(runtime, "old", "allowOnce")
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("conflict", [False, True])
def test_newer_approval_is_reconsidered_by_main_agent(isolated, conflict):
    class Model:
        model = "fake"

        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0
            self.task_id = ""

        async def stream(self, *, system, messages, tools, tool_choice=None):
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await self.release.wait()
                name = "task_stop" if conflict else "task_update"
                args = {"task_id": self.task_id}
                if not conflict:
                    args["instruction"] = "change unrelated file"
                yield {"type": "tool_use", "id": "first", "name": name, "input": args}
            elif self.calls == 2:
                assert messages[0]["content"].startswith("运行状态（由系统提供，仅供参考）：")
                assert "later-approval" in messages[0]["content"]
                if conflict:
                    yield {"type": "text_delta", "text": "newer decision applies"}
                else:
                    yield {"type": "tool_use", "id": "retry", "name": "task_update",
                           "input": {"task_id": self.task_id, "instruction": "change unrelated file"}}
            else:
                yield {"type": "text_delta", "text": "updated"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = Model()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            def setup(session):
                task = runtime.create_task(session, "existing task", 1)
                task.update(status="waiting_user", delivery_unknown=True)
                store.put(session, f"task:{task['id']}", task)
                return task
            task = await runtime.change(setup)
            model.task_id = task["id"]
            await accept(runtime, "older", "modify existing task", None)
            await asyncio.wait_for(model.started.wait(), 4)
            def approve(session):
                seq = store.bump(session, user_event=True)["user_seq"]
                store.put(session, "approval:later-approval", {"type": "approval", "request_id": "later-approval",
                          "task_id": task["id"], "status": "resolved", "option_id": "allowOnce", "user_seq": seq})
                store.put(session, "decision:later-approval", {"type": "user_event", "event_type": "approval",
                          "id": "later-approval", "task_id": task["id"], "user_seq": seq,
                          "option_id": "allowOnce", "status": "processed"})
            await runtime.change(approve)
            model.release.set()
            def finished():
                with db.new_session() as session:
                    return (store.get(session, "reply:older") or {}).get("status") == "completed"
            await wait_for(finished)
            with db.new_session() as session:
                current = store.get(session, f"task:{task['id']}")
                assert not current["stop_requested"]
                messages = store.records(session, "agent_message", context_id=task["context_id"])
                assert bool(messages) is not conflict
        finally:
            model.release.set()
            await runtime.close()
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


def test_progress_cannot_offer_review_until_final_report(isolated):
    from app.agent import agent_bus

    class ProgressModel(FakeModel):
        async def stream(self, *, system, messages, tools, tool_choice=None):
            if not tool_choice and "旧轮或已判断" in system:
                assert "task_review" not in {tool["name"] for tool in tools}
                yield {"type": "text_delta", "text": "进展已收到。"}
                return
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run():
        runtime = Runtime(client=ProgressModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "并行任务", 1)
            task.update(status="reviewing", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                           content="先前进展", kind="progress", message_id="progress", run_id="run")
            agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                           content="最终结果", kind="review", message_id="final", run_id="run")
            session.commit()
            progress = store.get(session, "bus:progress")
            final = store.get(session, "bus:final")
        await runtime._report(progress)
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}")["status"] == "reviewing"
        await runtime._report(final)
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}")["status"] == "completed"
            assert store.get(session, "bus:progress")["delivery"] == "processed"
            assert store.get(session, "bus:final")["delivery"] == "processed"
    asyncio.run(run())


@pytest.mark.parametrize("repeat", [False, True])
def test_unavailable_tool_is_not_executed_or_retried_forever(isolated, repeat):
    class ConfusedModel:
        model = "fake"

        def __init__(self):
            self.calls = 0

        async def stream(self, *, system, messages, tools, tool_choice=None):
            assert "task_review" not in {tool["name"] for tool in tools}
            self.calls += 1
            if self.calls == 1 or repeat:
                yield {"type": "tool_use", "id": f"bad-{self.calls}", "name": "task_review",
                       "input": {"task_id": task["id"], "decision": "complete"}}
            else:
                yield {"type": "text_delta", "text": "我会等待最终报告。"}

    async def run():
        nonlocal task
        model = ConfusedModel()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "处理任务", 1)
            task["status"] = "reviewing"
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        await runtime._dialog("请根据进展回复用户。")
        history = await checkpoint_history(runtime, "main")
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}")["status"] == "reviewing"
            assert not store.records(session, "agent_action")
            assert store.visible(session)[-1]["status"] == ("failed" if repeat else "completed")
        blocks = [block for row in history if isinstance(row["content"], list) for block in row["content"]]
        assert any(block.get("name") == "task_review" for block in blocks)
        assert any(block.get("type") == "tool_result" and block.get("is_error") for block in blocks)
        assert model.calls == 2
    task = None
    asyncio.run(run())


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
        from conftest import install_runtime
        install_runtime(app, runtime)
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
                sent = await client.post("/api/chat/messages", json={"assistant_id": (await runtime.snapshot())["assistant_id"], "client_message_id": "http-one", "text": "任务：后台执行"})
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


def test_approval_rejects_option_not_offered(isolated):
    async def run():
        runtime = Runtime(client=FakeModel())
        await runtime.start()
        try:
            def setup(session):
                task = runtime.create_task(session, "等待审批", 1)
                task.update(status="waiting_user", run_id="run")
                store.put(session, f"task:{task['id']}", task)
                store.put(session, "approval:limited", {
                    "type": "approval", "request_id": "limited", "task_id": task["id"],
                    "status": "requested", "scope": ["bash", "command"],
                    "options": [{"option_id": "allowOnce"}, {"option_id": "deny"}],
                })
            await runtime.change(setup)
            with pytest.raises(Conflict, match="审批选项不属于当前请求"):
                await decide(runtime, "limited", "allowSession")
            with db.new_session() as session:
                assert store.get(session, "approval:limited")["status"] == "requested"
            assert (await decide(runtime, "limited", "allowOnce"))["ok"]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_approval_stays_available_and_session_grant_is_shared(isolated):
    from app.agent.worker import Worker
    from app.sessions import session_permissions
    from fake_sandbox import FakeSandbox

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
        worker.sandbox = FakeSandbox()
        path = "/mnt/user-data/workspace/x"
        execution = {"id": "call1", "call_id": "c1", "name": "write_file", "input": {"path": path, "content": "a"}}
        waiting = asyncio.create_task(worker.approve(execution, "需要确认", sandboxed=True, canonical_path=path))
        try:
            await wait_for(lambda: bool(runtime.approval_events))
            request_id = next(iter(runtime.approval_events))
            message = await accept(runtime, "before-approval", "不要写入", None)
            assert (await runtime.snapshot())["approvals"][0]["request_id"] == request_id
            assert (await decide(runtime, request_id, "allowSession"))["ok"]
            assert (await waiting)["approved"]
            assert session_permissions.allows(scoped("main"), "write_file", {"path": path, "content": "b"}, canonical_path=path)
            with db.new_session() as session:
                approval = store.get(session, f"approval:{request_id}")
                assert approval["user_seq"] > message["user_seq"]
            with pytest.raises(Conflict):
                await decide(runtime, request_id, "deny")
        finally:
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
            session_permissions.forget_session(scoped("main"))
    asyncio.run(run())


def test_file_owner_requires_reread_before_other_task_writes(isolated):
    from app.agent.worker import Worker
    from app.sessions import execution_state
    from fake_sandbox import FakeSandbox

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
            worker.sandbox = FakeSandbox()

        async def call(worker, name, inputs):
            execution = {"id": new_id(), "call_id": new_id(), "name": name, "input": inputs, "status": "pending"}
            await runtime.change(lambda session: store.put(session, f"execution:{execution['id']}", {
                **execution, "type": "tool_execution"}))
            return await worker.file_tool(execution)

        assert "OK" in await call(workers[0], "write_file", {"path": "/mnt/user-data/workspace/x.txt", "content": "原始"})
        assert "原始" in await call(workers[1], "read_file", {"path": "/mnt/user-data/workspace/x.txt"})
        pending = asyncio.create_task(call(workers[1], "write_file", {"path": "/mnt/user-data/workspace/x.txt", "content": "覆盖"}))
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
        assert "原始" in await call(workers[1], "read_file", {"path": "/mnt/user-data/workspace/x.txt"})
        assert "OK" in await call(workers[1], "write_file", {"path": "/mnt/user-data/workspace/x.txt", "content": "覆盖"})
        assert (isolated / ("a" * 32) / "x.txt").read_text(encoding="utf-8") == "覆盖"
    asyncio.run(run())


def test_thirtieth_tool_step_requires_review(isolated, monkeypatch):
    from app.agent.worker import run_worker
    from app.agent import tools

    class ToolModel:
        model = "fake"
        async def stream(self, **_kwargs):
            yield {"type": "tool_use", "id": "last", "name": "glob", "input": {"pattern": "*.txt", "path": "/mnt/user-data/workspace"}}
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


def test_scheduler_caps_parallel_workers_at_six(isolated):
    async def run():
        model = FakeModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            def create(session):
                for number in range(7):
                    runtime.create_task(session, f"任务 {number}", number + 1)
            await runtime.change(create)
            await wait_for(lambda: model.workers == 6, timeout=12)
            await asyncio.sleep(0.4)
            with db.new_session() as session:
                states = [t["status"] for t in store.records(session, "task_state")]
            assert states.count("running") == 6 and states.count("queued") == 1
            assert model.max_workers == 6
            model.release_worker.set()
            def all_done():
                with db.new_session() as session:
                    return all(t["status"] == "completed" for t in store.records(session, "task_state"))
            await wait_for(all_done, timeout=20)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_helper_allocation_and_busy_message_delivery(isolated):
    from app.agent.worker import Worker

    class HelperModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.wrote_todo = False

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if system.startswith("你是后台工作助手") and not self.wrote_todo:
                self.wrote_todo = True
                yield {"type": "tool_use", "id": "helper-todo", "name": "write_todos", "input": {
                    "todos": [{"content": "协助事项", "status": "in_progress"}]}}
                yield {"type": "stop_reason", "reason": "end_turn"}
                return
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run():
        model = HelperModel()
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
            thread = runtime.graph.thread_id(f"{helper['context_id']}:loop", parent_context=helper["context_id"])
            state = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
            assert state.values["todos"] == [{"content": "协助事项", "status": "in_progress"}]
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


def test_compaction_keeps_latest_message_and_isolates_contexts(isolated, monkeypatch):
    from app.agent import compact
    from app.agent.context import messages_from_records

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 5)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            for i in range(5):
                store.stage_context(session, "ctx-a", "user", f"请求{i}" + "文" * 500)
                store.stage_context(session, "ctx-a", "assistant", f"答复{i}" + "字" * 500)
            store.stage_context(session, "ctx-a", "user", "最新请求")
            store.stage_context(session, "ctx-b", "user", "独立任务")
            session.commit()
            first = messages_from_records(store.records(session, "context_pending", context_id="ctx-a"))
            second = messages_from_records(store.records(session, "context_pending", context_id="ctx-b"))
        assemble = lambda state, _: ("system", compact.history_for_state(state), [])
        update = await compact.compact_state(runtime, {"messages": first}, {}, assemble)
        removed = {message.id for message in update["messages"]}
        projected = compact.history_for_state({"messages": [message for message in first if message.id not in removed],
                                               "summary_text": update["summary_text"]})
        assert "此前对话摘要" in projected[0]["content"]
        assert projected[-1]["content"] == "最新请求"
        assert len(removed) == 10
        assert second[0].content == "独立任务"
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
            store.stage_context(session, task["context_id"], "assistant", [{
                "type": "tool_use", "id": "c1", "name": "write_file", "input": {"path": "/mnt/user-data/workspace/x", "content": "x"}}],
                                 key=f"step:{task['id']}:run:1")
            store.put(session, f"execution:{execution_id}", {"type": "tool_execution", "id": execution_id,
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run", "call_id": "c1",
                      "name": "write_file", "input": {"path": "/mnt/user-data/workspace/x", "content": "x"}, "status": "pending"})
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
            await decide(runtime, new_request, "deny")
            def execution_finished():
                with db.new_session() as session:
                    return store.get(session, f"execution:{execution_id}")["status"] == "completed"
            await wait_for(execution_finished)
            assert not (isolated / ("a" * 32) / "x").exists()
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
            store.stage_context(session, task["context_id"], "assistant", [{
                "type": "tool_use", "id": "c1", "name": "write_file", "input": {"path": "/mnt/user-data/workspace/x", "content": "x"}}],
                                 key=f"step:{task['id']}:run:1")
            store.put(session, f"execution:{identity}", {"type": "tool_execution", "id": identity,
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run", "call_id": "c1",
                      "name": "write_file", "input": {"path": "/mnt/user-data/workspace/x", "content": "x"}, "status": "started"})
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
                execution = store.get(session, f"execution:{identity}")
                assert execution["result_unknown"] and "结果未知" in execution["output"]
            assert not (isolated / ("a" * 32) / "x").exists()
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
            await accept(runtime, "one", "第一个问题", None)
            await asyncio.wait_for(model.started.wait(), 3)
            await accept(runtime, "two", "第二个问题", None)
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


def test_restart_regenerates_unfinished_reply_without_recreating_task(isolated):
    class ReplyModel(FakeModel):
        def __init__(self, interrupt):
            super().__init__()
            self.interrupt = interrupt
            self.reply_started = asyncio.Event()
            self.replies = 0
            self.dialog_tools = set()
            self.created = False

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if system.startswith("你是后台工作助手"):
                async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                    yield event
            elif self.interrupt and not self.created:
                self.created = True
                yield {"type": "tool_use", "id": "create", "name": "task_create",
                       "input": {"title": "检查文件", "goal": "任务检查文件"}}
                yield {"type": "stop_reason", "reason": "end_turn"}
            else:
                self.replies += 1
                self.dialog_tools = {tool["name"] for tool in tools}
                if self.interrupt:
                    yield {"type": "text_delta", "text": "未完成的回复"}
                    self.reply_started.set()
                    await asyncio.Event().wait()
                else:
                    yield {"type": "text_delta", "text": "恢复后的回复"}
                    yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        first = ReplyModel(interrupt=True)
        runtime = Runtime(client=first)
        await runtime.start()
        try:
            await accept(runtime, "recover", "任务检查文件", None)
            await asyncio.wait_for(first.reply_started.wait(), 4)
            with db.new_session() as session:
                assert store.get(session, "reply:recover")["status"] == "pending"
                assert len(store.records(session, "task_state")) == 1
        finally:
            await runtime.close()

        second = ReplyModel(interrupt=False)
        runtime = Runtime(client=second)
        await runtime.start()
        try:
            def completed():
                with db.new_session() as session:
                    return store.get(session, "reply:recover")["status"] == "completed"
            await wait_for(completed)
            with db.new_session() as session:
                assert len(store.records(session, "task_state")) == 1
                visible = store.visible(session)
                assert any(m["text"] == "未完成的回复" and m["status"] == "interrupted" for m in visible)
                assert sum(m["text"] == "恢复后的回复" for m in visible) == 1
            assert second.replies == 1 and "task_create" in second.dialog_tools
        finally:
            await runtime.close()

        third = ReplyModel(interrupt=False)
        runtime = Runtime(client=third)
        await runtime.start()
        try:
            await asyncio.sleep(0.3)
            assert third.replies == 0
        finally:
            await runtime.close()
    asyncio.run(run())


def test_pending_reply_survives_missing_model_configuration(isolated):
    async def run():
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "你好")
            store.put(session, "reply:pending", {"type": "main_reply", "id": "pending", "source_seq": 1,
                      "note": "继续回答", "status": "pending"})
            session.commit()
        runtime = Runtime(client=FakeModel())
        runtime.client = None
        await runtime.start()
        try:
            await asyncio.sleep(0.3)
            with db.new_session() as session:
                assert store.get(session, "reply:pending")["status"] == "pending"
                assert any("模型配置不完整" in message["text"] for message in store.visible(session))
        finally:
            await runtime.close()
        runtime = Runtime(client=FakeModel())
        await runtime.start()
        try:
            def completed():
                with db.new_session() as session:
                    return store.get(session, "reply:pending")["status"] == "completed"
            await wait_for(completed)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_agent_wait_delivers_messages_before_and_after_registration(isolated):
    from app.agent import agent_bus
    from app.agent.worker import AgentWaiting, Worker

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
        with pytest.raises(AgentWaiting):
            await worker.wait_for_agent(b["context_id"], "wait-two")
        with db.new_session() as session:
            assert store.get(session, "agent-wait:wait-two")["status"] == "waiting"
            assert store.get(session, f"task:{a['id']}")["status"] == "waiting_agent"
        await runtime.change(lambda session: agent_bus.send(session, source=b["context_id"], target=a["context_id"],
                             task_id=b["id"], content="随后回复"))
        assert await worker.wait_for_agent(b["context_id"], "wait-two") == "随后回复"
    asyncio.run(run())


def test_waiting_agents_release_slots_and_resume_after_restart(isolated):
    class WaitModel(FakeModel):
        def __init__(self, contexts):
            super().__init__()
            self.contexts = contexts
            self.d_started = asyncio.Event()
            self.release_d = asyncio.Event()
            self.waiters_started = 0
            self.all_started = asyncio.Event()
            self.release_waiters = asyncio.Event()

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if not system.startswith("你是后台工作助手"):
                async for part in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                    yield part
                return
            goal = next(m["content"].split("：", 1)[1] for m in messages
                        if isinstance(m["content"], str) and m["content"].startswith("任务目标："))
            results = any(isinstance(m["content"], list) and any(b.get("type") == "tool_result" for b in m["content"])
                          for m in messages)
            if goal in {"A", "B", "C"} and not results:
                self.waiters_started += 1
                if self.waiters_started == 3:
                    self.all_started.set()
                await self.release_waiters.wait()
                yield {"type": "tool_use", "id": "todo", "name": "write_todos",
                       "input": {"todos": [{"content": f"{goal} 的工作", "status": "in_progress"}]}}
                yield {"type": "tool_use", "id": "wait", "name": "agent_wait",
                       "input": {"context_id": self.contexts["D"]}}
            elif goal == "D" and not results:
                self.d_started.set()
                await self.release_d.wait()
                for name in ("A", "B", "C"):
                    yield {"type": "tool_use", "id": f"send-{name}", "name": "agent_send",
                           "input": {"context_id": self.contexts[name], "content": f"给 {name} 的回复"}}
            elif goal == "D":
                yield {"type": "text_delta", "text": "D 已完成"}
            else:
                await asyncio.Event().wait()
            yield {"type": "stop_reason", "reason": "tool_use"}

    async def run():
        with db.new_session() as session:
            store.initialize(session, isolated)
            setup = Runtime(client=FakeModel())
            tasks = {name: setup.create_task(session, name, index) for index, name in enumerate("ABCD", 1)}
            session.commit()
        contexts = {name: task["context_id"] for name, task in tasks.items()}
        first = WaitModel(contexts)
        runtime = Runtime(client=first)
        await runtime.start()
        try:
            await asyncio.wait_for(first.all_started.wait(), 8)
            first.release_waiters.set()
            await asyncio.wait_for(first.d_started.wait(), 8)
            def all_waiting():
                with db.new_session() as session:
                    return all(store.get(session, f"task:{tasks[name]['id']}")["status"] == "waiting_agent" for name in "ABC")
            await wait_for(all_waiting)
            with db.new_session() as session:
                from app.agent import shared_state
                assert all(store.get(session, f"task:{tasks[name]['id']}")["status"] == "waiting_agent" for name in "ABC")
                runs = {name: store.get(session, f"task:{tasks[name]['id']}")["run_id"] for name in "ABC"}
                assert len([w for w in store.records(session, "agent_wait") if w["status"] == "waiting"]) == 3
                assert {(item["todo"], item["status"]) for item in shared_state.snapshot(session)} == {
                    (f"{name} 的工作", "waiting_agent") for name in "ABC"}
        finally:
            await runtime.close()
        second = WaitModel(contexts)
        runtime = Runtime(client=second)
        await runtime.start()
        try:
            await asyncio.wait_for(second.d_started.wait(), 8)
            with db.new_session() as session:
                assert {(item["todo"], item["status"]) for item in shared_state.snapshot(session)} == {
                    (f"{name} 的工作", "waiting_agent") for name in "ABC"}
            second.release_d.set()
            def resumed():
                with db.new_session() as session:
                    waits = store.records(session, "agent_wait")
                    return len(waits) == 3 and all(w["status"] == "completed" for w in waits)
            await wait_for(resumed, timeout=8)
            with db.new_session() as session:
                for name in "ABC":
                    task = store.get(session, f"task:{tasks[name]['id']}")
                    assert task["run_id"] == runs[name]
            for name in "ABC":
                async with asyncio.timeout(4):
                    while True:
                        history = await checkpoint_history(runtime, contexts[name])
                        results = [block for message in history if isinstance(message["content"], list)
                                   for block in message["content"] if block.get("type") == "tool_result"]
                        if any(f"给 {name} 的回复" in block["content"] for block in results):
                            break
                        await asyncio.sleep(0.02)
                replies = [block for block in results if block.get("tool_use_id") == "wait"]
                assert len(replies) == 1 and f"给 {name} 的回复" in replies[0]["content"]
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("helper", [False, True])
def test_wait_target_inherits_earliest_task_order(isolated, helper):
    from app.agent.worker import AgentWaiting, Worker

    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "A", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            reviewing = runtime.create_task(session, "等待判断", 2)
            reviewing["status"] = "reviewing"
            store.put(session, f"task:{reviewing['id']}", reviewing)
            queued = runtime.create_task(session, "其他任务", 3)
            if helper:
                target_context = "priority-helper"
                store.put(session, f"helper:{target_context}", {"type": "agent_run", "task_id": task["id"],
                          "context_id": target_context, "run_id": "helper-run", "caller": task["context_id"],
                          "status": "queued", "steps": 0})
                store.stage_context(session, target_context, "user", "协助任务：完成 A")
            else:
                target = runtime.create_task(session, "依赖任务", 4)
                target_context = target["context_id"]
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        with pytest.raises(AgentWaiting):
            await worker.wait_for_agent(target_context, "wait-priority")
        await runtime._schedule_once()
        try:
            await asyncio.wait_for(runtime.client.worker_started.wait(), 4)
            with db.new_session() as session:
                assert store.get(session, f"task:{queued['id']}")["status"] == "queued"
                if helper:
                    assert store.get(session, f"helper:{target_context}")["status"] == "running"
                else:
                    selected = store.get(session, f"task:{target['id']}")
                    assert selected["status"] == "running" and selected["order"] == 1
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("helper", [False, True])
@pytest.mark.parametrize("ready", [False, True])
def test_stopping_a_waiting_agent_completes_its_tool_call(isolated, helper, ready):
    from app.agent import agent_bus
    from app.agent.worker import AgentWaiting, Worker

    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "A", 1)
            target = runtime.create_task(session, "B", 2)
            task.update(status="running", run_id="run")
            target["status"] = "running"
            store.put(session, f"task:{task['id']}", task)
            store.put(session, f"task:{target['id']}", target)
            if ready:
                reviewing = runtime.create_task(session, "较早任务", 0)
                reviewing["status"] = "reviewing"
                store.put(session, f"task:{reviewing['id']}", reviewing)
                runtime.create_task(session, "较晚排队任务", 99)
            context = "helper-context" if helper else task["context_id"]
            if helper:
                store.put(session, f"helper:{context}", {"type": "agent_run", "task_id": task["id"],
                          "context_id": context, "run_id": "run", "caller": task["context_id"],
                          "status": "running", "steps": 1})
            call = {"type": "tool_use", "id": "call", "name": "agent_wait", "input": {"context_id": target["context_id"]}}
            store.stage_context(session, context, "assistant", [call], key="step:wait")
            store.put(session, "execution:wait", {"type": "tool_execution", "id": "wait", "call_id": "call",
                      "task_id": task["id"], "context_id": context, "run_id": "run", "name": "agent_wait",
                      "input": call["input"], "status": "pending"})
            store.put(session, "group:wait", {"type": "tool_group", "id": "wait", "context_id": context,
                      "calls": ["wait"], "status": "pending"})
            session.commit()
        worker = Worker(runtime, task["id"], context if helper else None)
        worker.context_id, worker.run_id = context, "run"
        with pytest.raises(AgentWaiting):
            await worker.process_group("wait")
        if ready:
            await runtime.change(lambda session: agent_bus.send(session, source=target["context_id"], target=context,
                                 task_id=target["id"], content="回复已到", message_id="reply-before-stop"))
            with db.new_session() as session:
                tasks, helpers = store.records(session, "task_state"), store.records(session, "agent_run")
                changed, _ = runtime._resume_agent_waits(session, tasks, helpers)
                assert changed
                assert store.get(session, "agent-wait:wait")["status"] == "ready"
                session.commit()
        await runtime.change(lambda session: runtime._main_action(session, "task_stop", {"task_id": task["id"]}, None, "stop"))
        if ready and not helper:
            from app.agent.runtime import _task_context
            with db.new_session() as session:
                summary, _ = _task_context(session, task["id"])
                assert summary[0]["stop_requested"]
        await runtime._schedule_once()
        try:
            def stopped():
                with db.new_session() as session:
                    key = f"helper:{context}" if helper else f"task:{task['id']}"
                    return store.get(session, key)["status"] == "stopped"
            await wait_for(stopped)
            with db.new_session() as session:
                assert store.get(session, "agent-wait:wait")["status"] == "completed"
                assert store.get(session, "group:wait")["status"] == "completed"
                result = store.get(session, "execution:wait")["output"]
                assert result.startswith("Error:") and "停止" in result
        finally:
            await runtime.close()
    asyncio.run(run())


def test_waiting_agent_resumes_on_user_update_and_detects_wait_cycle(isolated):
    from app.agent import agent_bus
    from app.agent.worker import AgentWaiting, Worker

    async def run():
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            a, b = runtime.create_task(session, "A", 1), runtime.create_task(session, "B", 2)
            for task in (a, b):
                task.update(status="running", run_id="run")
                store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:wait-a", {"type": "tool_execution", "id": "wait-a", "call_id": "call-a",
                      "task_id": a["id"], "context_id": a["context_id"], "run_id": "run", "name": "agent_wait",
                      "input": {"context_id": b["context_id"]}, "status": "pending"})
            store.put(session, "group:wait-a", {"type": "tool_group", "id": "wait-a", "context_id": a["context_id"],
                      "calls": ["wait-a"], "status": "pending"})
            session.commit()
        wa, wb = Worker(runtime, a["id"]), Worker(runtime, b["id"])
        wa.context_id, wa.run_id = a["context_id"], "run"
        wb.context_id, wb.run_id = b["context_id"], "run"
        with pytest.raises(AgentWaiting):
            await wa.process_group("wait-a")
        with pytest.raises(AgentWaiting):
            await wb.wait_for_agent(a["context_id"], "wait-b")
        with db.new_session() as session:
            assert any(m["kind"] == "cycle" for m in store.records(session, "agent_message", context_id="main"))
        await runtime.change(lambda session: agent_bus.send(session, source="main", target=a["context_id"],
                             task_id=a["id"], content="修改后的要求", message_id="update"))
        await runtime._schedule_once()
        try:
            await asyncio.wait_for(runtime.client.worker_started.wait(), 4)
            history = await checkpoint_history(runtime, a["context_id"])
            result = next(block for message in history if isinstance(message["content"], list)
                          for block in message["content"] if block.get("type") == "tool_result")
            assert result["is_error"] and "新要求" in result["content"]
            assert any(message["content"] == "修改后的要求" for message in history)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_report_delivery_failure_retries_without_duplicate_context(isolated, monkeypatch):
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
        async def keep_records():
            pass
        monkeypatch.setattr(runtime, "_cleanup_finished", keep_records)
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
            records = await checkpoint_history(runtime, "main")
            with db.new_session() as session:
                assert sum(1 for r in records if r["id"] == "bus:retry") == 1
                assert store.get(session, f"task:{task['id']}")["status"] == "completed"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_invalid_main_tool_keeps_pending_approval_and_does_not_act(isolated):
    class InvalidModel:
        model = "fake"

        def __init__(self):
            self.calls = 0

        async def stream(self, *, system, messages, tools, tool_choice=None):
            self.calls += 1
            if self.calls == 1:
                yield {"type": "tool_use", "id": "bad", "name": "task_stop", "input": {"task_id": 1}}
            else:
                yield {"type": "text_delta", "text": "invalid action"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        runtime = Runtime(client=InvalidModel())
        await runtime.start()
        try:
            def setup(session):
                task = runtime.create_task(session, "waiting for approval", 1)
                task.update(status="waiting_user", delivery_unknown=True)
                store.put(session, f"task:{task['id']}", task)
                store.put(session, "approval:pending", {"type": "approval", "request_id": "pending",
                          "status": "requested", "task_id": task["id"]})
                return task
            task = await runtime.change(setup)
            await accept(runtime, "invalid-action", "stop this task", None)
            def finished():
                with db.new_session() as session:
                    return (store.get(session, "reply:invalid-action") or {}).get("status") == "completed"
            await wait_for(finished)
            with db.new_session() as session:
                assert store.get(session, "approval:pending")["status"] == "requested"
                assert not store.get(session, f"task:{task['id']}")["stop_requested"]
            history = await checkpoint_history(runtime, "main")
            results = [block for message in history if isinstance(message["content"], list)
                       for block in message["content"] if block.get("type") == "tool_result"]
            assert results[0].get("is_error")
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
            await accept(runtime, "logging", "简单问题", None)
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


def test_task_context_is_bounded_and_old_details_remain_searchable(isolated):
    import re
    from app.agent import agent_bus
    from app.agent.runtime import _TASK_CONTEXT_LIMIT, _task_context
    from app.models import Part

    runtime = Runtime(client=FakeModel())
    with db.new_session() as session:
        store.initialize(session, isolated)
        for index in range(70):
            task = runtime.create_task(session, f"进行中 {index} " + "长" * 700, index + 1)
            task.update(status="running", files=["路径" * 150] * 4)
            store.put(session, f"task:{task['id']}", task)
        old = runtime.create_task(session, "远古枫叶 " + "原" * 13000, 100)
        old["status"] = "completed"
        store.put(session, f"task:{old['id']}", old)
        for index in range(3):
            agent_bus.send(session, source=old["context_id"], target="main", task_id=old["id"],
                           kind="review", content=f"较早报告 {index}", message_id=f"older-report-{index}")
        agent_bus.send(session, source=old["context_id"], target="main", task_id=old["id"],
                       kind="review", content="历史报告" + "结" * 12000, message_id="old-report")
        for index in range(12):
            task = runtime.create_task(session, f"近期关键词 {index}", 101 + index)
            task["status"] = "completed"
            store.put(session, f"task:{task['id']}", task)
            session.get(Part, scoped(f"task:{task['id']}")).time_updated = 1000 + index
        session.get(Part, scoped(f"task:{old['id']}")).time_updated = 1
        session.commit()
        summaries, omitted = _task_context(session)
        assert omitted and len(store.dump(summaries)) <= _TASK_CONTEXT_LIMIT
        assert any(t["status"] == "running" for t in summaries)
        assert any(t["status"] == "completed" for t in summaries)
        assert old["id"] not in {t["id"] for t in summaries}
        priority, _ = _task_context(session, old["id"])
        assert priority[0]["id"] == old["id"]

        search = json.loads(runtime._main_action(session, "task_search", {"keyword": "枫叶"}, None, "search"))
        assert [t["id"] for t in search["tasks"]] == [old["id"]]
        first = json.loads(runtime._main_action(session, "task_search", {"keyword": "关键词"}, None, "search"))
        second = json.loads(runtime._main_action(session, "task_search", {"keyword": "关键词", "offset": first["next_offset"]}, None, "search"))
        assert len(first["tasks"]) == 10 and len(second["tasks"]) == 2 and second["next_offset"] is None
        assert not ({t["id"] for t in first["tasks"]} & {t["id"] for t in second["tasks"]})
        def read_details(report_offset):
            chunks, offset = [], 0
            while True:
                part = runtime._main_action(session, "task_inspect", {
                    "task_id": old["id"], "offset": offset, "report_offset": report_offset}, None, "inspect")
                assert len(part) <= 8000
                match = re.search(r"\n\[继续读取：offset=(\d+)\]$", part)
                chunks.append(part[:match.start()] if match else part)
                if not match:
                    return json.loads("".join(chunks))
                offset = int(match.group(1))
        details = read_details(0)
        assert details["goal"] == old["goal"]
        assert details["reports"][0]["content"] == "历史报告" + "结" * 12000
        assert details["reports_next_offset"] == 3
        assert read_details(3)["reports"][0]["content"] == "较早报告 0"
        assert runtime._main_action(session, "task_search", {"keyword": "枫叶", "offset": -1}, None, "search").startswith("Error:")
        assert runtime._main_action(session, "task_search", {"keyword": "枫叶", "offset": 10**100}, None, "search").startswith("Error:")
        assert runtime._main_action(session, "task_inspect", {"task_id": old["id"], "report_offset": True}, None, "inspect").startswith("Error:")


def test_old_task_can_be_found_from_main_chat(isolated):
    from app.agent import agent_bus
    from app.models import Part

    class SearchModel(FakeModel):
        async def stream(self, *, system, messages, tools, tool_choice=None):
            blocks = [block for message in messages if isinstance(message["content"], list)
                      for block in message["content"]]
            names = [block.get("name") for block in blocks if block.get("type") == "tool_use"]
            if "task_search" not in names:
                yield {"type": "tool_use", "id": "search", "name": "task_search", "input": {"keyword": "枫叶"}}
            elif "task_inspect" not in names:
                output = next(block["content"] for block in blocks if block.get("tool_use_id") == "search")
                task_id = json.loads(output)["tasks"][0]["id"]
                yield {"type": "tool_use", "id": "inspect", "name": "task_inspect", "input": {"task_id": task_id}}
            else:
                output = next(block["content"] for block in blocks if block.get("tool_use_id") == "inspect")
                assert "历史报告：已经完成" in output
                yield {"type": "text_delta", "text": "枫叶任务已经完成。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        runtime = Runtime(client=SearchModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            old = runtime.create_task(session, "远古枫叶", 1)
            old["status"] = "completed"
            store.put(session, f"task:{old['id']}", old)
            agent_bus.send(session, source=old["context_id"], target="main", task_id=old["id"],
                           kind="review", content="历史报告：已经完成", message_id="old-report")
            report = store.get(session, "bus:old-report")
            agent_bus.processed(session, report)
            for index in range(7):
                task = runtime.create_task(session, f"近期任务 {index}", index + 2)
                task["status"] = "completed"
                store.put(session, f"task:{task['id']}", task)
                session.get(Part, scoped(f"task:{task['id']}")).time_updated = 1000 + index
            session.get(Part, scoped(f"task:{old['id']}")).time_updated = 1
            session.commit()
        await runtime.start()
        try:
            await accept(runtime, "old-question", "以前枫叶任务结果是什么？", None)
            def answered():
                with db.new_session() as session:
                    return any(m["text"] == "枫叶任务已经完成。" for m in store.visible(session))
            await wait_for(answered)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_omitted_active_task_can_be_found_and_stopped(isolated):
    class SearchAndStop(FakeModel):
        async def stream(self, *, system, messages, tools, tool_choice=None):
            if system.startswith("你是后台工作助手"):
                async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                    yield event
                return
            else:
                blocks = [block for message in messages if isinstance(message["content"], list)
                          for block in message["content"]]
                names = [block.get("name") for block in blocks if block.get("type") == "tool_use"]
                if "task_search" not in names:
                    yield {"type": "tool_use", "id": "search", "name": "task_search", "input": {"keyword": "特殊目标五十五"}}
                elif "task_stop" not in names:
                    output = next(block["content"] for block in blocks if block.get("tool_use_id") == "search")
                    task_id = json.loads(output)["tasks"][0]["id"]
                    yield {"type": "tool_use", "id": "stop", "name": "task_stop", "input": {"task_id": task_id}}
                else:
                    yield {"type": "text_delta", "text": "已停止指定任务。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        runtime = Runtime(client=SearchAndStop())
        with db.new_session() as session:
            store.initialize(session, isolated)
            for index in range(54):
                runtime.create_task(session, f"普通任务 {index}", index + 1)
            target = runtime.create_task(session, "特殊目标五十五", 55)
            session.commit()
        await runtime.start()
        try:
            await accept(runtime, "stop-omitted", "停止特殊目标五十五", None)
            def stopped():
                with db.new_session() as session:
                    return store.get(session, f"task:{target['id']}")["status"] == "stopped"
            await wait_for(stopped, timeout=8)
            with db.new_session() as session:
                assert sum(task["status"] == "stopped" for task in store.records(session, "task_state")) == 1
        finally:
            await runtime.close()
    asyncio.run(run())


def test_worker_rejects_host_commands(isolated, monkeypatch):
    import subprocess
    from app.agent import hooks
    from app.agent.worker import Worker

    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: calls.append(True))
    states = []
    monkeypatch.setattr(hooks, "trigger", lambda event, data: states.append(data["status"]) if event == "PostToolUse" else None)

    async def run():
        runtime = Runtime(client=FakeModel(), sandbox_leases=None)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "测试命令", 1)
            task.update(status="running", run_id="run", steps=1)
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:one", {"type": "tool_execution", "id": "one", "call_id": "call",
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run",
                      "name": "bash", "input": {"command": "ignored"}, "status": "pending"})
            store.put(session, "group:one", {"type": "tool_group", "id": "one", "context_id": task["context_id"],
                      "calls": ["one"], "status": "pending"})
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        async def allow(_execution):
            return None
        monkeypatch.setattr(worker, "authorize", allow)
        await worker.process_group("one")
        with db.new_session() as session:
            result = store.get(session, "execution:one")["output"]
        assert result.startswith("Error:") and "禁止宿主命令" in result
        assert states == ["error"] and not calls
    asyncio.run(run())


class DirectModel:
    model = "fake"

    def __init__(self, tool, args):
        self.tool, self.args = tool, args
        self.issued = False
        self.worker_calls = 0
        self.reports = []
        self.direct_tools = set()
        self.updated = False

    async def stream(self, *, system, messages, tools, tool_choice=None):
        if system.startswith("你是后台工作助手"):
            self.worker_calls += 1
            raise AssertionError("简单任务不应启动后台助手模型")
        elif not self.updated and any(message.get("role") == "user" and message.get("content") == "不要写入"
                                      for message in messages):
            self.updated = True
            with db.new_session() as session:
                task = next(t for t in store.records(session, "task_state") if t.get("direct"))
            yield {"type": "tool_use", "id": new_id(), "name": "task_update",
                   "input": {"task_id": task["id"], "instruction": "不要写入"}}
        elif "请根据直接执行" in system:
            report = next(m["content"] for m in reversed(messages)
                          if isinstance(m.get("content"), str) and "直接执行【" in m["content"])
            self.reports.append(report)
            yield {"type": "text_delta", "text": "已收到直接执行结果。"}
        elif not self.issued:
            self.issued = True
            self.direct_tools = {tool["name"] for tool in tools}
            yield {"type": "tool_use", "id": "direct-call", "name": self.tool, "input": self.args}
        else:
            yield {"type": "text_delta", "text": "已处理新消息。"}
        yield {"type": "stop_reason", "reason": "end_turn"}


def test_quick_file_read_runs_without_worker_model(isolated):
    (isolated / ("a" * 32) / "note.txt").write_text("workspace", encoding="utf-8")

    async def run():
        model = DirectModel("read_file", {"path": "/mnt/user-data/workspace/note.txt"})
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "read", "读取 note.txt", "build")
            await wait_for(lambda: bool(model.reports))
            assert model.worker_calls == 0 and "read_file" in model.direct_tools and "bash" in model.direct_tools
            assert "workspace" in model.reports[0]
            def completed():
                with db.new_session() as session:
                    tasks = store.records(session, "task_state")
                    return len(tasks) == 1 and tasks[0]["direct"] and tasks[0]["status"] == "completed"
            await wait_for(completed)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_direct_read_runs_while_worker_slots_are_full(isolated):
    (isolated / ("a" * 32) / "note.txt").write_text("workspace", encoding="utf-8")

    async def run():
        background = FakeModel()
        runtime = Runtime(client=background)
        await runtime.start()
        try:
            for index in range(6):
                await accept(runtime, f"task-{index}", f"任务 {index}", "build")
                await wait_for(lambda: background.workers == index + 1)
            model = DirectModel("read_file", {"path": "/mnt/user-data/workspace/note.txt"})
            runtime.client = model
            await accept(runtime, "read", "读取 note.txt", "build")
            await wait_for(lambda: bool(model.reports))
            assert background.workers == 6 and model.worker_calls == 0
        finally:
            background.release_worker.set()
            await runtime.close()
    asyncio.run(run())


def test_direct_steps_preserve_file_version_check(isolated):
    class MultiStepModel(DirectModel):
        def __init__(self):
            super().__init__("read_file", {"path": "/mnt/user-data/workspace/note.txt"})
            self.read_done, self.continue_edit = asyncio.Event(), asyncio.Event()

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if "请根据直接执行" in system and not self.reports:
                self.reports.append("已读取")
                self.read_done.set()
                await self.continue_edit.wait()
                yield {"type": "tool_use", "id": "edit-call", "name": "str_replace",
                       "input": {"path": "/mnt/user-data/workspace/note.txt", "old_str": "old", "new_str": "edited"}}
                yield {"type": "stop_reason", "reason": "tool_use"}
                return
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run():
        path = isolated / ("a" * 32) / "note.txt"
        path.write_text("old", encoding="utf-8")
        model = MultiStepModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "edit", "读取并修改文件", "build")
            await asyncio.wait_for(model.read_done.wait(), 3)
            path.write_text("old changed", encoding="utf-8")
            model.continue_edit.set()
            await wait_for(lambda: bool(runtime.approval_events))
            await decide(runtime, next(iter(runtime.approval_events)), "allowOnce")
            await wait_for(lambda: len(model.reports) == 2)
            assert "请先读取文件的当前版本" in model.reports[-1]
            assert path.read_text(encoding="utf-8") == "old changed"
            with db.new_session() as session:
                assert len(store.records(session, "task_state")) == 1
            assert model.worker_calls == 0
        finally:
            model.continue_edit.set()
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("option", ["allowOnce", "deny"])
def test_direct_approval_does_not_block_chat(isolated, option):
    class RecordingDirectModel(DirectModel):
        def __init__(self):
            super().__init__("write_file", {"path": "/mnt/user-data/workspace/quick.txt", "content": "内容"})
            self.requests = []

        async def stream(self, *, system, messages, tools, tool_choice=None):
            self.requests.append(deepcopy(messages))
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run():
        model = RecordingDirectModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "write", "创建文件", "build")
            await wait_for(lambda: bool(runtime.approval_events))
            request_id = next(iter(runtime.approval_events))
            assert not (isolated / ("a" * 32) / "quick.txt").exists()
            await accept(runtime, "question", "另一个问题", "build")
            def answered():
                with db.new_session() as session:
                    return any(m["text"] == "已处理新消息。" for m in store.visible(session))
            await wait_for(answered)
            assert request_id in runtime.approval_events
            question = next(messages for messages in model.requests if messages[-1].get("content") == "另一个问题")
            assert question[0]["content"].startswith("运行状态（由系统提供，仅供参考）：")
            assert request_id in question[0]["content"] and '"status":"requested"' in question[0]["content"]
            await decide(runtime, request_id, option)
            await wait_for(lambda: bool(model.reports))
            await wait_for(lambda: runtime.main_call is None)
            assert (isolated / ("a" * 32) / "quick.txt").exists() == (option == "allowOnce")
            assert model.worker_calls == 0
            assert model.reports[0].startswith("内部消息（非用户直接输入）：")
            history = await checkpoint_history(runtime, "main")
            assert "运行状态（由系统提供，仅供参考）：" not in json.dumps(history, ensure_ascii=False)
            report = next(row for row in history if row["id"].startswith("bus:direct:"))
            assert report["content"].startswith("直接执行【")
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("update_first", [True, False])
def test_direct_operation_respects_update_and_approval_order(isolated, monkeypatch, update_first):
    from app.agent.worker import Worker

    async def run():
        model = DirectModel("write_file", {"path": "/mnt/user-data/workspace/ordered.txt", "content": "内容"})
        runtime = Runtime(client=model)
        reached_start, release_start = asyncio.Event(), asyncio.Event()
        original_started = Worker.started

        async def pause_before_start(worker, execution):
            reached_start.set()
            await release_start.wait()
            return await original_started(worker, execution)

        monkeypatch.setattr(Worker, "started", pause_before_start)
        await runtime.start()
        try:
            await accept(runtime, "write", "创建文件", "build")
            await wait_for(lambda: bool(runtime.approval_events))
            request_id = next(iter(runtime.approval_events))
            if not update_first:
                await decide(runtime, request_id, "allowOnce")
                await asyncio.wait_for(reached_start.wait(), 3)
            await accept(runtime, "update", "不要写入", "build")
            def update_delivered():
                with db.new_session() as session:
                    return any(m.get("source") == "main" and m.get("delivery") == "pending"
                               for m in store.records(session, "agent_message"))
            await wait_for(update_delivered)
            if update_first:
                await decide(runtime, request_id, "allowOnce")
                await asyncio.wait_for(reached_start.wait(), 3)
            release_start.set()
            await wait_for(lambda: bool(model.reports))
            assert (isolated / ("a" * 32) / "ordered.txt").exists() == update_first
            if not update_first:
                assert "收到新的用户要求" in model.reports[0]
            assert model.worker_calls == 0
        finally:
            release_start.set()
            await runtime.close()
    asyncio.run(run())


def test_direct_started_tool_is_not_replayed_after_restart(isolated, monkeypatch):
    from app.agent import tools

    calls = []
    monkeypatch.setattr(tools, "execute_tool", lambda *args: calls.append(args) or "unexpected")

    async def run():
        model = DirectModel("read_file", {"path": "/mnt/user-data/workspace/note.txt"})
        model.issued = True
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "pwd", 1)
            group = f"{task['id']}:run:group"
            identity = f"{task['id']}:run:call"
            task.update(status="running", run_id="run", direct=True, direct_phase="executing", direct_group=group)
            store.put(session, f"task:{task['id']}", task)
            store.stage_context(session, task["context_id"], "assistant", [{
                "type": "tool_use", "id": "call", "name": "read_file", "input": {"path": "/mnt/user-data/workspace/note.txt"}}],
                                 key=f"step:{group}")
            store.put(session, f"execution:{identity}", {"type": "tool_execution", "id": identity,
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run", "call_id": "call",
                      "name": "read_file", "input": {"path": "/mnt/user-data/workspace/note.txt"}, "status": "started"})
            store.put(session, f"group:{group}", {"type": "tool_group", "id": group,
                      "context_id": task["context_id"], "calls": [identity], "status": "pending"})
            session.commit()
        await runtime.start()
        try:
            await wait_for(lambda: bool(model.reports))
            assert not calls and "结果未知" in model.reports[0]
            assert model.worker_calls == 0
        finally:
            await runtime.close()
    asyncio.run(run())


def test_direct_pending_approval_is_reissued_after_restart(isolated):
    async def run():
        model = DirectModel("write_file", {"path": "/mnt/user-data/workspace/recovered.txt", "content": "ok"})
        model.issued = True
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "写入文件", 1)
            group, identity = f"{task['id']}:run:group", f"{task['id']}:run:call"
            task.update(status="waiting_user", run_id="run", direct=True, direct_phase="executing", direct_group=group)
            store.put(session, f"task:{task['id']}", task)
            call = {"type": "tool_use", "id": "call", "name": "write_file",
                    "input": {"path": "/mnt/user-data/workspace/recovered.txt", "content": "ok"}}
            store.stage_context(session, task["context_id"], "assistant", [call], key=f"step:{group}")
            store.put(session, f"execution:{identity}", {"type": "tool_execution", "id": identity,
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run", "call_id": "call",
                      "name": call["name"], "input": call["input"], "status": "pending"})
            store.put(session, f"group:{group}", {"type": "tool_group", "id": group,
                      "context_id": task["context_id"], "calls": [identity], "status": "pending"})
            store.put(session, "approval:old", {"type": "approval", "request_id": "old", "status": "requested"})
            session.commit()
        await runtime.start()
        try:
            await wait_for(lambda: bool(runtime.approval_events))
            request = next(iter(runtime.approval_events))
            assert request != "old"
            with db.new_session() as session:
                assert store.get(session, "approval:old")["status"] == "expired"
            await decide(runtime, request, "allowOnce")
            await wait_for(lambda: bool(model.reports))
            assert (isolated / ("a" * 32) / "recovered.txt").read_text(encoding="utf-8") == "ok"
            assert model.worker_calls == 0
        finally:
            await runtime.close()
    asyncio.run(run())


def test_direct_step_limit_does_not_execute_unavailable_tool(isolated, monkeypatch):
    from app.agent import agent_bus
    from app.agent.worker import Worker

    calls = []
    async def unexpected(_worker, execution):
        calls.append(execution)
        return "unexpected"
    monkeypatch.setattr(Worker, "bash_tool", unexpected)

    class LimitModel:
        model = "fake"

        async def stream(self, *, system, messages, tools, tool_choice=None):
            assert "bash" not in {tool["name"] for tool in tools}
            yield {"type": "tool_use", "id": new_id(), "name": "bash", "input": {"command": "pwd"}}
            yield {"type": "stop_reason", "reason": "tool_use"}

    async def run():
        runtime = Runtime(client=LimitModel())

        async def keep_records():
            pass

        monkeypatch.setattr(runtime, "_cleanup_finished", keep_records)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "达到上限", 1)
            task.update(status="running", run_id="run", direct=True, direct_phase="reporting", steps=40)
            store.put(session, f"task:{task['id']}", task)
            agent_bus.send(session, source=task["context_id"], target="main", task_id=task["id"],
                           content="直接执行【达到上限】的工具结果：完成", kind="direct", message_id="limit")
            session.commit()
        await runtime.start()
        try:
            def failed():
                with db.new_session() as session:
                    return store.get(session, f"task:{task['id']}")["status"] == "failed"
            await wait_for(failed)
            assert not calls
        finally:
            await runtime.close()
    asyncio.run(run())


def test_direct_task_cannot_wait_for_dependency(isolated):
    runtime = Runtime(client=DirectModel("bash", {"command": "pwd"}))
    with db.new_session() as session:
        store.initialize(session, isolated)
        direct = runtime.create_task(session, "快速任务", 1)
        direct.update(status="running", direct=True)
        store.put(session, f"task:{direct['id']}", direct)
        other = runtime.create_task(session, "后台任务", 2)
        result = runtime._main_action(session, "task_wait_for", {
            "task_id": direct["id"], "dependency_id": other["id"]}, 1, "wait")
        assert result.startswith("Error:")
        assert store.get(session, f"task:{direct['id']}")["dependency"] is None


class FirstActionModel:
    model = "fake"

    def __init__(self, name, args, intent="chat"):
        self.name, self.args, self.intent = name, args, intent
        self.calls = []
        self.reply_messages = []

    async def stream(self, *, system, messages, tools, tool_choice=None):
        if system.startswith("你是后台工作助手"):
            raise AssertionError("首个直接动作不应启动后台助手模型")
        if not self.calls:
            self.calls.append("first")
            assert system.startswith("你是用户长期使用的主助手")
            yield {"type": "tool_use", "id": new_id(), "name": self.name, "input": self.args}
        else:
            self.calls.append("reply")
            self.reply_messages = messages
            yield {"type": "text_delta", "text": "已处理结果。"}
        yield {"type": "stop_reason", "reason": "end_turn"}


def test_first_action_can_load_skill(isolated, monkeypatch):
    from app.agent import skills

    directory = skills.user_skills_path("a" * 32) / "review-guide"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: review-guide\ndescription: 用于代码审查\n---\n说明正文标记", encoding="utf-8")

    async def run():
        model = FirstActionModel("load_skill", {"name": "review-guide"})
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "skill-first", "加载审查技能", "build")
            await wait_for(lambda: bool(model.reply_messages))
            assert model.calls == ["first", "reply"]
            assert "说明正文标记" in json.dumps(model.reply_messages, ensure_ascii=False)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_first_action_reads_file_in_two_model_calls(isolated):
    (isolated / ("a" * 32) / "note.txt").write_text("workspace", encoding="utf-8")

    async def run():
        model = FirstActionModel("read_file", {"path": "/mnt/user-data/workspace/note.txt"})
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "read-first", "读取 note.txt", "build")
            await wait_for(lambda: bool(model.reply_messages))
            assert model.calls == ["first", "reply"]
            assert "workspace" in json.dumps(model.reply_messages, ensure_ascii=False)
            history = await checkpoint_history(runtime, "main")
            action = next(block for entry in history if isinstance(entry["content"], list)
                          for block in entry["content"] if block.get("type") == "tool_use")
            assert action["name"] == "read_file"
            assert action["input"] == {"path": "/mnt/user-data/workspace/note.txt"}
        finally:
            await runtime.close()
    asyncio.run(run())


def test_first_action_can_search_history_in_two_model_calls(isolated):
    async def run():
        model = FirstActionModel("task_search", {"keyword": "枫叶"}, intent="query")
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "远古枫叶", 1)
            task["status"] = "completed"
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        await runtime.start()
        try:
            await accept(runtime, "search-first", "查找枫叶任务", "build")
            await wait_for(lambda: bool(model.reply_messages))
            assert model.calls == ["first", "reply"]
            assert task["id"] in json.dumps(model.reply_messages, ensure_ascii=False)
        finally:
            await runtime.close()
    asyncio.run(run())
