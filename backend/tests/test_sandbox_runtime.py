import asyncio
import threading

import pytest

from app import db
from app.agent.runtime import Conflict, Runtime
from app.agent.worker import Worker
from app.config import settings
from app.models import new_id
from app.sandbox.lease import SandboxLeaseManager
from app.sessions import chat_store as store, execution_state
from fake_sandbox import FakeProvider, FakeSandbox


class RecordingSandbox(FakeSandbox):
    def __init__(self):
        self.commands = []
        self.stopped = threading.Event()
        self.started = threading.Event()
        self.wait = False
        self.operations = []

    def execute_command_in_scope(self, command, *, scope_id, timeout=None):
        self.commands.append((scope_id, command))
        self.started.set()
        if self.wait:
            assert self.stopped.wait(5)
            return "Error: 命令已停止"
        return "沙箱命令完成"

    def request_command_cancel(self, scope_id):
        self.stopped.set()

    def read_file(self, path, start_line=None, end_line=None):
        self.operations.append("read_file")
        return super().read_file(path, start_line, end_line)

    def write_file(self, path, content, append=False):
        self.operations.append("write_file")
        return super().write_file(path, content, append)


@pytest.fixture
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    db.dispose_engine()
    db.init_db(tmp_path)
    provider = FakeProvider()
    provider.sandbox = RecordingSandbox()
    runtime = Runtime(client=object(), sandbox_leases=SandboxLeaseManager(provider))
    with db.new_session() as session:
        store.initialize(session, tmp_path)
        task = runtime.create_task(session, "测试沙箱", 1)
        task.update(status="running", run_id=new_id())
        store.put(session, f"task:{task['id']}", task)
        session.commit()
    worker = Worker(runtime, task["id"])
    worker.context_id, worker.run_id = task["context_id"], task["run_id"]
    yield runtime, worker, provider.sandbox
    db.dispose_engine()


async def group(runtime, worker, name, args):
    identity, group_id, call_id = new_id(), new_id(), new_id()
    def save(session):
        store.put(session, f"execution:{identity}", {"type": "tool_execution", "id": identity, "call_id": call_id,
                  "task_id": worker.task_id, "context_id": worker.context_id, "run_id": worker.run_id,
                  "name": name, "input": args, "status": "pending"})
        store.put(session, f"group:{group_id}", {"type": "tool_group", "id": group_id,
                  "context_id": worker.context_id, "calls": [identity], "status": "pending"})
    await runtime.change(save)
    return group_id, identity


async def pending_approval(runtime):
    async with asyncio.timeout(5):
        while True:
            snapshot = await runtime.snapshot()
            if snapshot["approvals"]:
                return snapshot
            await asyncio.sleep(0.02)


def test_build_requires_each_command_approval_and_yolo_runs_in_sandbox(isolated_runtime):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        try:
            for command in ("echo one", "echo two"):
                group_id, identity = await group(runtime, worker, "bash", {"command": command})
                operation = asyncio.create_task(worker.process_group(group_id))
                snapshot = await pending_approval(runtime)
                approval = snapshot["approvals"][0]
                assert [option["option_id"] for option in approval["options"]] == ["allowOnce", "deny"]
                assert len(sandbox.commands) == (0 if command.endswith("one") else 1)
                with pytest.raises(Conflict):
                    await runtime.decide(approval["request_id"], "allowSession", assistant_id=snapshot["assistant_id"])
                await runtime.decide(approval["request_id"], "allowOnce", assistant_id=snapshot["assistant_id"])
                await operation
                with db.new_session() as session:
                    assert store.get(session, f"execution:{identity}")["output"] == "沙箱命令完成"
            execution_state.save_execution_state(settings.data_dir, "yolo")
            group_id, _ = await group(runtime, worker, "bash", {"command": "echo three"})
            await worker.process_group(group_id)
            assert [item[1] for item in sandbox.commands] == ["echo one", "echo two", "echo three"]
            assert not (await runtime.snapshot())["approvals"]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_stopping_a_sandbox_command_signals_its_scope(isolated_runtime):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        execution_state.save_execution_state(settings.data_dir, "yolo")
        sandbox.wait = True
        try:
            group_id, identity = await group(runtime, worker, "bash", {"command": "sleep 30"})
            operation = asyncio.create_task(worker.process_group(group_id))
            assert await asyncio.to_thread(sandbox.started.wait, 5)
            worker.stop.set()
            await asyncio.wait_for(operation, 5)
            assert sandbox.stopped.is_set()
            with db.new_session() as session:
                assert "已停止" in store.get(session, f"execution:{identity}")["output"]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_file_tools_use_the_sandbox_and_keep_versions(isolated_runtime):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        execution_state.save_execution_state(settings.data_dir, "edit")
        try:
            path = "/mnt/user-data/workspace/nested/dir/test.txt"
            group_id, _ = await group(runtime, worker, "write_file", {"path": path, "content": "hello"})
            await worker.process_group(group_id)
            group_id, identity = await group(runtime, worker, "read_file", {"path": path})
            await worker.process_group(group_id)
            assert sandbox.operations.count("write_file") == 1 and sandbox.operations.count("read_file") >= 2
            with db.new_session() as session:
                assert "hello" in store.get(session, f"execution:{identity}")["output"]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_existing_file_requires_reread_between_writes(isolated_runtime):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        execution_state.save_execution_state(settings.data_dir, "edit")
        path = "/mnt/user-data/workspace/versioned.txt"

        async def call(name, args):
            group_id, identity = await group(runtime, worker, name, args)
            await worker.process_group(group_id)
            with db.new_session() as session:
                return store.get(session, f"execution:{identity}")["output"]

        try:
            assert await call("write_file", {"path": path, "content": "old old"}) == "OK"
            assert "先读取" in await call("write_file", {"path": path, "content": "blocked"})
            assert "old old" in await call("read_file", {"path": path})
            assert await call("str_replace", {"path": path, "old_str": "old", "new_str": "new"}) == "OK"
            assert sandbox.read_file(path) == "new old"
            assert "先读取" in await call("write_file", {"path": path, "content": "!", "append": True})
            assert "new old" in await call("read_file", {"path": path})
            assert await call("write_file", {"path": path, "content": "! old", "append": True}) == "OK"
            assert sandbox.read_file(path) == "new old! old"
            assert "new old! old" in await call("read_file", {"path": path})
            assert await call("str_replace", {"path": path, "old_str": "old", "new_str": "NEW", "replace_all": True}) == "OK"
            assert sandbox.read_file(path) == "new NEW! NEW"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_write_guard_matches_deerflow_uninspectable_content(isolated_runtime):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        execution_state.save_execution_state(settings.data_dir, "edit")
        path = "/mnt/user-data/workspace/error-prefix.txt"
        sandbox.write_file(path, "Error: existing content")
        try:
            group_id, identity = await group(runtime, worker, "write_file", {"path": path, "content": "replaced"})
            await worker.process_group(group_id)
            with db.new_session() as session:
                assert store.get(session, f"execution:{identity}")["output"] == "OK"
            assert sandbox.read_file(path) == "replaced"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_direct_task_reuses_command_scope_until_finished(isolated_runtime):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        execution_state.save_execution_state(settings.data_dir, "yolo")
        def mark_direct(session):
            task = store.get(session, f"task:{worker.task_id}")
            task["direct"] = True
            store.put(session, f"task:{worker.task_id}", task)
        await runtime.change(mark_direct)
        try:
            for command in ("export EXAMPLE=one", "echo $EXAMPLE"):
                group_id, _ = await group(runtime, worker, "bash", {"command": command})
                await worker.process_group(group_id)
            assert len(sandbox.commands) == 2 and sandbox.commands[0][0] == sandbox.commands[1][0]
            assert worker.task_id in runtime.direct_sandboxes
            await runtime.release_direct_sandbox(worker.task_id)
            assert worker.task_id not in runtime.direct_sandboxes
        finally:
            await runtime.close()
    asyncio.run(run())


def test_invalid_timeout_is_rejected_before_the_command_starts(isolated_runtime, monkeypatch):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        execution_state.save_execution_state(settings.data_dir, "yolo")
        monkeypatch.setattr(settings, "bash_timeout", 0)
        try:
            group_id, identity = await group(runtime, worker, "bash", {"command": "echo test"})
            await worker.process_group(group_id)
            assert not sandbox.commands
            with db.new_session() as session:
                execution = store.get(session, f"execution:{identity}")
                assert "超时配置无效" in execution["output"] and not execution.get("result_unknown")
        finally:
            await runtime.close()
    asyncio.run(run())


def test_worker_reuses_its_shell_after_waiting_for_another_agent(isolated_runtime):
    from app.agent import agent_bus
    from app.agent.worker import AgentWaiting

    class Model:
        model = "fake"

        def __init__(self, target):
            self.target = target
            self.steps = 0

        async def stream(self, **kwargs):
            self.steps += 1
            if self.steps == 1:
                yield {"type": "tool_use", "id": "one", "name": "bash", "input": {"command": "export X=ok"}}
            elif self.steps == 2:
                yield {"type": "tool_use", "id": "wait", "name": "agent_wait", "input": {"context_id": self.target}}
            elif self.steps == 3:
                yield {"type": "tool_use", "id": "two", "name": "bash", "input": {"command": "echo $X"}}
            else:
                yield {"type": "text_delta", "text": "完成"}
            yield {"type": "stop_reason", "reason": "end_turn" if self.steps == 4 else "tool_use"}

    async def run():
        runtime, worker, sandbox = isolated_runtime
        execution_state.save_execution_state(settings.data_dir, "yolo")
        with db.new_session() as session:
            target = runtime.create_task(session, "同伴", 2)
            target["status"] = "running"
            store.put(session, f"task:{target['id']}", target)
            session.commit()
        runtime.client = Model(target["context_id"])
        try:
            with pytest.raises(AgentWaiting):
                await worker.run()
            assert len(sandbox.commands) == 1 and runtime.worker_sandboxes
            def reply(session):
                task = store.get(session, f"task:{worker.task_id}")
                task["status"] = "running"
                store.put(session, f"task:{worker.task_id}", task)
                agent_bus.send(session, source=target["context_id"], target=worker.context_id,
                               task_id=target["id"], content="同伴回复")
            await runtime.change(reply)
            resumed = Worker(runtime, worker.task_id)
            await resumed.run()
            assert len(sandbox.commands) == 2 and sandbox.commands[0][0] == sandbox.commands[1][0]
            assert not runtime.worker_sandboxes
        finally:
            await runtime.close()
    asyncio.run(run())


def test_failed_scope_cleanup_still_releases_other_scopes(isolated_runtime):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        released = []

        class Middleware:
            async def release(self, context):
                released.append(context["owner_id"])
                if context["owner_id"] == "failed":
                    raise RuntimeError("cleanup failed")

        runtime.agent_loop.sandbox_middleware = Middleware()
        runtime.direct_sandboxes = {"one": {"owner_id": "failed"}, "two": {"owner_id": "direct"}}
        runtime.worker_sandboxes = {"worker:run": {"owner_id": "worker", "context_id": "worker", "run_id": "run"}}
        try:
            with pytest.raises(RuntimeError, match="cleanup failed"):
                await runtime.release_sandboxes()
            assert released == ["failed", "direct", "worker"]
            assert not runtime.direct_sandboxes and not runtime.worker_sandboxes
            assert any("共享沙箱清理未确认" in item["text"] for item in (await runtime.snapshot())["messages"])
        finally:
            await runtime.close()
    asyncio.run(run())


def test_cancelled_file_operation_keeps_lease_until_thread_finishes(isolated_runtime, monkeypatch):
    async def run():
        runtime, worker, sandbox = isolated_runtime
        started, finish = threading.Event(), threading.Event()
        original = sandbox.read_file
        provider = runtime.agent_loop.sandbox_middleware.leases.provider
        release = provider.release
        released = []

        def slow(*args, **kwargs):
            started.set()
            assert finish.wait(5)
            return original(*args, **kwargs)

        def record_release(sandbox_id):
            released.append(sandbox_id)
            release(sandbox_id)

        monkeypatch.setattr(sandbox, "read_file", slow)
        monkeypatch.setattr(provider, "release", record_release)
        try:
            group_id, _ = await group(runtime, worker, "read_file", {"path": "/mnt/user-data/workspace/missing.txt"})
            operation = asyncio.create_task(worker.process_group(group_id))
            assert await asyncio.to_thread(started.wait, 5)
            operation.cancel()
            await asyncio.sleep(0.1)
            assert not operation.done() and not released
            finish.set()
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert len(released) == 1
        finally:
            finish.set()
            await runtime.close()
    asyncio.run(run())


def test_gateway_shutdown_continues_cleanup_after_one_user_fails():
    from app.gateway import Gateway

    events = []

    class RuntimeStub:
        def __init__(self, label, fail=False):
            self.label, self.fail = label, fail

        async def close(self):
            events.append(self.label)
            if self.fail:
                raise RuntimeError("cleanup failed")

    class Service:
        def __init__(self, label):
            self.label = label

        async def close(self):
            events.append(self.label)

    class Provider:
        def close(self):
            events.append("provider")

    async def run():
        gateway = object.__new__(Gateway)
        gateway.runtimes = {"a" * 32: RuntimeStub("first", fail=True), "b" * 32: RuntimeStub("second")}
        gateway.sandbox_provider = Provider()
        gateway.memory, gateway.rag = Service("memory"), Service("rag")
        with pytest.raises(RuntimeError, match="清理失败"):
            await gateway.close()
        assert events == ["first", "second", "provider", "memory", "rag"]
    asyncio.run(run())
