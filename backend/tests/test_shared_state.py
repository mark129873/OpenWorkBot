import asyncio
import json

import pytest
from sqlalchemy import select

from app import db
from app.agent import agent_bus, shared_state
from app.agent.runtime import Runtime
from app.models import AgentMessage, Part, SharedState
from app.sessions import chat_store as store
from app.user_context import bind_user, scoped
from test_chat_runtime import accept, checkpoint_history, isolated, wait_for


def test_board_filters_and_keeps_other_agents_and_users(isolated):
    with db.new_session() as session:
        store.initialize(session, isolated)
        runtime = Runtime(client=object())
        first = runtime.create_task(session, "任务一", 1)
        second = runtime.create_task(session, "任务二", 2)
        for task in (first, second):
            task.update(status="running", run_id=f"run-{task['id']}")
            store.put(session, f"task:{task['id']}", task)
        items = [{"content": "进行一", "status": "in_progress"},
                 {"content": "待处理", "status": "pending"},
                 {"content": "已完成", "status": "completed"},
                 {"content": "进行二", "status": "in_progress"}]
        assert shared_state.replace(session, first["context_id"], first["id"], first["run_id"], items)
        assert shared_state.replace(session, second["context_id"], second["id"], second["run_id"], items[:1])
        shared_state.replace(session, first["context_id"], first["id"], first["run_id"], items[1:])
        first["status"] = "waiting_agent"
        store.put(session, f"task:{first['id']}", first)
        board = shared_state.snapshot(session)
        assert {(item["agent"], item["todo"], item["status"]) for item in board} == {
            (first["context_id"], "进行二", "waiting_agent"),
            (second["context_id"], "进行一", "running")}
        assert not shared_state.replace(session, first["context_id"], first["id"], "old-run", items)
        first["stop_requested"] = True
        store.put(session, f"task:{first['id']}", first)
        assert [item["agent"] for item in shared_state.snapshot(session)] == [second["context_id"]]
        session.commit()
    with bind_user("b" * 32), db.new_session() as session:
        store.initialize(session, isolated)
        assert shared_state.snapshot(session) == []
        assert store.records(session, "agent_message") == []


def test_agents_receive_live_board_and_finished_worker_disappears(isolated):
    class Model:
        model = "fake"

        def __init__(self):
            self.worker_calls = 0
            self.started, self.release = asyncio.Event(), asyncio.Event()
            self.main_messages, self.worker_messages = None, None

        async def stream(self, *, system, messages, tools, **kwargs):
            if system.startswith("你是后台工作助手"):
                self.worker_calls += 1
                if self.worker_calls == 1:
                    yield {"type": "tool_use", "id": "todo", "name": "write_todos", "input": {"todos": [
                        {"content": "后台正在工作", "status": "in_progress"},
                        {"content": "后台等待处理", "status": "pending"}]}}
                else:
                    self.worker_messages = messages
                    self.started.set()
                    await self.release.wait()
                    yield {"type": "text_delta", "text": "完成"}
            else:
                self.main_messages = messages
                yield {"type": "text_delta", "text": "收到"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = Model()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "empty", "简单问题", None)
            await wait_for(lambda: model.main_messages is not None)
            assert any(isinstance(message.get("content"), str) and "\n[]\n</shared_state>" in message["content"]
                       for message in model.main_messages)
            model.main_messages = None
            task = await runtime.change(lambda session: runtime.create_task(session, "后台任务", 1))
            await asyncio.wait_for(model.started.wait(), 8)
            await accept(runtime, "board", "查看进展", None)
            await wait_for(lambda: model.main_messages is not None)
            assert "后台正在工作" in json.dumps(model.worker_messages, ensure_ascii=False)
            assert "后台正在工作" in json.dumps(model.main_messages, ensure_ascii=False)
            assert "后台等待处理" not in json.dumps(model.main_messages, ensure_ascii=False)
            history = await checkpoint_history(runtime, "main")
            assert "<shared_state>" not in json.dumps(history, ensure_ascii=False)
            model.release.set()
            def board_empty():
                with db.new_session() as session:
                    return not shared_state.snapshot(session)
            await wait_for(board_empty)
            with db.new_session() as session:
                assert not list(session.scalars(select(SharedState).where(SharedState.task_id == task["id"])))
        finally:
            model.release.set()
            await runtime.close()
    asyncio.run(run())


def test_stop_clears_only_that_tasks_board(isolated):
    async def run():
        runtime = Runtime(client=object())
        with db.new_session() as session:
            store.initialize(session, isolated)
        def create(session):
            tasks = [runtime.create_task(session, f"任务{index}", index) for index in (1, 2)]
            for task in tasks:
                task.update(status="waiting_dependency", run_id=f"run-{task['id']}")
                store.put(session, f"task:{task['id']}", task)
                shared_state.replace(session, task["context_id"], task["id"], task["run_id"],
                                     [{"content": task["goal"], "status": "in_progress"}], thread_id="thread")
            return tasks
        first, second = await runtime.change(create)
        await runtime.change(lambda session: runtime._stop_task(session, store.get(session, f"task:{first['id']}")))
        with db.new_session() as session:
            assert [row.task_id for row in session.scalars(select(SharedState))] == [second["id"]]
            assert [item["task_id"] for item in store.records(session, "todo_sync")] == [second["id"]]
        await runtime.close()
    asyncio.run(run())


def test_resuming_tool_group_keeps_board_until_todo_is_replayed(isolated, monkeypatch):
    from app.agent.worker import Worker

    async def run():
        runtime = Runtime(client=object())
        with db.new_session() as session:
            store.initialize(session, isolated)
        def create(session):
            task = runtime.create_task(session, "恢复工作", 1)
            task.update(status="running", run_id="run-one")
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:todo", {"type": "tool_execution", "id": "todo", "call_id": "todo",
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run-one",
                      "name": "write_todos", "input": {"todos": [{"content": "新工作", "status": "in_progress"}]},
                      "status": "completed", "output": "已更新 TODO。"})
            store.put(session, "group:one", {"type": "tool_group", "id": "one", "context_id": task["context_id"],
                      "calls": ["todo"], "status": "pending"})
            return task
        task = await runtime.change(create)
        thread = runtime.graph.thread_id(f"{task['context_id']}:loop", parent_context=task["context_id"])
        await runtime.agent_loop.graph.aupdate_state({"configurable": {"thread_id": thread}}, {
            "todos": [{"content": "旧工作", "status": "in_progress"}]})
        await runtime.change(lambda session: shared_state.replace(
            session, task["context_id"], task["id"], "run-one", [{"content": "新工作", "status": "in_progress"}],
            thread_id=thread, execution_id="todo"))
        started, release = asyncio.Event(), asyncio.Event()
        original = Worker.process_execution
        async def paused(worker, identity):
            started.set()
            await release.wait()
            return await original(worker, identity)
        monkeypatch.setattr(Worker, "process_execution", paused)
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run-one"
        replay = asyncio.create_task(worker.process_group("one"))
        try:
            await asyncio.wait_for(started.wait(), 8)
            with db.new_session() as session:
                assert [item["todo"] for item in shared_state.snapshot(session)] == ["新工作"]
            release.set()
            await asyncio.wait_for(replay, 8)
        finally:
            release.set()
            await asyncio.gather(replay, return_exceptions=True)
            await runtime.close()
    asyncio.run(run())


def test_interrupted_projection_recovers_from_checkpoint(isolated):
    from app.agent.checkpoint import SqliteSaver

    async def run():
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            await saver.setup()
            runtime = Runtime(checkpointer=saver)
            runtime.client = None
            await runtime.start()
            try:
                def create(session):
                    task = runtime.create_task(session, "恢复任务", 1)
                    task.update(status="running", run_id="run-one")
                    store.put(session, f"task:{task['id']}", task)
                    return task
                task = await runtime.change(create)
                thread = runtime.graph.thread_id(f"{task['context_id']}:loop", parent_context=task["context_id"])
                await runtime.agent_loop.graph.aupdate_state({"configurable": {"thread_id": thread}}, {
                    "todos": [{"content": "已提交的状态", "status": "in_progress"}]})
                await runtime.change(lambda session: shared_state.replace(
                    session, task["context_id"], task["id"], "run-one",
                    [{"content": "未提交的状态", "status": "in_progress"}], thread_id=thread))
            finally:
                await runtime.close()
            resumed = Runtime(checkpointer=saver)
            resumed.client = None
            await resumed.start()
            try:
                with db.new_session() as session:
                    assert [item["todo"] for item in shared_state.snapshot(session)] == ["已提交的状态"]
                    assert not store.records(session, "todo_sync")
            finally:
                await resumed.close()
    asyncio.run(run())


@pytest.mark.parametrize("direct", [False, True])
def test_completed_todo_is_replayed_after_restart_before_board_write(isolated, direct):
    from app.agent.checkpoint import SqliteSaver
    from app.agent.worker import Worker

    async def idle():
        await asyncio.Event().wait()

    async def run():
        items = [{"content": "中断前已完成的 TODO", "status": "in_progress"}]
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            await saver.setup()
            runtime = Runtime(checkpointer=saver)
            runtime.client, runtime._schedule = None, idle
            await runtime.start()
            try:
                def create(session):
                    task = runtime.create_task(session, "中断恢复", 1)
                    task.update(status="running", run_id="run-one", direct=direct)
                    if direct:
                        task.update(direct_phase="executing", direct_group="group-one")
                    store.put(session, f"task:{task['id']}", task)
                    store.put(session, "execution:todo", {"type": "tool_execution", "id": "todo",
                              "task_id": task["id"], "context_id": task["context_id"], "run_id": "run-one",
                              "call_id": "todo", "name": "write_todos", "input": {"todos": items},
                              "status": "completed", "output": "已更新 TODO。"})
                    store.put(session, "group:group-one", {"type": "tool_group", "id": "group-one",
                              "context_id": task["context_id"], "calls": ["todo"], "status": "pending"})
                    return task
                task = await runtime.change(create)
                runtime.graph.thread_id(f"{task['context_id']}:loop", parent_context=task["context_id"])
            finally:
                await runtime.close()
            resumed = Runtime(checkpointer=saver)
            resumed.client, resumed._schedule = None, idle
            await resumed.start()
            try:
                with db.new_session() as session:
                    board = shared_state.snapshot(session)
                    assert [item["todo"] for item in board] == ([] if direct else [items[0]["content"]])
                worker = Worker(resumed, task["id"])
                worker.context_id, worker.run_id = task["context_id"], "run-one"
                await worker.process_group("group-one")
                context = "main" if direct else task["context_id"]
                thread = resumed.graph.thread_id(f"{context}:loop", parent_context=context)
                state = await resumed.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
                assert state.values["todos"] == items
                with db.new_session() as session:
                    assert [item["todo"] for item in shared_state.snapshot(session)] == [items[0]["content"]]
                    assert store.get(session, "group:group-one")["status"] == "completed"
            finally:
                await resumed.close()
    asyncio.run(run())


def test_main_todo_recovers_once_from_interrupted_tool_group(isolated):
    from langchain_core.messages import AIMessage
    from app.agent.checkpoint import SqliteSaver

    async def run():
        items = [{"content": "主助手已完成的 TODO 更新", "status": "in_progress"}]
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            await saver.setup()
            runtime = Runtime(checkpointer=saver)
            runtime.client = None
            await runtime.start()
            try:
                thread = runtime.graph.thread_id("main:loop", parent_context="main")
                call = AIMessage(id="step:old", content="", tool_calls=[
                    {"id": "todo", "name": "write_todos", "args": {"todos": items}, "type": "tool_call"}])
                await runtime.agent_loop.graph.aupdate_state({"configurable": {"thread_id": thread}},
                                                            {"todos": [], "messages": [call]})
                await runtime.change(lambda session: store.put(session, "group:main:old", {
                    "type": "main_tool_group", "id": "main:old", "context_id": "main", "status": "pending",
                    "calls": [{"id": "todo", "name": "write_todos", "input": {"todos": items}}],
                    "outputs": {"todo": "已更新 TODO。"}}))
            finally:
                await runtime.close()
            resumed = Runtime(checkpointer=saver)
            resumed.client = None
            await resumed.start()
            try:
                config = {"configurable": {"thread_id": thread}}
                state = await resumed.agent_loop.graph.aget_state(config)
                assert state.values["todos"] == items
                assert state.values["messages"][0].tool_calls[0]["name"] == "write_todos"
                with db.new_session() as session:
                    assert store.get(session, "group:main:old")["todo_recovered"]
                    assert shared_state.snapshot(session) == []
                newer = [{"content": "更新后的 TODO", "status": "in_progress"}]
                await resumed.agent_loop.graph.aupdate_state(config, {"todos": newer})
            finally:
                await resumed.close()
            again = Runtime(checkpointer=saver)
            again.client = None
            await again.start()
            try:
                assert (await again.agent_loop.graph.aget_state(config)).values["todos"] == newer
            finally:
                await again.close()
    asyncio.run(run())


def test_message_bus_uses_own_table_and_clear_removes_both(isolated):
    with db.new_session() as session:
        store.initialize(session, isolated)
        agent_bus.send(session, source="main", target="worker", task_id="task", content="通知", message_id="one")
        shared_state.replace(session, "main", None, "run", [{"content": "进行", "status": "in_progress"}])
        session.commit()
        assert session.get(AgentMessage, scoped("bus:one")) is not None
        assert session.get(Part, scoped("bus:one")) is None
        message = store.records(session, "agent_message", context_id="worker", delivery="pending")[0]
        agent_bus.deliver(session, message)
        agent_bus.processed(session, message)
        session.commit()
        assert store.get(session, "bus:one")["delivery"] == "processed"
        store.clear(session)
        session.commit()
        assert not list(session.scalars(select(AgentMessage)))
        assert not list(session.scalars(select(SharedState)))
