import asyncio

from app import db
from app.agent.runtime import Runtime
from app.agent.worker import Worker
from app.sessions import chat_store as store
from test_chat_runtime import accept, isolated, wait_for


class Model:
    model = "fake"

    def __init__(self, calls):
        self.calls = calls
        self.count = 0

    async def stream(self, *, system, messages, tools, tool_choice=None):
        assert not tool_choice or tool_choice["name"] != "route_input"
        self.count += 1
        if self.count == 1:
            for call in self.calls:
                yield {"type": "tool_use", **call}
        else:
            yield {"type": "text_delta", "text": "收到"}
        yield {"type": "stop_reason", "reason": "end_turn"}


def test_main_todo_is_written_to_agent_state(isolated):
    async def run():
        items = [{"content": "第一步", "status": "in_progress"}]
        runtime = Runtime(client=Model([{"id": "todo", "name": "write_todos", "input": {"todos": items}}]))
        await runtime.start()
        try:
            await accept(runtime, "todo", "请处理多步骤任务", None)
            def completed():
                with db.new_session() as session:
                    return (store.get(session, "reply:todo") or {}).get("status") == "completed"
            await wait_for(completed)
            thread = runtime.graph.thread_id("main:loop", parent_context="main")
            state = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
            assert state.values["todos"] == items
            with db.new_session() as session:
                assert store.history(session, "main")[-1]["role"] == "assistant"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_main_todo_persists_across_messages_and_sqlite_reopen(isolated):
    from app.agent.checkpoint import SqliteSaver

    async def run():
        items = [{"content": "持续事项", "status": "pending"}]
        model = Model([{"id": "todo", "name": "write_todos", "input": {"todos": items}}])
        for identity in ("first", "second"):
            async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
                await saver.setup()
                runtime = Runtime(client=model, checkpointer=saver)
                await runtime.start()
                try:
                    await accept(runtime, identity, "继续处理", None)
                    def completed():
                        with db.new_session() as session:
                            return (store.get(session, f"reply:{identity}") or {}).get("status") == "completed"
                    await wait_for(completed)
                    thread = runtime.graph.thread_id("main:loop", parent_context="main")
                    state = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
                    assert state.values["todos"] == items
                finally:
                    await runtime.close()
    asyncio.run(run())


def test_failed_main_action_skips_later_actions(isolated):
    async def run():
        model = Model([
            {"id": "stop", "name": "task_stop", "input": {"task_id": "missing"}},
            {"id": "create", "name": "task_create", "input": {"title": "不应创建", "goal": "不应执行"}},
        ])
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "one", "先停止，再创建任务", None)
            await wait_for(lambda: model.count >= 2)
            with db.new_session() as session:
                assert not store.records(session, "task_state")
                results = [block for row in store.history(session, "main") if isinstance(row["content"], list)
                           for block in row["content"] if block.get("type") == "tool_result"]
                assert "不存在" in results[0]["content"] and "未执行" in results[1]["content"]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_stop_and_create_are_registered_together_in_order(isolated, monkeypatch):
    async def run():
        model = Model([])
        runtime = Runtime(client=model)

        async def idle_scheduler():
            await asyncio.Event().wait()

        monkeypatch.setattr(runtime, "_schedule", idle_scheduler)
        await runtime.start()
        try:
            task = await runtime.change(lambda session: runtime.create_task(session, "任务甲", 1))
            model.calls = [
                {"id": "stop", "name": "task_stop", "input": {"task_id": task["id"]}},
                {"id": "create", "name": "task_create", "input": {"title": "任务乙", "goal": "执行任务乙"}},
            ]
            await accept(runtime, "actions", "停止甲，再创建乙", None)
            def completed():
                with db.new_session() as session:
                    groups = store.records(session, "main_tool_group")
                    return groups and groups[0]["status"] == "completed"
            await wait_for(completed)
            with db.new_session() as session:
                group = store.records(session, "main_tool_group")[0]
                assert [call["name"] for call in group["calls"]] == ["task_stop", "task_create"]
                assert store.get(session, f"task:{task['id']}")["status"] == "stopped"
                assert any(item["goal"] == "执行任务乙" for item in store.records(session, "task_state"))
        finally:
            await runtime.close()
    asyncio.run(run())


def test_mixed_main_tools_are_ordered_without_blocking_chat(isolated, monkeypatch):
    async def run():
        events, started, release = [], asyncio.Event(), asyncio.Event()
        model = Model([
            {"id": "first", "name": "task_search", "input": {"keyword": "first"}},
            {"id": "bash", "name": "bash", "input": {"command": "pwd"}},
            {"id": "last", "name": "task_search", "input": {"keyword": "last"}},
        ])
        runtime = Runtime(client=model)
        original = runtime._main_tool

        def main_tool(session, call, source_seq, review_report_id=None):
            events.append(call["input"].get("keyword"))
            return original(session, call, source_seq, review_report_id)

        async def bash(worker, execution):
            events.append("bash")
            started.set()
            await release.wait()
            return "ok"

        monkeypatch.setattr(runtime, "_main_tool", main_tool)
        monkeypatch.setattr(Worker, "bash_tool", bash)
        await runtime.start()
        try:
            await accept(runtime, "one", "按顺序执行", None)
            await asyncio.wait_for(started.wait(), 5)
            assert events == ["first", "bash"]
            await accept(runtime, "two", "另一个问题", None)
            await wait_for(lambda: model.count >= 2)
            assert events == ["first", "bash"]
            release.set()
            await wait_for(lambda: events == ["first", "bash", "last"])
        finally:
            release.set()
            await runtime.close()
    asyncio.run(run())


def test_mixed_todo_updates_main_state_before_next_tool(isolated, monkeypatch):
    from app.agent.checkpoint import SqliteSaver

    async def run(checkpointer):
        started, release = asyncio.Event(), asyncio.Event()
        items = [{"content": "主助手事项", "status": "pending"}]
        model = Model([
            {"id": "todo", "name": "write_todos", "input": {"todos": items}},
            {"id": "bash", "name": "bash", "input": {"command": "pwd"}},
        ])
        runtime = Runtime(client=model, checkpointer=checkpointer)

        async def bash(worker, execution):
            started.set()
            await release.wait()
            return "ok"

        monkeypatch.setattr(Worker, "bash_tool", bash)
        await runtime.start()
        try:
            await accept(runtime, "one", "先列计划再执行", None)
            await asyncio.wait_for(started.wait(), 5)
            thread = runtime.graph.thread_id("main:loop", parent_context="main")
            state = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
            assert state.values["todos"] == items
        finally:
            release.set()
            await runtime.close()
    async def with_sqlite():
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            await saver.setup()
            await run(saver)
    asyncio.run(with_sqlite())


def test_mixed_group_stops_after_failed_action(isolated, monkeypatch):
    class RecordingModel(Model):
        async def stream(self, *, system, messages, tools, tool_choice=None):
            self.messages = messages
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run():
        called = []
        model = RecordingModel([
            {"id": "stop", "name": "task_stop", "input": {"task_id": "missing"}},
            {"id": "bash", "name": "bash", "input": {"command": "pwd"}},
            {"id": "create", "name": "task_create", "input": {"title": "不应创建", "goal": "不应执行"}},
        ])
        runtime = Runtime(client=model)

        async def bash(worker, execution):
            called.append("bash")
            return "ok"

        monkeypatch.setattr(Worker, "bash_tool", bash)
        await runtime.start()
        try:
            await accept(runtime, "failure", "依次停止、执行命令和创建任务", None)
            await wait_for(lambda: model.count >= 2)
            assert not called
            assert "前一项工具失败" in str(model.messages)
            with db.new_session() as session:
                assert not any(not task.get("direct") for task in store.records(session, "task_state"))
        finally:
            await runtime.close()
    asyncio.run(run())


def test_mixed_todo_does_not_overwrite_newer_message(isolated, monkeypatch):
    from app.agent.checkpoint import SqliteSaver

    class RecordingModel(Model):
        def __init__(self, calls):
            super().__init__(calls)
            self.reported = asyncio.Event()

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if "重新判断 TODO" in str(messages):
                self.reported.set()
            async for event in super().stream(system=system, messages=messages, tools=tools, tool_choice=tool_choice):
                yield event

    async def run(checkpointer):
        started, release = asyncio.Event(), asyncio.Event()
        model = RecordingModel([
            {"id": "bash", "name": "bash", "input": {"command": "pwd"}},
            {"id": "todo", "name": "write_todos", "input": {
                "todos": [{"content": "旧计划", "status": "pending"}]}},
        ])
        runtime = Runtime(client=model, checkpointer=checkpointer)

        async def bash(worker, execution):
            started.set()
            await release.wait()
            return "ok"

        monkeypatch.setattr(Worker, "bash_tool", bash)
        await runtime.start()
        lock, held = None, False
        try:
            await accept(runtime, "first", "执行命令并列计划", None)
            await asyncio.wait_for(started.wait(), 5)
            thread = runtime.graph.thread_id("main:loop", parent_context="main")
            lock = runtime.graph.locks[thread]
            await lock.acquire()
            held = True
            release.set()
            def todo_validated():
                with db.new_session() as session:
                    return any(item["name"] == "write_todos" and item["status"] == "completed"
                               for item in store.records(session, "tool_execution"))
            await wait_for(todo_validated)
            await accept(runtime, "second", "计划有变", None)
            lock.release()
            held = False
            await asyncio.wait_for(model.reported.wait(), 5)
            state = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})
            assert not state.values.get("todos")
        finally:
            release.set()
            if held:
                lock.release()
            await runtime.close()
    async def with_sqlite():
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            await saver.setup()
            await run(saver)
    asyncio.run(with_sqlite())


def test_new_message_does_not_lose_interrupted_request(isolated):
    class InterruptedModel:
        model = "fake"

        def __init__(self):
            self.started = asyncio.Event()
            self.calls = 0
            self.contexts = []

        async def stream(self, *, system, messages, tools, tool_choice=None):
            self.calls += 1
            self.contexts.append(str(messages))
            if self.calls == 1:
                self.started.set()
                await asyncio.Event().wait()
            yield {"type": "text_delta", "text": "已处理"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = InterruptedModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "first", "先处理甲", None)
            await asyncio.wait_for(model.started.wait(), 4)
            await accept(runtime, "second", "再处理乙", None)
            def processed():
                with db.new_session() as session:
                    return all((store.get(session, f"input:{identity}") or {}).get("status") == "processed"
                               for identity in ("first", "second"))
            await wait_for(processed)
            assert "先处理甲" in model.contexts[1] and "再处理乙" in model.contexts[1]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_unavailable_tools_do_not_retry_the_message_forever(isolated):
    class UnavailableModel:
        model = "fake"

        def __init__(self):
            self.calls = 0

        async def stream(self, *, system, messages, tools, tool_choice=None):
            self.calls += 1
            yield {"type": "tool_use", "id": str(self.calls), "name": "missing_tool", "input": {}}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = UnavailableModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "invalid", "测试无效工具", None)
            def failed():
                with db.new_session() as session:
                    return (store.get(session, "input:invalid") or {}).get("status") == "failed"
            await wait_for(failed)
            assert model.calls == 2
        finally:
            await runtime.close()
    asyncio.run(run())


def test_main_and_worker_keep_separate_todos(isolated):
    from app.sessions import execution_state

    class SeparateModel:
        model = "fake"

        def __init__(self):
            self.main_calls = 0
            self.worker_calls = 0
            self.started, self.release = asyncio.Event(), asyncio.Event()

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if system.startswith("你是后台工作助手"):
                self.worker_calls += 1
                if self.worker_calls == 1:
                    yield {"type": "tool_use", "id": "worker-todo", "name": "write_todos",
                           "input": {"todos": [{"content": "后台事项", "status": "pending"}]}}
                else:
                    self.started.set()
                    await self.release.wait()
                    yield {"type": "text_delta", "text": "完成"}
            else:
                self.main_calls += 1
                if self.main_calls == 1:
                    yield {"type": "tool_use", "id": "main-todo", "name": "write_todos",
                           "input": {"todos": [{"content": "主助手事项", "status": "in_progress"}]}}
                elif self.main_calls == 2:
                    yield {"type": "tool_use", "id": "create", "name": "task_create",
                           "input": {"title": "后台任务", "goal": "执行后台任务"}}
                else:
                    yield {"type": "text_delta", "text": "已受理"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        execution_state.save_permission_rules(isolated, {
            "version": 1, "allow": [], "ask": [], "deny": [{"toolName": "write_todos"}]})
        model = SeparateModel()
        runtime = Runtime(client=model)
        await runtime.start()
        try:
            await accept(runtime, "separate", "创建后台任务", None)
            await asyncio.wait_for(model.started.wait(), 5)
            with db.new_session() as session:
                task = next(item for item in store.records(session, "task_state") if not item.get("direct"))
            main_thread = runtime.graph.thread_id("main:loop", parent_context="main")
            worker_thread = runtime.graph.thread_id(f"{task['context_id']}:loop", parent_context=task["context_id"])
            main = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": main_thread}})
            worker = await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": worker_thread}})
            assert main.values["todos"][0]["content"] == "主助手事项"
            assert worker.values["todos"][0]["content"] == "后台事项"
        finally:
            model.release.set()
            await runtime.close()
    asyncio.run(run())
