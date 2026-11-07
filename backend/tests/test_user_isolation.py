import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app import db
from app.agent import tools
from app.agent.permission_service import check_permission
from app.config import settings
from app.sessions import chat_store
from app.user_context import bind_user


def test_chat_and_workspace_are_user_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    try:
        for user_id, text in (("a" * 32, "甲"), ("b" * 32, "乙")):
            with bind_user(user_id), db.new_session() as session:
                tools.workspace_root().mkdir(parents=True)
                chat_store.initialize(session, settings.data_dir)
                chat_store.add_visible(session, "user", text, message_id="same-id")
                chat_store.put(session, "same-key", {"type": "example", "text": text})
                session.commit()
        for user_id, text in (("a" * 32, "甲"), ("b" * 32, "乙")):
            with bind_user(user_id), db.new_session() as session:
                assert [item["text"] for item in chat_store.visible(session)] == [text]
                assert chat_store.get(session, "same-key")["text"] == text
                assert tools.safe_path("file.txt").parent == tools.workspace_root()
                other_id = "b" * 32 if user_id.startswith("a") else "a" * 32
                with pytest.raises(ValueError):
                    tools.safe_path("../" + other_id + "/file.txt")
    finally:
        db.dispose_engine()


def test_chat_pagination_keeps_history_when_new_messages_arrive(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    try:
        with db.new_session() as session:
            chat_store.initialize(session, settings.data_dir)
            for index in range(5):
                chat_store.add_visible(session, "user", str(index))
            session.commit()
            page, cursor = chat_store.visible_page(session, 2)
            assert [item["text"] for item in page] == ["3", "4"] and cursor is not None
            chat_store.add_visible(session, "user", "5")
            session.commit()
            older, cursor = chat_store.visible_page(session, 2, cursor)
            assert [item["text"] for item in older] == ["1", "2"] and cursor is not None
            oldest, cursor = chat_store.visible_page(session, 2, cursor)
            assert [item["text"] for item in oldest] == ["0"] and cursor is None
    finally:
        db.dispose_engine()


def test_host_commands_are_unavailable_in_every_mode():
    assert "bash" not in tools.TOOL_HANDLERS
    assert all(tool["name"] != "bash" for tool in tools.TOOLS)
    for mode in ("build", "edit", "yolo"):
        result = check_permission(mode, {}, "bash", {"command": "echo hi"})
        assert result["decision"] == "deny"


def test_graph_persists_delivery_before_action(tmp_path, monkeypatch):
    from app.agent.checkpoint import SqliteSaver
    from app.agent.graph import AgentGraph

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    db.dispose_engine()
    db.init_db(settings.data_dir)

    async def run():
        with bind_user("a" * 32), db.new_session() as session:
            chat_store.initialize(session, settings.data_dir)
        async with SqliteSaver.from_conn_string(str(settings.data_dir / "app.db")) as saver:
            await saver.setup()
            with bind_user("a" * 32):
                graph = AgentGraph(saver)
                message = {"id": "delivery-1", "content": "结果"}
                seen = []
                thread_id = graph.thread_id("worker")

                async def action():
                    state = await graph.graph.aget_state({"configurable": {"thread_id": thread_id}})
                    seen.append(state.values.get("deliveries", {}).get(message["id"]) == message)

                await graph.invoke("worker", incoming=message, action=action)
                assert seen == [True]
                calls = []

                async def failed():
                    calls.append("failed")
                    raise ValueError("模型中断")

                async def next_action():
                    calls.append("next")

                with pytest.raises(ValueError):
                    await graph.invoke("worker", action=failed)
                await graph.invoke("worker", action=next_action)
                assert calls == ["failed", "next"]
                await graph.delete_thread(thread_id)
                assert not await graph.has_delivery("worker", message)

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


@pytest.mark.parametrize("saved", [False, True])
def test_delivery_failure_checks_checkpoint_and_pauses_if_unknown(tmp_path, monkeypatch, saved):
    from langgraph.checkpoint.memory import InMemorySaver
    from app.agent import agent_bus
    from app.agent.runtime import Runtime

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    db.dispose_engine()
    db.init_db(settings.data_dir)

    async def run():
        saver = InMemorySaver()
        runtime = Runtime(checkpointer=saver)
        runtime.client = None
        with db.new_session() as session:
            chat_store.initialize(session, settings.data_dir)
            task = runtime.create_task(session, "测试送达", 1)
            task.update(status="running", run_id="run")
            chat_store.put(session, f"task:{task['id']}", task)
            agent_bus.send(session, source="main", target=task["context_id"], task_id=task["id"],
                           content="重要消息", message_id="delivery")
            message = chat_store.get(session, "bus:delivery")
            session.commit()
        invoke = runtime.graph.invoke

        async def failed(*args, **kwargs):
            if saved:
                await invoke(*args, **kwargs)
            raise OSError("checkpoint error")

        monkeypatch.setattr(runtime.graph, "invoke", failed)
        assert await runtime.deliver_message(message) is saved
        with db.new_session() as session:
            row = chat_store.get(session, "bus:delivery")
            current = chat_store.get(session, f"task:{task['id']}")
            if saved:
                assert row["delivery"] == "delivered"
                assert [item["content"] for item in chat_store.history(session, task["context_id"])].count("重要消息") == 1
            else:
                assert row["delivery"] == "unknown" and current["status"] == "waiting_user"
                assert any("送达状态无法确定" in item["text"] for item in chat_store.visible(session))
        await runtime.close()
        if not saved:
            resumed = Runtime(checkpointer=saver)
            resumed.client = None
            await resumed.start()
            resumed.client = object()
            await resumed._schedule_once()
            with db.new_session() as session:
                assert chat_store.get(session, f"task:{task['id']}")["status"] == "waiting_user"
            assert not resumed.workers
            await resumed.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_worker_result_truncates_at_utf8_boundary():
    from app.agent.worker import _limit_result

    assert _limit_result("简短结果") == "简短结果"
    result = _limit_result("长" * 10000)
    assert len(result.encode("utf-8")) <= 16 * 1024
    assert result.endswith("[结果过长，已截断]")


def test_memory_search_is_user_scoped_and_within_token_budget(tmp_path, monkeypatch):
    import tiktoken
    from app.agent.memory import _snapshot, _store

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    _store("a" * 32, "甲", [
        {"id": "1", "content": "每周徒步", "confidence": 0.99},
        {"id": "2", "content": "喜欢喝咖啡", "confidence": 0.8},
    ])
    _store("b" * 32, "乙", [{"id": "3", "content": "喜欢喝绿茶", "confidence": 0.9}])
    result = _snapshot("a" * 32, "咖啡")
    assert result.index("喜欢喝咖啡") < result.index("每周徒步")
    assert "喜欢喝绿茶" not in result
    assert len(tiktoken.get_encoding("cl100k_base").encode(result)) <= 2000
    assert (settings.data_dir / "memory-index.sqlite").exists()


def test_memory_keeps_the_highest_confidence_facts(tmp_path, monkeypatch):
    from app.agent.memory import Memory, _read_files
    from app.models import User

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    with db.new_session() as session:
        session.add(User(id="a" * 32, username="a", password="a"))
        session.commit()

    class Model:
        async def stream(self, **_kwargs):
            facts = [{"content": f"fact-{index}", "confidence": 0.7 + index * 0.002} for index in range(110)]
            facts.append({"content": "uncertain", "confidence": 0.2})
            yield {"type": "text_delta", "text": json.dumps({"summary": "summary", "facts": facts})}

    async def run():
        memory = Memory()
        await memory._extract("a" * 32, Model(), [], 0)
        summary, facts = _read_files("a" * 32)
        assert summary == "summary" and len(facts) == 100
        assert {item["content"] for item in facts} == {f"fact-{index}" for index in range(10, 110)}
        await memory.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_memory_reset_fences_late_extraction_and_preserves_other_user(tmp_path, monkeypatch):
    from app.agent.memory import Memory, _read_files
    from app.models import User

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    with db.new_session() as session:
        session.add_all([User(id="a" * 32, username="a", password="a"),
                         User(id="b" * 32, username="b", password="b")])
        session.commit()

    class Model:
        model = "fake"

        def __init__(self, started=None, release=None):
            self.started, self.release = started, release

        async def stream(self, **_kwargs):
            if self.started:
                self.started.set()
                await self.release.wait()
            yield {"type": "text_delta", "text": '{"summary":"记忆","facts":[{"content":"喜欢喝茶","confidence":0.9}]}' }
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        memory = Memory()
        with bind_user("b" * 32):
            await memory._extract("b" * 32, Model(), [], 0)
        started, release = asyncio.Event(), asyncio.Event()

        async def old_extract():
            with bind_user("a" * 32):
                await memory._extract("a" * 32, Model(started, release), [], 0)

        old = asyncio.create_task(old_extract())
        await started.wait()
        with db.new_session() as session:
            user = session.get(User, "a" * 32)
            user.generation += 1
            session.commit()
        with bind_user("a" * 32):
            await memory.clear()
            with pytest.raises(PermissionError):
                memory.enqueue(Model(), [], 1, role="worker")
        release.set()
        await old
        assert _read_files("a" * 32) == ("", [])
        assert _read_files("b" * 32)[1][0]["content"] == "喜欢喝茶"
        await memory.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_reset_deletes_only_current_users_data(tmp_path, monkeypatch):
    from app.agent.checkpoint import SqliteSaver
    from app.agent.memory import _read_files, _store
    from app.agent.runtime import Runtime
    from app.models import User

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    monkeypatch.setattr(settings, "openai_api_key", "")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    with db.new_session() as session:
        session.add_all([User(id="a" * 32, username="a", password="a"),
                         User(id="b" * 32, username="b", password="b")])
        session.commit()

    async def run():
        async with SqliteSaver.from_conn_string(str(settings.data_dir / "app.db")) as saver:
            await saver.setup()
            runtimes = []
            for user_id in ("a" * 32, "b" * 32):
                with bind_user(user_id):
                    runtime = Runtime(checkpointer=saver)
                    await runtime.start()
                    runtimes.append(runtime)
                    with db.new_session() as session:
                        chat_store.add_visible(session, "user", user_id[0], message_id="message")
                        session.commit()
                    _store(user_id, "summary", [{"id": "fact", "content": user_id[0], "confidence": 0.9}])
                    await runtime.graph.invoke("main", incoming={"id": "delivery", "content": user_id[0]})
                    (settings.workspaces_dir / user_id / "keep.txt").write_text("keep", encoding="utf-8")
                    log_dir = settings.modelio_dir / user_id
                    log_dir.mkdir(parents=True)
                    (log_dir / "model-io-main.jsonl").write_text("old", encoding="utf-8")
            (settings.modelio_dir / "model-io-legacy.jsonl").write_text("legacy", encoding="utf-8")
            with bind_user("a" * 32):
                old = await runtimes[0].snapshot()
                old_thread = runtimes[0].graph.thread_id("main")
                await runtimes[0].reset(old["assistant_id"], "reset-1")
                await asyncio.wait_for(runtimes[0].reset_task, 10)
                new = await runtimes[0].snapshot()
                assert new["assistant_id"] != old["assistant_id"] and new["messages"] == []
                assert _read_files("a" * 32) == ("", [])
                assert not (settings.modelio_dir / ("a" * 32) / "model-io-main.jsonl").exists()
                assert (settings.workspaces_dir / ("a" * 32) / "keep.txt").exists()
                assert await saver.aget_tuple({"configurable": {"thread_id": old_thread}}) is None
            with bind_user("b" * 32):
                assert (await runtimes[1].snapshot())["messages"][0]["text"] == "b"
                assert _read_files("b" * 32)[1][0]["content"] == "b"
                assert (settings.modelio_dir / ("b" * 32) / "model-io-main.jsonl").exists()
            assert (settings.modelio_dir / "model-io-legacy.jsonl").exists()
            for user_id, runtime in zip(("a" * 32, "b" * 32), runtimes):
                with bind_user(user_id):
                    await runtime.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_reset_resumes_after_restart_during_deletion(tmp_path, monkeypatch):
    from app.agent.checkpoint import SqliteSaver
    from app.agent.runtime import Conflict, Runtime
    from app.models import User

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    with db.new_session() as session:
        session.add(User(id="a" * 32, username="a", password="a"))
        session.commit()

    async def run():
        async with SqliteSaver.from_conn_string(str(settings.data_dir / "app.db")) as saver:
            await saver.setup()
            runtime = Runtime(checkpointer=saver)
            runtime.client = None
            await runtime.start()
            await runtime.change(lambda session: chat_store.add_visible(session, "user", "旧对话"))
            await runtime.graph.invoke("main", incoming={"id": "old", "content": "旧对话"})
            thread_id = runtime.graph.thread_id("main")
            old_id = (await runtime.snapshot())["assistant_id"]

            async def failed_clear():
                raise OSError("memory error")

            monkeypatch.setattr(runtime.memory, "clear", failed_clear)
            await runtime.reset(old_id, "restart-reset")
            await runtime.reset_task
            reset = (await runtime.snapshot())["reset"]
            assert reset["status"] == "blocked" and reset["deletion_started"]
            with pytest.raises(Conflict, match="不能取消"):
                await runtime.cancel_reset(old_id, "restart-reset")
            await runtime.close()

            resumed = Runtime(checkpointer=saver)
            resumed.client = None
            await resumed.start()
            await asyncio.wait_for(resumed.reset_task, 10)
            current = await resumed.snapshot()
            assert current["assistant_id"] != old_id and current["messages"] == []
            assert await saver.aget_tuple({"configurable": {"thread_id": thread_id}}) is None
            await resumed.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_shared_state_and_publish_snapshot(tmp_path, monkeypatch):
    from app.agent import coordination
    from app.agent.runtime import Runtime

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    try:
        with bind_user("a" * 32), db.new_session() as session:
            chat_store.initialize(session, settings.data_dir)
            runtime = Runtime(client=object())
            first = runtime.create_task(session, "任务甲", 1, key="first")
            second = runtime.create_task(session, "任务乙", 2, key="second")
            for actor in ("main", first["context_id"]):
                coordination.call(session, "agent_subscribe", {"topic": "进展"},
                                  actor=actor, task_id=first["id"], request_id=actor)
            sent = coordination.call(session, "agent_publish", {"topic": "进展", "content": "消息"},
                                     actor=first["context_id"], task_id=first["id"], request_id="publish")
            coordination.call(session, "agent_subscribe", {"topic": "进展"},
                              actor=second["context_id"], task_id=second["id"], request_id="second")
            assert coordination.call(session, "agent_publish", {"topic": "进展", "content": "消息"},
                                     actor=first["context_id"], task_id=first["id"], request_id="publish") == sent
            assert len(chat_store.records(session, "agent_message")) == 2
            created = json.loads(coordination.call(session, "state_write", {
                "kind": "finding", "content": "初始", "expected_version": 0},
                actor=first["context_id"], task_id=first["id"], request_id="state-1"))
            state_id = created["state"]["id"]
            updated = json.loads(coordination.call(session, "state_write", {
                "state_id": state_id, "kind": "finding", "content": "更新", "expected_version": 1},
                actor=second["context_id"], task_id=second["id"], request_id="state-2"))
            conflict = json.loads(coordination.call(session, "state_write", {
                "state_id": state_id, "kind": "finding", "content": "覆盖", "expected_version": 1},
                actor="main", task_id=None, request_id="state-3"))
            assert updated["state"]["version"] == 2 and conflict["conflict"]
            session.commit()
        with bind_user("b" * 32), db.new_session() as session:
            chat_store.initialize(session, settings.data_dir)
            assert coordination.call(session, "state_read", {"state_id": state_id},
                                     actor="main", task_id=None, request_id="read").startswith("Error:")
    finally:
        db.dispose_engine()


def test_finished_task_cleanup_keeps_only_dedupe_and_chat(tmp_path, monkeypatch):
    from app.agent.checkpoint import SqliteSaver
    from app.agent.runtime import Runtime
    from app.models import User

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    with db.new_session() as session:
        session.add(User(id="a" * 32, username="a", password="a"))
        session.commit()

    async def run():
        async with SqliteSaver.from_conn_string(str(settings.data_dir / "app.db")) as saver:
            await saver.setup()
            with bind_user("a" * 32):
                runtime = Runtime(client=object(), checkpointer=saver)
                with db.new_session() as session:
                    chat_store.initialize(session, settings.data_dir)
                    task = runtime.create_task(session, "待清理内容", 1, key="request")
                    task.update(status="completed", result_sent=True)
                    chat_store.put(session, f"task:{task['id']}", task)
                    chat_store.add_visible(session, "assistant", "已完成", message_id="result")
                    session.commit()
                await runtime.graph.invoke(task["context_id"], incoming={"id": "one", "content": "结果"})
                thread_id = runtime.graph.thread_id(task["context_id"])
                await runtime._cleanup_finished()
                with db.new_session() as session:
                    assert chat_store.get(session, f"task:{task['id']}") is None
                    assert chat_store.get(session, "task-key:request") == {"type": "task_key", "processed": True}
                    assert runtime.create_task(session, "不同参数", 1, key="request") == {"processed": True}
                    assert chat_store.visible(session)[0]["text"] == "已完成"
                assert await saver.aget_tuple({"configurable": {"thread_id": thread_id}}) is None
                await runtime.close()
    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_dead_recipient_notifies_main_before_sender_finishes(tmp_path, monkeypatch):
    from app.agent import agent_bus
    from app.agent.runtime import Runtime

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    db.dispose_engine()
    db.init_db(settings.data_dir)

    async def run():
        runtime = Runtime(client=object())
        with db.new_session() as session:
            chat_store.initialize(session, settings.data_dir)
            sender = runtime.create_task(session, "发送任务", 1)
            target = runtime.create_task(session, "接收任务", 2)
            sender["status"], target["status"] = "running", "stopped"
            chat_store.put(session, f"task:{sender['id']}", sender)
            chat_store.put(session, f"task:{target['id']}", target)
            agent_bus.send(session, source=sender["context_id"], target=target["context_id"],
                           task_id=sender["id"], content="等待处理", message_id="cross-task")
            session.commit()
        await runtime._cleanup_finished()
        with db.new_session() as session:
            assert chat_store.get(session, "bus:cross-task")["delivery"] == "failed"
            notice = chat_store.get(session, "bus:undelivered:cross-task")
            assert notice["context_id"] == "main" and "等待处理" in notice["content"]
            assert chat_store.get(session, f"task:{sender['id']}")["status"] == "running"
        await runtime.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_main_and_worker_execute_through_graph_and_cleanup(tmp_path, monkeypatch):
    from app.agent.checkpoint import SqliteSaver
    from app.agent.runtime import Runtime
    from app.models import User

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    with db.new_session() as session:
        session.add(User(id="a" * 32, username="a", password="a"))
        session.commit()

    class Model:
        model = "fake"

        async def stream(self, *, system, messages, tools, tool_choice=None):
            if tool_choice and tool_choice["name"] == "route_input":
                request = json.loads(messages[-1]["content"])
                yield {"type": "tool_use", "id": "route", "name": "route_input", "input": {
                    "message_id": request["message_id"], "user_seq": request["user_seq"],
                    "intents": [{"type": "create", "instruction": "完成测试任务"}], "needs_clarification": False}}
            elif system.startswith("你是后台工作助手"):
                yield {"type": "text_delta", "text": "工作完成"}
            elif tool_choice and tool_choice["name"] == "task_review":
                tasks = json.loads(messages[-1]["content"].split("当前任务：", 1)[1])
                task = next(item for item in tasks if item["status"] == "reviewing")
                yield {"type": "tool_use", "id": "review", "name": "task_review",
                       "input": {"task_id": task["id"], "decision": "complete"}}
            else:
                yield {"type": "text_delta", "text": "任务已完成"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        async with SqliteSaver.from_conn_string(str(settings.data_dir / "app.db")) as saver:
            await saver.setup()
            with bind_user("a" * 32):
                runtime = Runtime(client=Model(), checkpointer=saver)
                await runtime.start()
                try:
                    assistant_id = (await runtime.snapshot())["assistant_id"]
                    await runtime.accept("message", "请完成任务", "build", assistant_id=assistant_id)
                    async with asyncio.timeout(12):
                        while True:
                            with db.new_session() as session:
                                tasks = chat_store.records(session, "task_state")
                                marker = chat_store.get(session, "task-key:route:message:0")
                                messages = chat_store.visible(session)
                            if marker and marker.get("processed") and not tasks:
                                break
                            await asyncio.sleep(0.05)
                    assert any(item["text"] == "任务已完成" for item in messages)
                    main_thread = runtime.graph.thread_id("main")
                    assert await saver.aget_tuple({"configurable": {"thread_id": main_thread}})
                finally:
                    await runtime.close()
    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_worker_limit_is_six_for_each_user(tmp_path, monkeypatch):
    from app.agent.checkpoint import SqliteSaver
    from app.agent.runtime import Runtime
    from app.models import User

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    db.dispose_engine()
    db.init_db(settings.data_dir)
    with db.new_session() as session:
        session.add_all([User(id="a" * 32, username="a", password="a"),
                         User(id="b" * 32, username="b", password="b")])
        session.commit()

    class Model:
        model = "fake"

        def __init__(self):
            self.active = 0
            self.peak = 0
            self.release = asyncio.Event()

        async def stream(self, **_kwargs):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await self.release.wait()
                yield {"type": "text_delta", "text": "完成"}
                yield {"type": "stop_reason", "reason": "end_turn"}
            finally:
                self.active -= 1

    async def run():
        async with SqliteSaver.from_conn_string(str(settings.data_dir / "app.db")) as saver:
            await saver.setup()
            runtimes, models = [], []
            for user_id in ("a" * 32, "b" * 32):
                with bind_user(user_id):
                    model = Model()
                    runtime = Runtime(client=model, checkpointer=saver)
                    await runtime.start()
                    def create_tasks(session):
                        for index in range(7):
                            runtime.create_task(session, f"任务 {index}", index + 1)
                    await runtime.change(create_tasks)
                    runtimes.append(runtime)
                    models.append(model)
            try:
                async with asyncio.timeout(15):
                    while any(model.active < 6 for model in models):
                        await asyncio.sleep(0.05)
                assert [model.active for model in models] == [6, 6]
                for user_id in ("a" * 32, "b" * 32):
                    with bind_user(user_id), db.new_session() as session:
                        assert sum(item["status"] == "queued" for item in chat_store.records(session, "task_state")) == 1
            finally:
                for model in models:
                    model.release.set()
                for user_id, runtime in zip(("a" * 32, "b" * 32), runtimes):
                    with bind_user(user_id):
                        await runtime.close()
    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_login_csrf_and_account_isolation(tmp_path, monkeypatch):
    from app.main import create_app

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    monkeypatch.setattr(settings, "openai_api_key", "")
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1111)) as first:
        assert first.get("/api/chat").status_code == 401
        csrf = first.get("/api/auth/session").json()["csrf"]
        assert first.post("/api/auth/register", json={"username": "alice", "password": "a"}).status_code == 403
        response = first.post("/api/auth/register", json={"username": "alice", "password": "a"},
                              headers={"X-CSRF-Token": csrf})
        assert response.status_code == 201
        alice_csrf = response.json()["csrf"]
        alice_id = first.get("/api/chat").json()["assistant_id"]
        message = {"assistant_id": alice_id, "client_message_id": "same-id", "text": "alice"}
        assert first.post("/api/chat/messages", json=message,
                          headers={"X-CSRF-Token": alice_csrf}).status_code == 202

        second = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 2222))
        csrf = second.get("/api/auth/session").json()["csrf"]
        response = second.post("/api/auth/register", json={"username": "bob", "password": "b"},
                               headers={"X-CSRF-Token": csrf})
        assert response.status_code == 201
        bob_csrf = response.json()["csrf"]
        bob = second.get("/api/chat").json()
        assert bob["assistant_id"] != alice_id and bob["messages"] == []
        assert second.post("/api/chat/messages", json=message,
                           headers={"X-CSRF-Token": bob_csrf}).status_code == 409
        assert first.get("/api/chat").json()["messages"][0]["text"] == "alice"
        second.close()
