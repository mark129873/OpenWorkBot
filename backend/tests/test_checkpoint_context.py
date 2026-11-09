import asyncio
import json
from copy import deepcopy

import pytest

from app import db
from app.agent.checkpoint import SqliteSaver
from app.agent.runtime import Runtime
from app.config import settings
from app.models import User
from app.sessions import chat_store as store
from test_chat_runtime import accept, isolated, wait_for


class MemorySpy:
    def __init__(self):
        self.normal = []
        self.compactions = []

    async def snapshot(self, query):
        return ""

    def enqueue(self, client, records, events, generation, *, role):
        self.normal.append((deepcopy(records), deepcopy(events), role))

    def enqueue_compaction(self, client, records, events, generation):
        self.compactions.append((deepcopy(records), deepcopy(events)))


class Model:
    def __init__(self):
        self.requests = []

    async def stream(self, *, system, messages, tools, **kwargs):
        if system.startswith("你是会话压缩器"):
            yield {"type": "text_delta", "text": "已讨论此前工作。"}
        else:
            self.requests.append(deepcopy(messages))
            yield {"type": "text_delta", "text": "收到。"}
        yield {"type": "stop_reason", "reason": "end_turn"}


def _install_user():
    with db.new_session() as session:
        session.add(User(id="a" * 32, username="checkpoint-user", password="test"))
        session.commit()


async def _wait_for_reply(runtime, identity):
    def done():
        with db.new_session() as session:
            reply = store.get(session, f"reply:{identity}")
            return bool(reply and reply["status"] == "completed" and runtime.main_call is None)
    await wait_for(done)


async def _main_state(runtime):
    async with runtime.graph.write_lock:
        thread = runtime.graph.thread_id("main:loop", parent_context="main")
    return (await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})).values


def test_main_context_survives_restart_in_sqlite(isolated):
    async def run():
        _install_user()
        model, memory = Model(), MemorySpy()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=memory)
            await runtime.start()
            await accept(runtime, "first", "第一条问题", None)
            await _wait_for_reply(runtime, "first")
            before = await _main_state(runtime)
            assert [m.content for m in before["messages"] if m.type == "human"] == ["第一条问题"]
            await runtime.close()

            runtime = Runtime(client=model, checkpointer=saver, memory=memory)
            await runtime.start()
            try:
                await accept(runtime, "second", "继续处理", None)
                await _wait_for_reply(runtime, "second")
                state = await _main_state(runtime)
                assert [m.content for m in state["messages"] if m.type == "human"] == ["第一条问题", "继续处理"]
                assert any(row["content"] == "第一条问题" for row in model.requests[-1])
                with db.new_session() as session:
                    assert not store.records(session, "agent_context", context_id="main")
                    assert not store.records(session, "context_pending", context_id="main")
                    assert [row["text"] for row in store.visible(session) if row["role"] == "user"] == [
                        "第一条问题", "继续处理"]
                assert memory.normal[-1][0][0]["id"] == "input:first"
            finally:
                await runtime.close()
    asyncio.run(run())


def test_compaction_updates_checkpoint_and_flushes_old_messages(isolated, monkeypatch):
    async def run():
        _install_user()
        monkeypatch.setattr(settings, "context_window", 300)
        monkeypatch.setattr(settings, "compact_output_reserve", 50)
        monkeypatch.setattr(settings, "compact_buffer", 50)
        model, memory = Model(), MemorySpy()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=memory)
            await runtime.start()
            try:
                for index in range(4):
                    await accept(runtime, f"input-{index}", f"问题{index}:" + "文" * 240, None)
                    await _wait_for_reply(runtime, f"input-{index}")
                state = await _main_state(runtime)
                assert state["summary_text"] == "已讨论此前工作。"
                assert any(message.id == "input:input-3" for message in state["messages"])
                assert not any(message.id == "input:input-0" for message in state["messages"])
                assert memory.compactions
                assert any(row["id"] == "input:input-0" for row in memory.compactions[0][0])
            finally:
                await runtime.close()
    asyncio.run(run())


def test_checkpoint_history_keeps_tool_calls_paired(isolated):
    from app.agent.llm import openai_messages

    class ToolModel(Model):
        async def stream(self, *, system, messages, tools, **kwargs):
            openai_messages(system, messages)
            self.requests.append(deepcopy(messages))
            if len(self.requests) == 1:
                yield {"type": "tool_use", "id": "search", "name": "task_search", "input": {"keyword": "测试"}}
            else:
                yield {"type": "text_delta", "text": "已查询。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        _install_user()
        model, memory = ToolModel(), MemorySpy()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=memory)
            await runtime.start()
            try:
                await accept(runtime, "first", "查询任务", None)
                await _wait_for_reply(runtime, "first")
                await accept(runtime, "second", "继续", None)
                await _wait_for_reply(runtime, "second")
                state = await _main_state(runtime)
                assert sum(message.type == "tool" for message in state["messages"]) == 1
                assert sum(bool(getattr(message, "tool_calls", None)) for message in state["messages"]) == 1
                with db.new_session() as session:
                    assert not store.records(session, "context_pending", context_id="main")
            finally:
                await runtime.close()
    asyncio.run(run())


def test_direct_tool_report_keeps_checkpoint_history_valid(isolated):
    from app.agent.llm import openai_messages

    class DirectModel(Model):
        async def stream(self, *, system, messages, tools, **kwargs):
            openai_messages(system, messages)
            self.requests.append(deepcopy(messages))
            if len(self.requests) == 1:
                yield {"type": "tool_use", "id": "read", "name": "read_file",
                       "input": {"path": "/mnt/user-data/workspace/note.txt"}}
            else:
                yield {"type": "text_delta", "text": "已读取。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        _install_user()
        (isolated / ("a" * 32) / "note.txt").write_text("工作区内容", encoding="utf-8")
        model, memory = DirectModel(), MemorySpy()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=memory)
            await runtime.start()
            try:
                await accept(runtime, "first", "读取文件", "build")
                await wait_for(lambda: len(model.requests) >= 2 and runtime.main_call is None, timeout=8)
                assert "工作区内容" in json.dumps(model.requests[-1], ensure_ascii=False)
                await accept(runtime, "second", "继续", "build")
                await _wait_for_reply(runtime, "second")
                assert len(model.requests) >= 3
            finally:
                await runtime.close()
    asyncio.run(run())


def test_checkpoint_contexts_are_isolated_between_users(isolated):
    from app.user_context import bind_user

    async def run():
        _install_user()
        other = "b" * 32
        with db.new_session() as session:
            session.add(User(id=other, username="other-user", password="test"))
            session.commit()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            first, second = Runtime(client=Model(), checkpointer=saver, memory=MemorySpy()), None
            await first.start()
            with bind_user(other):
                second = Runtime(client=Model(), checkpointer=saver, memory=MemorySpy())
                await second.start()
            try:
                await accept(first, "same-id", "第一位用户", None)
                await _wait_for_reply(first, "same-id")
                with bind_user(other):
                    await accept(second, "same-id", "第二位用户", None)
                    await _wait_for_reply(second, "same-id")
                    second_state = await _main_state(second)
                first_state = await _main_state(first)
                assert [m.content for m in first_state["messages"] if m.type == "human"] == ["第一位用户"]
                assert [m.content for m in second_state["messages"] if m.type == "human"] == ["第二位用户"]
            finally:
                await first.close()
                with bind_user(other):
                    await second.close()
    asyncio.run(run())


def test_checkpoint_input_is_not_duplicated_after_acknowledgement_failure(isolated, monkeypatch):
    from app.agent import context as agent_context

    async def run():
        _install_user()
        model, memory = Model(), MemorySpy()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=memory)
            with db.new_session() as session:
                store.initialize(session, isolated)
                store.stage_context(session, "main", "user", "只投递一次", key="input:retry")
                session.commit()
            acknowledge = agent_context.acknowledge
            failed = False

            async def fail_once(*args):
                nonlocal failed
                if not failed:
                    failed = True
                    raise OSError("暂存确认失败")
                await acknowledge(*args)

            monkeypatch.setattr(agent_context, "acknowledge", fail_once)
            with pytest.raises(OSError, match="暂存确认失败"):
                await runtime._dialog("请回答。")
            with db.new_session() as session:
                assert len(store.records(session, "context_pending", context_id="main")) == 1
            await runtime._dialog("请回答。")
            state = await _main_state(runtime)
            assert [message.id for message in state["messages"]].count("input:retry") == 1
            with db.new_session() as session:
                assert not store.records(session, "context_pending", context_id="main")
            await runtime.close()
    asyncio.run(run())


def test_completed_reply_survives_late_acknowledgement_failure(isolated, monkeypatch):
    from app.agent import context as agent_context

    async def run():
        _install_user()
        model = Model()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=MemorySpy())
            with db.new_session() as session:
                store.initialize(session, isolated)
                store.stage_context(session, "main", "user", "第一条", key="input:first")
                session.commit()
            acknowledge, calls = agent_context.acknowledge, 0

            async def fail_after_reply(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("暂存清理失败")
                await acknowledge(*args)

            monkeypatch.setattr(agent_context, "acknowledge", fail_after_reply)
            await runtime._dialog("请回答。")
            with db.new_session() as session:
                assert len(store.records(session, "context_pending", context_id="main")) == 1
                store.stage_context(session, "main", "user", "继续", key="input:second")
                session.commit()
            await runtime._dialog("继续处理。")
            state = await _main_state(runtime)
            assert sum(message.content == "收到。" for message in state["messages"] if message.type == "ai") == 2
            with db.new_session() as session:
                assert not store.records(session, "context_pending", context_id="main")
            await runtime.close()
    asyncio.run(run())


def test_failed_checkpoint_keeps_input_for_retry(isolated):
    from langgraph.checkpoint.memory import InMemorySaver

    class FailingSaver(InMemorySaver):
        fail = True

        async def aput(self, config, checkpoint, metadata, new_versions):
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            if self.fail and any(getattr(message, "id", None) == "input:retry" for message in messages):
                raise OSError("检查点写入失败")
            return await super().aput(config, checkpoint, metadata, new_versions)

    async def run():
        _install_user()
        saver = FailingSaver()
        runtime = Runtime(client=Model(), checkpointer=saver, memory=MemorySpy())
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "可靠投递", key="input:retry")
            session.commit()
        with pytest.raises(OSError, match="检查点写入失败"):
            await runtime._dialog("请回答。")
        with db.new_session() as session:
            assert len(store.records(session, "context_pending", context_id="main")) == 1
        saver.fail = False
        await runtime._dialog("请回答。")
        state = await _main_state(runtime)
        assert [message.id for message in state["messages"]].count("input:retry") == 1
        with db.new_session() as session:
            assert not store.records(session, "context_pending", context_id="main")
        await runtime.close()
    asyncio.run(run())


def test_failed_memory_flush_does_not_block_compaction(isolated, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    from app.agent import compact

    async def run():
        monkeypatch.setattr(settings, "context_window", 300)
        monkeypatch.setattr(settings, "compact_output_reserve", 50)
        monkeypatch.setattr(settings, "compact_buffer", 50)
        runtime = Runtime(client=Model(), memory=MemorySpy())
        messages = [HumanMessage(id="first", content="文" * 1000),
                    AIMessage(id="answer", content="答复"), HumanMessage(id="last", content="继续")]

        async def fail(_messages):
            raise OSError("队列不可用")

        context = {"flush_memory": fail}
        projected = await compact.context_history(runtime, {"messages": messages}, context)
        assert "此前对话摘要" in projected[0]["content"]
        assert context["compacted_context"]["ids"] == ["first", "answer"]
        await runtime.close()
    asyncio.run(run())


def test_worker_restart_preserves_unknown_tool_result(isolated, monkeypatch):
    from app.agent.llm import openai_messages
    from app.agent.worker import Worker

    class WorkerModel(Model):
        def __init__(self):
            super().__init__()
            self.worker_messages = None

        async def stream(self, *, system, messages, tools, **kwargs):
            openai_messages(system, messages)
            if system.startswith("你是后台工作助手"):
                self.worker_messages = deepcopy(messages)
            yield {"type": "text_delta", "text": "已检查结果。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def unexpected(*args):
        raise AssertionError("结果未知的工具不能自动重跑")

    async def run():
        _install_user()
        model = WorkerModel()
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=MemorySpy())
            with db.new_session() as session:
                store.initialize(session, isolated)
                task = runtime.create_task(session, "执行操作", 1)
                task.update(status="running", run_id="run", steps=1)
                store.put(session, f"task:{task['id']}", task)
                group = f"{task['id']}:run:1"
                call = {"type": "tool_use", "id": "call", "name": "bash", "input": {"command": "example"}}
                store.stage_context(session, task["context_id"], "assistant", [call], key=f"step:{group}")
                store.put(session, "execution:call", {"type": "tool_execution", "id": "call", "call_id": "call",
                          "task_id": task["id"], "context_id": task["context_id"], "run_id": "run",
                          "name": "bash", "input": call["input"], "status": "started"})
                store.put(session, f"group:{group}", {"type": "tool_group", "id": group,
                          "context_id": task["context_id"], "calls": ["call"], "status": "pending"})
                session.commit()
        monkeypatch.setattr(Worker, "execute", unexpected)
        async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
            runtime = Runtime(client=model, checkpointer=saver, memory=MemorySpy())

            async def keep_records():
                pass

            monkeypatch.setattr(runtime, "_cleanup_finished", keep_records)
            await runtime.start()
            try:
                await wait_for(lambda: model.worker_messages is not None, timeout=8)
                assert "结果未知" in json.dumps(model.worker_messages, ensure_ascii=False)
                with db.new_session() as session:
                    assert store.get(session, "execution:call")["result_unknown"]
            finally:
                await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("failed", [False, True])
def test_resuming_group_reuses_completed_tool_without_reopening_sandbox(isolated, monkeypatch, failed):
    from app.agent.worker import Worker

    async def fail_acquire(*args, **kwargs):
        raise AssertionError("已完成的工具不应重新打开沙箱")

    async def run():
        runtime = Runtime(client=Model(), memory=MemorySpy())
        monkeypatch.setattr(runtime.sandbox_leases, "acquire", fail_acquire)
        previous = "Error: 上一步失败" if failed else "已保存的结果"
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "恢复任务", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            calls = [{"type": "tool_use", "id": "old", "name": "bash", "input": {"command": "example"}},
                     {"type": "tool_use", "id": "new", "name": "agent_list", "input": {}}]
            store.stage_context(session, task["context_id"], "assistant", calls, key="step:group")
            for call, status in zip(calls, ("completed", "pending")):
                store.put(session, f"execution:{call['id']}", {"type": "tool_execution", "id": call["id"],
                          "call_id": call["id"], "task_id": task["id"], "context_id": task["context_id"],
                          "run_id": "run", "name": call["name"], "input": call["input"], "status": status,
                          **({"output": previous} if status == "completed" else {})})
            store.put(session, "group:group", {"type": "tool_group", "id": "group", "context_id": task["context_id"],
                                               "calls": ["old", "new"], "status": "pending"})
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        await worker.process_group("group")
        with db.new_session() as session:
            assert store.get(session, "execution:old")["output"] == previous
            following = store.get(session, "execution:new")
            assert following["status"] == "completed"
            assert following["output"].startswith("Error:") is failed
            assert store.get(session, "group:group")["status"] == "completed"
        await runtime.close()
    asyncio.run(run())


def test_restart_repairs_unregistered_tool_call_without_running_it(isolated):
    from langgraph.checkpoint.memory import InMemorySaver
    from app.agent.llm import openai_messages

    class CrashAfterModel(InMemorySaver):
        fail = True

        async def aput(self, config, checkpoint, metadata, new_versions):
            result = await super().aput(config, checkpoint, metadata, new_versions)
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            if self.fail and any(getattr(message, "tool_calls", None) for message in messages):
                self.fail = False
                raise OSError("模型结果写入后中断")
            return result

    class ToolModel(Model):
        async def stream(self, *, system, messages, tools, **kwargs):
            openai_messages(system, messages)
            self.requests.append(deepcopy(messages))
            if len(self.requests) == 1:
                yield {"type": "tool_use", "id": "one", "name": "task_search", "input": {"keyword": "test"}}
            else:
                yield {"type": "text_delta", "text": "已恢复。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        _install_user()
        model, saver = ToolModel(), CrashAfterModel()
        runtime = Runtime(client=model, checkpointer=saver, memory=MemorySpy())
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "先查询", key="input:first")
            session.commit()
        with pytest.raises(OSError, match="模型结果写入后中断"):
            await runtime._dialog("请处理。")
        with db.new_session() as session:
            assert not store.records(session, "agent_action")
            store.stage_context(session, "main", "user", "继续", key="input:second")
            session.commit()
        await runtime._dialog("继续处理。")
        assert "未确认" in json.dumps(model.requests[-1], ensure_ascii=False)
        with db.new_session() as session:
            assert not store.records(session, "agent_action")
        await runtime.close()
    asyncio.run(run())


def test_uncommitted_assistant_reply_is_not_reused(isolated):
    from langgraph.checkpoint.memory import InMemorySaver

    class CrashAfterModel(InMemorySaver):
        fail = True

        async def aput(self, config, checkpoint, metadata, new_versions):
            result = await super().aput(config, checkpoint, metadata, new_versions)
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            if self.fail and any(getattr(message, "type", None) == "ai" for message in messages):
                self.fail = False
                raise OSError("回复尚未确认")
            return result

    class ReplyModel(Model):
        async def stream(self, *, system, messages, tools, **kwargs):
            self.requests.append(deepcopy(messages))
            yield {"type": "text_delta", "text": "未确认回复" if len(self.requests) == 1 else "新回复"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        _install_user()
        model = ReplyModel()
        runtime = Runtime(client=model, checkpointer=CrashAfterModel(), memory=MemorySpy())
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "第一次", key="input:first")
            session.commit()
        with pytest.raises(OSError, match="回复尚未确认"):
            await runtime._dialog("请处理。")
        with db.new_session() as session:
            store.stage_context(session, "main", "user", "继续", key="input:second")
            session.commit()
        await runtime._dialog("继续处理。")
        assert "未确认回复" not in json.dumps(model.requests[-1], ensure_ascii=False)
        state = await _main_state(runtime)
        assert not any(message.content == "未确认回复" for message in state["messages"])
        await runtime.close()
    asyncio.run(run())


def test_repair_reuses_recorded_results_and_preserves_unpaired_data(isolated):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.agent.context import confirmed, projected_history, repair_main_history
    from app.agent.llm import openai_messages

    with db.new_session() as session:
        store.initialize(session, isolated)
        store.put(session, "group:main:turn", {"type": "main_tool_group", "outputs": {"one": "已保存"}})
        session.commit()
    messages = [HumanMessage(id="input:first", content="查询"),
                AIMessage(id="step:turn", content="", tool_calls=[
                    {"id": "one", "name": "task_search", "args": {}},
                    {"id": "two", "name": "task_search", "args": {}}]),
                ToolMessage(id="extra", tool_call_id="other", content="额外数据"),
                HumanMessage(id="input:next", content="继续")]
    repaired, changed = repair_main_history(messages)
    assert changed
    assert [message.content for message in repaired if message.type == "tool"] == [
        "已保存", "Error: 此前工具调用的结果未确认，请先检查，禁止自动重跑。"]
    assert any(message.id == "extra" and "额外数据" in message.content for message in repaired)
    openai_messages("test", projected_history(repaired)[0])
    pending = [{"id": "result", "role": "user", "content": [{"type": "tool_result",
                "tool_use_id": "other", "content": "额外数据"}]}]
    pending[0]["id"] = "extra-group"
    orphan = ToolMessage(id="extra-group:other", tool_call_id="other", content="额外数据")
    repaired_orphan, _ = repair_main_history([orphan])
    assert confirmed(pending, repaired_orphan) == ["extra-group"]


def test_memory_uses_only_eligible_checkpoint_messages():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from app.agent.context import records_from_messages
    from app.agent.memory import _conversation

    messages = [
        HumanMessage(id="input:one", content="我喜欢简洁的回复"),
        AIMessage(id="call", content="", tool_calls=[{"id": "tool", "name": "search", "args": {}}]),
        ToolMessage(id="result", content="工具原文", tool_call_id="tool"),
        HumanMessage(id="bus:report", content="内部报告"),
        AIMessage(id="interrupted", content="未完成", additional_kwargs={"memory_status": "interrupted"}),
        AIMessage(id="answer", content="以后会简洁回复。", additional_kwargs={"memory_status": "completed"}),
    ]
    records = records_from_messages(messages, for_memory=True)
    conversation = _conversation(records, {"one": {"user_text": "我喜欢简洁的回复"}}, {})
    assert [(row["id"], row["content"]) for row in conversation] == [
        ("input:one", "我喜欢简洁的回复"), ("answer", "以后会简洁回复。")]


def test_compaction_extraction_does_not_advance_normal_watermark(isolated):
    from app.agent.memory import Memory
    from app.agent.memory_store import _advance, _state

    class ExtractModel:
        async def stream(self, **kwargs):
            yield {"type": "text_delta", "text": json.dumps({"user": {}, "history": {},
                                                              "newFacts": [], "factsToRemove": []})}

    async def run():
        _install_user()
        user = "a" * 32
        _advance(user, "already-extracted")
        records = [{"id": "input:one", "role": "user", "content": "我喜欢简洁"},
                   {"id": "answer", "role": "assistant", "content": "收到"}]
        await Memory()._extract(user, ExtractModel(), records, {"one": {"user_text": "我喜欢简洁"}}, 0,
                                bypass_watermark=True)
        assert _state(user)["main"] == ["id", "already-extracted"]
    asyncio.run(run())


def test_compaction_queue_keeps_separate_batches(monkeypatch):
    from app.agent import memory as memory_module
    from app.agent.memory import Memory

    async def run():
        monkeypatch.setattr(memory_module, "_DEBOUNCE", 100)
        memory, calls = Memory(), []

        async def extract(_user, _client, records, _events, _generation, _signals, *, bypass_watermark=False):
            calls.append((records[0]["id"], bypass_watermark))

        monkeypatch.setattr(memory, "_extract", extract)
        memory.enqueue(None, [{"id": "normal"}], {}, 0, role="main")
        memory.enqueue_compaction(None, [{"id": "old-1"}], {}, 0)
        memory.enqueue_compaction(None, [{"id": "old-2"}], {}, 0)
        await memory.close()
        assert calls == [("old-1", True), ("old-2", True), ("normal", False)]
    asyncio.run(run())
