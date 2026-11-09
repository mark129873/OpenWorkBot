import asyncio
import json
import re
import time
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import db
from app.agent import compact, tool_output
from app.agent.runtime import Runtime
from app.agent.worker import Worker
from app.config import settings
from app.sessions import chat_store as store, execution_state
from app.user_context import bind_user
from fake_sandbox import FakeSandbox
from test_chat_runtime import FakeModel, isolated, wait_for


def test_three_independent_triggers_and_full_request_count():
    config = replace(settings, context_window=1_000_000, compact_output_reserve=32_000, compact_buffer=13_000,
                     compact_trigger_messages=200, compact_trigger_tokens=800_000, compact_trigger_fraction=0.95)
    assert not compact.needs_compact(config, 199, 799_999)
    assert compact.needs_compact(config, 200, 10)
    assert compact.needs_compact(config, 1, 800_000)
    config = replace(config, compact_trigger_tokens=2_000_000, compact_trigger_fraction=0.8)
    assert compact.needs_compact(config, 1, 800_000)
    assert compact.estimate_request_tokens("system", [], [{"name": "x", "description": "工具" * 100}]) > (
        compact.estimate_request_tokens("system", [], []))


def test_recent_window_preserves_parallel_tools_and_current_request():
    messages = [
        SystemMessage(id="system", content="规则"),
        HumanMessage(id="input:latest", content="当前要求"),
        AIMessage(id="old-call", content="", tool_calls=[{"id": "old", "name": "read_file", "args": {}}]),
        ToolMessage(id="old-result", tool_call_id="old", content="旧结果"),
        HumanMessage(id="bus:latest", content="最新任务修改"),
        AIMessage(id="parallel", content="", tool_calls=[{"id": "a", "name": "t", "args": {}},
                                                         {"id": "b", "name": "t", "args": {}}]),
        ToolMessage(id="result-a", tool_call_id="a", content="a"),
        ToolMessage(id="result-b", tool_call_id="b", content="b"),
        AIMessage(id="last", content="继续"),
    ]
    old, kept = compact.select_messages(messages, 3)
    assert [message.id for message in old] == ["old-call", "old-result"]
    assert [message.id for message in kept] == ["system", "input:latest", "bus:latest",
                                                 "parallel", "result-a", "result-b", "last"]


def test_usage_anchor_resets_for_changed_request():
    context = {}
    previous = [{"role": "user", "content": "a"}, {"role": "user", "content": "动态状态"}]
    compact.record_usage(context, "system", previous, [], {"input_tokens": 5000}, "model")
    extended = [previous[0], {"role": "assistant", "content": "新增"}, previous[1]]
    assert compact.request_tokens(context, "system", extended, [], "model") >= 5000
    assert compact.request_tokens(context, "changed", extended, [], "model") < 5000


def test_existing_summary_counts_as_one_message(isolated, monkeypatch):
    class Model:
        async def stream(self, *, system, messages, tools, **kwargs):
            yield {"type": "text_delta", "text": "新摘要"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 4)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        runtime = Runtime(client=Model())
        messages = [HumanMessage(id="input:old", content="旧要求" * 50),
                    AIMessage(id="old-answer", content="旧回答" * 50),
                    HumanMessage(id="input:new", content="新要求")]
        assemble = lambda state, _: ("system", compact.history_for_state(state), [])
        assert await compact.compact_state(runtime, {"messages": messages}, {}, assemble) is None
        update = await compact.compact_state(runtime, {"messages": messages, "summary_text": "之前摘要"}, {}, assemble)
        assert update["summary_text"] == "新摘要"
        await runtime.close()

    asyncio.run(run())


def test_request_estimation_does_not_block_event_loop(isolated, monkeypatch):
    async def run():
        runtime = Runtime(client=FakeModel())
        real_count = compact.request_tokens

        def slow_count(*args):
            time.sleep(0.2)
            return real_count(*args)

        monkeypatch.setattr(compact, "request_tokens", slow_count)
        assemble = lambda state, _: ("system", compact.history_for_state(state), [])
        task = asyncio.create_task(compact.compact_state(
            runtime, {"messages": [HumanMessage(id="input:new", content="新要求")]}, {}, assemble))
        await asyncio.wait_for(asyncio.sleep(0.01), 0.1)
        assert not task.done()
        assert await task is None
        await runtime.close()

    asyncio.run(run())


def test_unconfirmed_pending_message_does_not_start_compaction(isolated, monkeypatch):
    from app.agent import langgraph_loop

    class Model:
        called = False

        async def stream(self, *, system, messages, tools, **kwargs):
            self.called = True
            yield {"type": "text_delta", "text": "不应调用"}

    async def run():
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "待确认", key="input:pending")
            session.commit()
        monkeypatch.setattr(langgraph_loop, "confirmed", lambda *_args: [])
        with pytest.raises(ValueError, match="未能确认"):
            await runtime._dialog("处理")
        assert not model.called
        with db.new_session() as session:
            assert len(store.records(session, "context_pending", context_id="main")) == 1
        await runtime.close()

    asyncio.run(run())


def test_compaction_is_checkpointed_before_model_failure(isolated, monkeypatch):
    class Model:
        model = "test"

        async def stream(self, *, system, messages, tools, **kwargs):
            if system.startswith("你是会话压缩器"):
                yield {"type": "text_delta", "text": "此前目标和约束。"}
                yield {"type": "stop_reason", "reason": "end_turn"}
            else:
                raise RuntimeError("模型不可用")

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 3)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        runtime = Runtime(client=Model())
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "旧要求" * 100, key="input:old")
            store.stage_context(session, "main", "assistant", "旧结果" * 100, key="step:old")
            store.stage_context(session, "main", "user", "当前要求", key="input:new")
            session.commit()
        with pytest.raises(RuntimeError, match="模型不可用"):
            await runtime._dialog("继续")
        async with runtime.graph.write_lock:
            thread = runtime.graph.thread_id("main:loop", parent_context="main")
        state = (await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})).values
        assert state["summary_text"] == "此前目标和约束。"
        assert [message.id for message in state["messages"]] == ["input:new"]
        with db.new_session() as session:
            assert not store.records(session, "context_pending", context_id="main")
        await runtime.close()

    asyncio.run(run())


def test_tool_archive_read_after_task_cleanup_and_reset(isolated, monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "tool_output_threshold", 50)
        monkeypatch.setattr(settings, "tool_output_preview_head", 10)
        monkeypatch.setattr(settings, "tool_output_preview_tail", 10)
        runtime = Runtime(client=FakeModel())
        runtime.client = None
        await runtime.start()
        execution_state.save_execution_state(isolated, "yolo")
        with db.new_session() as session:
            task = runtime.create_task(session, "测试工具归档", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:write", {"type": "tool_execution", "id": "write", "call_id": "call",
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run",
                      "name": "bash", "input": {"command": "large"}, "status": "pending"})
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        worker.sandbox, worker.sandbox_scope = FakeSandbox(), "scope"
        raw = "头" + "x" * 23000 + "中间内容" + "y" * 2000 + "尾"
        worker.sandbox.execute_command_in_scope = lambda *args, **kwargs: raw
        preview = await worker.process_execution("write")
        path = re.search(r"\[完整工具结果：([^\]]+)\]", preview).group(1)
        saved = isolated / ("a" * 32) / path.removeprefix("/mnt/user-data/workspace/")
        assert saved.read_text(encoding="utf-8") == raw
        assert "中间内容" not in preview
        with bind_user("b" * 32):
            (isolated / ("b" * 32)).mkdir()
            assert tool_output.read_reference(preview, max_chars=len(raw)) is None

        async def read(offset):
            identity = f"read-{offset}"
            with db.new_session() as session:
                store.put(session, f"execution:{identity}", {"type": "tool_execution", "id": identity,
                          "call_id": identity, "task_id": task["id"], "context_id": task["context_id"],
                          "run_id": "run", "name": "read_file",
                          "input": {"path": path, "offset": offset, "limit": 10000}, "status": "pending"})
                session.commit()
            return await worker.process_execution(identity)

        pages = [await read(offset) for offset in (0, 10000, 20000)]
        assert "".join(page.split("\n[继续读取：", 1)[0] for page in pages) == raw
        with db.new_session() as session:
            store.put(session, "execution:list", {"type": "tool_execution", "id": "list", "call_id": "list",
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run",
                      "name": "ls", "input": {"path": path.rsplit("/", 1)[0], "offset": 0, "limit": 1},
                      "status": "pending"})
            session.commit()
        assert path in await worker.process_execution("list")
        with db.new_session() as session:
            assert len(store.records(session, "tool_archive")) == 1
            current = store.get(session, f"task:{task['id']}")
            current.update(status="completed", result_sent=True)
            store.put(session, f"task:{task['id']}", current)
            session.commit()
        await runtime._cleanup_task(task["id"], {task["context_id"]})
        assert saved.is_file()
        user_file = isolated / ("a" * 32) / "user.txt"
        user_file.write_text("保留", encoding="utf-8")
        second = await tool_output.archive(runtime, "main", "bash", "another", "原始" * 100)
        second_path = re.search(r"\[完整工具结果：([^\]]+)\]", second).group(1)
        edited = isolated / ("a" * 32) / second_path.removeprefix("/mnt/user-data/workspace/")
        edited.write_text("用户修改的内容", encoding="utf-8")
        assert await tool_output.archive(runtime, "main", "bash", "another", "原始" * 100) == "原始" * 100
        assert edited.read_text(encoding="utf-8") == "用户修改的内容"
        old_id = (await runtime.snapshot())["assistant_id"]
        await runtime.reset(old_id, "reset-archive")

        def reset_done():
            with db.new_session() as session:
                return store.get(session, "meta")["assistant_id"] != old_id

        await wait_for(reset_done)
        assert not saved.exists()
        assert edited.read_text(encoding="utf-8") == "用户修改的内容"
        assert user_file.read_text(encoding="utf-8") == "保留"
        await runtime.close()

    asyncio.run(run())


def test_archive_failure_keeps_original_result(isolated, monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "tool_output_threshold", 5)
        monkeypatch.setattr(tool_output, "_write", lambda *_args: (_ for _ in ()).throw(OSError("磁盘失败")))
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            session.commit()
        original = "原始工具结果" * 100
        assert await tool_output.archive(runtime, "main", "bash", "call", original) == original
        await runtime.close()

    asyncio.run(run())


def test_archive_preserves_windows_line_endings(isolated, monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "tool_output_threshold", 5)
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            session.commit()
        original = "首行\r\n第二行\r\n" * 30
        preview = await tool_output.archive(runtime, "main", "bash", "windows-lines", original)
        assert tool_output.read_reference(preview, max_chars=len(original)) == original
        await runtime.close()

    asyncio.run(run())


def test_archived_error_output_is_read_as_bounded_file_content(isolated, monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "tool_output_threshold", 100)
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "测试", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        worker.sandbox, worker.sandbox_scope = FakeSandbox(), "scope"
        original = "Error: 结果未知\n" + "x" * 20000
        preview = await tool_output.archive(runtime, task["id"], "bash", "call", original)
        path = re.search(r"\[完整工具结果：([^\]]+)\]", preview).group(1)
        with db.new_session() as session:
            store.put(session, "execution:read-error", {"type": "tool_execution", "id": "read-error",
                      "call_id": "read-error", "task_id": task["id"], "context_id": task["context_id"],
                      "run_id": "run", "name": "read_file",
                      "input": {"path": path, "offset": 0, "limit": 1000}, "status": "pending"})
            session.commit()
        result = await worker.process_execution("read-error")
        assert result.startswith("[工具结果正文]\nError: 结果未知")
        assert len(result) < 1100 and "offset=1000" in result
        with db.new_session() as session:
            assert not store.get(session, "execution:read-error").get("result_unknown")
        await runtime.close()

    asyncio.run(run())


def test_large_rag_result_keeps_direct_answer_delivery(isolated, monkeypatch):
    async def run():
        monkeypatch.setattr(settings, "tool_output_threshold", 100)
        runtime = Runtime(client=FakeModel())
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "查询", 1, knowledge_question=True)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            store.put(session, "execution:query", {"type": "tool_execution", "id": "query", "call_id": "query",
                      "task_id": task["id"], "context_id": task["context_id"], "run_id": "run",
                      "name": "rag_search", "input": {"query": "问题"}, "status": "pending"})
            session.commit()
        worker = Worker(runtime, task["id"])
        worker.context_id, worker.run_id = task["context_id"], "run"
        answer = "知识答案" * 5000

        async def execute(_execution):
            return json.dumps({"status": "success", "answer": answer}, ensure_ascii=False)

        worker.execute = execute
        preview = await worker.process_execution("query")
        assert "[完整工具结果：" in preview
        await worker.finish("result", "完成")
        with db.new_session() as session:
            report = store.records(session, "agent_message", context_id="main")[-1]
            assert report["kind"] == "rag_answer"
            assert report["content"].startswith("知识答案")
            assert len(report["content"].encode("utf-8")) <= 16 * 1024
        await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize(("window", "expects_full"), [(100_000, True), (6_000, False)])
def test_summary_uses_archive_when_it_fits(isolated, monkeypatch, window, expects_full):
    class Model:
        model = "test"

        def __init__(self):
            self.input = None

        async def stream(self, *, system, messages, tools, **kwargs):
            assert system.startswith("你是会话压缩器") and not tools
            self.input = messages[0]["content"]
            yield {"type": "text_delta", "text": "旧任务摘要"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        monkeypatch.setattr(settings, "context_window", window)
        monkeypatch.setattr(settings, "compact_output_reserve", 1000)
        monkeypatch.setattr(settings, "compact_buffer", 100)
        monkeypatch.setattr(settings, "compact_trigger_messages", 4)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        monkeypatch.setattr(settings, "tool_output_threshold", 100)
        monkeypatch.setattr(settings, "tool_output_preview_head", 10)
        monkeypatch.setattr(settings, "tool_output_preview_tail", 10)
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            session.commit()
        original = "文" * 5000 + "结果中间标记" + "文" * 5000
        preview = await tool_output.archive(runtime, "main", "bash", "call", original)
        messages = [
            HumanMessage(id="input:old", content="先前要求"),
            AIMessage(id="call", content="", tool_calls=[{"id": "call", "name": "bash", "args": {}}]),
            ToolMessage(id="result", tool_call_id="call", content=preview),
            HumanMessage(id="input:new", content="当前要求"),
        ]
        assemble = lambda state, _: ("system", compact.history_for_state(state), [])
        update = await compact.compact_state(runtime, {"messages": messages, "summary_text": "先前摘要"}, {}, assemble)
        assert update["summary_text"] == "旧任务摘要"
        assert "先前摘要" in model.input and "[完整工具结果：" in model.input
        assert ("结果中间标记" in model.input) is expects_full
        await runtime.close()

    asyncio.run(run())


def test_context_overflow_compacts_then_retries_model_once(isolated, monkeypatch):
    class Model:
        model = "test"

        def __init__(self):
            self.normal_calls = 0
            self.summary_calls = 0

        async def stream(self, *, system, messages, tools, **kwargs):
            if system.startswith("你是会话压缩器"):
                self.summary_calls += 1
                yield {"type": "text_delta", "text": "原任务摘要"}
            else:
                self.normal_calls += 1
                if self.normal_calls == 1:
                    raise RuntimeError("context_length_exceeded")
                yield {"type": "text_delta", "text": "已继续。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "旧要求" * 100, key="input:old")
            store.stage_context(session, "main", "assistant", "旧回答" * 100, key="step:old")
            store.stage_context(session, "main", "user", "当前要求", key="input:new")
            session.commit()
        await runtime._dialog("继续")
        assert model.normal_calls == 2 and model.summary_calls == 1
        async with runtime.graph.write_lock:
            thread = runtime.graph.thread_id("main:loop", parent_context="main")
        state = (await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})).values
        assert state["summary_text"] == "原任务摘要"
        assert any(message.id == "input:new" for message in state["messages"])
        await runtime.close()

    asyncio.run(run())


def test_overflow_retry_does_not_repeat_completed_tool(isolated):
    class Model:
        model = "test"

        def __init__(self):
            self.normal_calls = 0
            self.summary_calls = 0

        async def stream(self, *, system, messages, tools, **kwargs):
            if system.startswith("你是会话压缩器"):
                self.summary_calls += 1
                yield {"type": "text_delta", "text": "任务已查询。"}
            else:
                self.normal_calls += 1
                if self.normal_calls == 1:
                    yield {"type": "tool_use", "id": "search", "name": "task_search", "input": {"keyword": ""}}
                elif self.normal_calls == 2:
                    raise RuntimeError("context_length_exceeded")
                else:
                    yield {"type": "text_delta", "text": "查询完成。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "历史要求" * 100, key="input:old")
            store.stage_context(session, "main", "assistant", "历史回复" * 100, key="step:old")
            store.stage_context(session, "main", "user", "查询任务", key="input:new")
            session.commit()
        original = runtime._main_tool
        calls = []

        def observe(*args, **kwargs):
            calls.append(args[1]["name"])
            return original(*args, **kwargs)

        runtime._main_tool = observe
        await runtime._dialog("处理")
        assert calls == ["task_search"]
        assert model.normal_calls == 3 and model.summary_calls == 1
        await runtime.close()

    asyncio.run(run())


def test_overflow_after_visible_output_is_not_retried(isolated):
    class Model:
        def __init__(self):
            self.calls = 0

        async def stream(self, *, system, messages, tools, **kwargs):
            self.calls += 1
            yield {"type": "text_delta", "text": "已显示"}
            raise RuntimeError("context_length_exceeded")

    async def run():
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "请求", key="input:new")
            session.commit()
        with pytest.raises(RuntimeError, match="context_length_exceeded"):
            await runtime._dialog("处理")
        assert model.calls == 1
        with db.new_session() as session:
            visible = store.visible(session)
            assert any(item["text"] == "已显示" and item["status"] == "failed" for item in visible)
        await runtime.close()

    asyncio.run(run())


def test_main_can_read_archive_without_general_direct_tools(isolated, monkeypatch):
    class Model:
        model = "test"

        def __init__(self):
            self.path = ""
            self.done = asyncio.Event()
            self.calls = 0

        async def stream(self, *, system, messages, tools, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert any(tool["name"] == "read_file" for tool in tools)
                yield {"type": "tool_use", "id": "archive-read", "name": "read_file",
                       "input": {"path": self.path, "offset": 0, "limit": 10000}}
            else:
                assert "完整归档内容" in str(messages)
                self.done.set()
                yield {"type": "text_delta", "text": "已读取。"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        monkeypatch.setattr(settings, "tool_output_threshold", 20)
        model = Model()
        runtime = Runtime(client=model)
        await runtime.start()
        preview = await tool_output.archive(runtime, "main", "bash", "archive-call", "完整归档内容" * 20)
        model.path = re.search(r"\[完整工具结果：([^\]]+)\]", preview).group(1)
        await runtime._dialog("继续处理", allow_direct=False)
        await asyncio.wait_for(model.done.wait(), 5)
        await runtime.close()

    asyncio.run(run())


@pytest.mark.parametrize("stop_reason", ["max_tokens", None])
def test_truncated_summary_keeps_original_context(isolated, monkeypatch, stop_reason):
    class Model:
        async def stream(self, *, system, messages, tools, **kwargs):
            yield {"type": "text_delta", "text": "不完整摘要"}
            if stop_reason:
                yield {"type": "stop_reason", "reason": stop_reason}

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 3)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        runtime = Runtime(client=Model())
        messages = [HumanMessage(id="input:old", content="旧任务" * 50),
                    AIMessage(id="old-answer", content="旧结果" * 50),
                    HumanMessage(id="input:new", content="新要求")]
        state = {"messages": messages, "summary_text": "原摘要"}
        assemble = lambda current, _: ("system", compact.history_for_state(current), [])
        with pytest.raises(compact.ContextCompactionError):
            await compact.compact_state(runtime, state, {}, assemble)
        assert state["messages"] == messages and state["summary_text"] == "原摘要"
        await runtime.close()

    asyncio.run(run())


def test_cancelled_summary_keeps_checkpoint_messages(isolated, monkeypatch):
    class Model:
        def __init__(self):
            self.started = asyncio.Event()

        async def stream(self, *, system, messages, tools, **kwargs):
            self.started.set()
            await asyncio.Event().wait()
            yield {"type": "text_delta", "text": "不会返回"}

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 3)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            store.stage_context(session, "main", "user", "旧要求" * 50, key="input:old")
            store.stage_context(session, "main", "assistant", "旧回答" * 50, key="step:old")
            store.stage_context(session, "main", "user", "当前要求", key="input:new")
            session.commit()
        call = asyncio.create_task(runtime._dialog("继续"))
        await asyncio.wait_for(model.started.wait(), 5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        async with runtime.graph.write_lock:
            thread = runtime.graph.thread_id("main:loop", parent_context="main")
        state = (await runtime.agent_loop.graph.aget_state({"configurable": {"thread_id": thread}})).values
        assert [message.id for message in state["messages"]] == ["input:old", "step:old", "input:new"]
        assert not state.get("summary_text")
        await runtime.close()

    asyncio.run(run())


def test_worker_receives_task_update_arriving_during_summary(isolated, monkeypatch):
    from app.agent import agent_bus

    class Model:
        def __init__(self):
            self.summarizing = asyncio.Event()
            self.release = asyncio.Event()
            self.history = None

        async def stream(self, *, system, messages, tools, **kwargs):
            if system.startswith("你是会话压缩器"):
                self.summarizing.set()
                await self.release.wait()
                yield {"type": "text_delta", "text": "旧工作摘要"}
            else:
                self.history = messages
                yield {"type": "text_delta", "text": "已处理最新要求"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 4)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "任务目标", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            store.stage_context(session, task["context_id"], "assistant", "旧处理" * 50, key="step:one")
            store.stage_context(session, task["context_id"], "user", "旧信息", key="old:info")
            store.stage_context(session, task["context_id"], "assistant", "旧结果" * 50, key="step:two")
            session.commit()
        worker = Worker(runtime, task["id"])
        call = asyncio.create_task(worker.run())
        await asyncio.wait_for(model.summarizing.wait(), 5)
        await runtime.change(lambda session: agent_bus.send(
            session, source="main", target=task["context_id"], task_id=task["id"],
            content="最新修改：不要执行删除", message_id="during-summary"))
        model.release.set()
        await asyncio.wait_for(call, 10)
        assert "最新修改：不要执行删除" in str(model.history)
        with db.new_session() as session:
            assert not store.records(session, "context_pending", context_id=task["context_id"])
        await runtime.close()

    asyncio.run(run())


def test_worker_waits_for_dependency_added_during_summary(isolated, monkeypatch):
    from app.agent.worker import AgentWaiting

    class Model:
        def __init__(self):
            self.summarizing = asyncio.Event()
            self.release = asyncio.Event()
            self.normal_calls = 0

        async def stream(self, *, system, messages, tools, **kwargs):
            if system.startswith("你是会话压缩器"):
                self.summarizing.set()
                await self.release.wait()
                yield {"type": "text_delta", "text": "旧工作摘要"}
            else:
                self.normal_calls += 1
                yield {"type": "text_delta", "text": "不应继续"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 3)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "任务目标", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            store.stage_context(session, task["context_id"], "assistant", "旧处理" * 50, key="step:one")
            store.stage_context(session, task["context_id"], "assistant", "旧结果" * 50, key="step:two")
            session.commit()
        worker = Worker(runtime, task["id"])
        call = asyncio.create_task(worker.run())
        await asyncio.wait_for(model.summarizing.wait(), 5)

        def add_dependency(session):
            current = store.get(session, f"task:{task['id']}")
            current["dependency"] = "other-task"
            store.put(session, f"task:{task['id']}", current)

        await runtime.change(add_dependency)
        model.release.set()
        with pytest.raises(AgentWaiting):
            await asyncio.wait_for(call, 10)
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}")["status"] == "waiting_dependency"
        assert model.normal_calls == 0
        await runtime.close()

    asyncio.run(run())


def test_worker_stop_during_summary_finishes_task(isolated, monkeypatch):
    from app.agent.worker import AgentWaiting

    class Model:
        def __init__(self):
            self.summarizing = asyncio.Event()
            self.release = asyncio.Event()
            self.normal_calls = 0

        async def stream(self, *, system, messages, tools, **kwargs):
            if system.startswith("你是会话压缩器"):
                self.summarizing.set()
                await self.release.wait()
                yield {"type": "text_delta", "text": "旧工作摘要"}
            else:
                self.normal_calls += 1
                yield {"type": "text_delta", "text": "不应继续"}
            yield {"type": "stop_reason", "reason": "end_turn"}

    async def run():
        monkeypatch.setattr(settings, "compact_trigger_messages", 3)
        monkeypatch.setattr(settings, "compact_keep_messages", 1)
        model = Model()
        runtime = Runtime(client=model)
        with db.new_session() as session:
            store.initialize(session, isolated)
            task = runtime.create_task(session, "任务目标", 1)
            task.update(status="running", run_id="run")
            store.put(session, f"task:{task['id']}", task)
            store.stage_context(session, task["context_id"], "assistant", "旧处理" * 50, key="step:one")
            store.stage_context(session, task["context_id"], "assistant", "旧结果" * 50, key="step:two")
            session.commit()
        worker = Worker(runtime, task["id"])
        call = asyncio.create_task(worker.run())
        await asyncio.wait_for(model.summarizing.wait(), 5)

        def request_stop(session):
            current = store.get(session, f"task:{task['id']}")
            current["stop_requested"] = True
            store.put(session, f"task:{task['id']}", current)

        await runtime.change(request_stop)
        model.release.set()
        with pytest.raises(AgentWaiting):
            await asyncio.wait_for(call, 10)
        with db.new_session() as session:
            assert store.get(session, f"task:{task['id']}")["status"] == "stopped"
        assert model.normal_calls == 0
        await runtime.close()

    asyncio.run(run())
