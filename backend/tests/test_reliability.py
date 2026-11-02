import asyncio
import os
import subprocess
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.agent import compact, loop, subtask, tools
from app.agent.llm import AnthropicStreamClient
from app.config import Settings, settings
from app.models import Base, Message, TurnUsage
from app.sessions import replay, store
from app.sessions.recorder import TurnRecorder


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield session
    engine.dispose()


def collect(generator):
    async def run():
        return [event async for event in generator]
    return asyncio.run(run())


class FakeClient:
    model = "fake"

    def __init__(self, steps):
        self.steps = iter(steps)

    async def stream(self, **_kwargs):
        for event in next(self.steps):
            if isinstance(event, Exception):
                raise event
            yield event


def turn(db, events, **kwargs):
    recorder = TurnRecorder(db, "session", "fake", 100)
    deps = loop.TurnDeps(client=FakeClient([events]), recorder=recorder,
                         system_prompt="", history=[{"role": "user", "content": "请求"}], **kwargs)
    return deps, collect(loop.run_turn(deps))


def usage(input_tokens, output_tokens, complete=False):
    return {"type": "usage", "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_read_tokens": 2, "cache_creation_tokens": 1, "complete": complete}


def test_stopped_and_unknown_tools_are_not_replayed_as_success(db, monkeypatch):
    stopped = False

    def execute(_name, _input):
        nonlocal stopped
        stopped = True
        return "已完成"

    monkeypatch.setattr(loop, "execute_tool", execute)
    events = [usage(10, 3, True),
              {"type": "tool_use", "id": "first", "name": "read_file", "input": {"path": "a"}},
              {"type": "tool_use", "id": "second", "name": "read_file", "input": {"path": "b"}}]
    deps, output = turn(db, events, stop_flag=lambda: stopped)
    assert output[-1]["state"] == "stopped"
    results = replay.load_history(db, "session")[-1]["content"]
    assert results[0]["content"] == "已完成" and not results[0].get("is_error")
    assert results[1]["is_error"] and "未执行" in results[1]["content"]
    items = replay.load_replay(db, "session")["turns"][0]["work_items"]
    assert next(item for item in items if item.get("input") == {"path": "b"})["status"] == "stopped"

    # 模拟进程在工具开始后丢失结果。
    part = next(db.scalars(select(Message).where(Message.id == deps.recorder._message_id)))
    tool = next(item for item in items if item.get("input") == {"path": "b"})
    deps.recorder.upsert_tool_part(part.id, tool["id"], {
        "tool_call_id": "second", "name": "read_file", "input": {"path": "b"}, "status": "running"})
    result = replay.load_history(db, "session")[-1]["content"][1]
    assert result["is_error"] and "结果未知" in result["content"]
    deps.recorder.upsert_tool_part(part.id, tool["id"], {
        "tool_call_id": "second", "name": "read_file", "input": {"path": "b"},
        "status": "stopped", "output": "部分结果"})
    result = replay.load_history(db, "session")[-1]["content"][1]
    assert result["is_error"] and "可能不完整" in result["content"] and "部分结果" in result["content"]


def test_truncated_reply_is_failed_and_never_executes_partial_tool(db, monkeypatch):
    monkeypatch.setattr(loop, "execute_tool", lambda *_: pytest.fail("截断工具不能执行"))
    deps, output = turn(db, [{"type": "text_delta", "text": "部分结果"},
                             {"type": "tool_use", "id": "x", "name": "write_file", "input": {}},
                             {"type": "stop_reason", "reason": "max_tokens"}, usage(12, 7)])
    assert output[-1]["state"] == "failed"
    message = db.get(Message, deps.recorder._message_id)
    assert store._load(message.data)["finish"] == "max_tokens"
    assert replay.load_replay(db, "session")["turns"][0]["final_text"] == "部分结果"
    fact = db.get(TurnUsage, ("session", deps.recorder.turn_id))
    assert (fact.input_tokens, fact.output_tokens, fact.cache_read_input_tokens) == (12, 7, 2)


def test_subtask_limit_reports_failure(db, monkeypatch):
    monkeypatch.setattr(subtask, "SUBTASK_MAX_TURNS", 2)
    monkeypatch.setattr(subtask, "execute_tool", lambda *_: "完成")
    step = [usage(2, 1, True), {"type": "tool_use", "id": "x", "name": "read_file", "input": {"path": "a"}}]
    recorder = TurnRecorder(db, "session", "fake", 100)
    recorder.begin_turn("请求")
    message_id = recorder.step_message()
    deps = loop.TurnDeps(client=FakeClient([step, step]), recorder=recorder, system_prompt="", queue=asyncio.Queue())
    output = collect(subtask.run_subtask_events(deps, message_id, {"goal": "目标"}))
    assert output[-1]["_subtask_status"] == "failed"
    assert output[-1]["_subtask_output"].startswith("Error:")
    events = []
    while not deps.queue.empty():
        events.append(deps.queue.get_nowait())
    assert events[0]["subtask_id"] == events[-1]["subtask_id"]
    assert events[-1]["status"] == "failed"


def test_truncated_subtask_does_not_execute_partial_tool(db, monkeypatch):
    monkeypatch.setattr(subtask, "execute_tool", lambda *_: pytest.fail("截断工具不能执行"))
    recorder = TurnRecorder(db, "session", "fake", 100)
    recorder.begin_turn("请求")
    message_id = recorder.step_message()
    step = [{"type": "tool_use", "id": "x", "name": "write_file", "input": {}},
            {"type": "stop_reason", "reason": "max_tokens"}]
    deps = loop.TurnDeps(client=FakeClient([step]), recorder=recorder, system_prompt="", queue=asyncio.Queue())
    output = collect(subtask.run_subtask_events(deps, message_id, {"goal": "目标"}))
    assert output[-1]["_subtask_status"] == "failed"
    assert "输出达到上限" in output[-1]["_subtask_output"]


def test_compaction_can_recover_after_three_failures(db, monkeypatch):
    attempts = 0

    async def summarize(_client, _history):
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise RuntimeError("暂不可用")
        return "简短摘要"

    monkeypatch.setattr(compact, "summarize_history", summarize)
    monkeypatch.setattr(compact, "microcompact_parts", lambda *_args, **_kwargs: 0)
    saved = []
    monkeypatch.setattr(store, "save_compaction", lambda *_args: saved.append(True))
    recorder = TurnRecorder(db, "session", "fake", 100)
    recorder.begin_turn("请求")
    deps = loop.TurnDeps(client=object(), recorder=recorder, system_prompt="", settings=Settings(),
                         history=[{"role": "user", "content": "旧内容" * 100},
                                  {"role": "user", "content": "当前请求"}], active_history_start=1)
    for _ in range(4):
        collect(compact.maybe_compact(deps, force=True))
    assert attempts == 4 and saved == [True]
    assert "简短摘要" in deps.history[0]["content"]


def test_truncated_compaction_summary_is_rejected():
    client = FakeClient([[{"type": "text_delta", "text": "不完整摘要"},
                          {"type": "stop_reason", "reason": "max_tokens"}]])
    with pytest.raises(ValueError, match="摘要输出达到上限"):
        asyncio.run(compact.summarize_history(client, [{"role": "user", "content": "请求"}]))


def test_trash_rejects_external_directory_link(tmp_path, monkeypatch):
    root, outside = tmp_path / "workspace", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = root / "file.txt"
    source.write_text("保留", encoding="utf-8")
    link = root / ".rubbish"
    try:
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], check=True, capture_output=True)
        monkeypatch.setattr(tools, "workspace_root", lambda: root)
        assert tools.tool_delete_file("file.txt").startswith("Error:")
        assert source.read_text(encoding="utf-8") == "保留"
        assert list(outside.iterdir()) == []
    finally:
        if link.exists():
            os.rmdir(link) if os.name == "nt" else link.unlink()


def test_trash_keeps_file_inside_workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "file.txt").write_text("可恢复", encoding="utf-8")
    monkeypatch.setattr(tools, "workspace_root", lambda: root)
    assert tools.tool_delete_file("file.txt").startswith("已删除")
    destinations = list((root / ".rubbish").iterdir())
    assert len(destinations) == 1 and destinations[0].read_text(encoding="utf-8") == "可恢复"


def test_stream_emits_stop_reason():
    def obj(**kwargs):
        return SimpleNamespace(**kwargs)

    class Stream:
        def __init__(self, **kwargs):
            assert kwargs["temperature"] == 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def __aiter__(self):
            yield obj(type="message_start", message=obj(usage=obj(input_tokens=8, output_tokens=0)))
            yield obj(type="message_delta", usage=obj(output_tokens=5), delta=obj(stop_reason="max_tokens"))
            yield obj(type="message_stop")

    client = object.__new__(AnthropicStreamClient)
    client._client = obj(messages=obj(stream=Stream))
    client.model, client._max_tokens = "fake", 100
    events = collect(client.stream(system="", messages=[], tools=[]))
    snapshots = [event for event in events if event["type"] == "usage"]
    assert [(e["input_tokens"], e["output_tokens"]) for e in snapshots] == [(8, 5)]
    assert {"type": "stop_reason", "reason": "max_tokens"} in events


def test_api_persists_truncated_turn_as_failed(tmp_path, monkeypatch):
    from app import db as database
    from app.api import sessions

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    monkeypatch.setattr(settings, "anthropic_api_key", "fake")
    monkeypatch.setattr(settings, "model_id", "fake")
    database.init_db(tmp_path)
    app = FastAPI()
    app.include_router(sessions.router)
    app.dependency_overrides[sessions.get_llm_client] = lambda: FakeClient([[
        {"type": "text_delta", "text": "部分回复"},
        {"type": "stop_reason", "reason": "max_tokens"}, usage(10, 5)]])
    try:
        with TestClient(app) as client:
            session_id = client.post("/api/sessions").json()["id"]
            response = client.post(f"/api/sessions/{session_id}/turn", json={"text": "请求"})
            assert response.status_code == 200 and '"state": "failed"' in response.text
            turn_data = client.get(f"/api/sessions/{session_id}").json()["turns"][0]
            assert turn_data["state"] == "failed" and turn_data["final_text"] == "部分回复"
            assert any("输出达到上限" in item.get("message", "") for item in turn_data["work_items"])
    finally:
        database.dispose_engine()
