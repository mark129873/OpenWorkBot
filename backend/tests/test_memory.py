import asyncio
import json
import subprocess
import sys

import pytest
import tiktoken
import yaml

from app import db
from app.agent import memory as memory_module
from app.agent.memory import Memory, _apply, _conversation, _parse, _signals
from app.agent.memory_recall import _rank, _snapshot_data
from app.agent.memory_store import MemoryRevisionConflict, _advance, _directory, _read_files, _state, _store, empty_memory
from app.config import settings
from app.models import User
from app.user_context import bind_user


USER = "a" * 32


@pytest.fixture
def memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    return settings.data_dir


def _fact(content, **overrides):
    return {"content": content, "confidence": 0.9, "category": "preference", "scope": "user",
            "durability": "durable", "authority": "descriptive", **overrides}


def _result(**overrides):
    return {"user": {}, "history": {}, "newFacts": [], "factsToRemove": [], **overrides}


def test_storage_uses_structured_summary_and_one_markdown_file_per_fact(memory_dir):
    summary = empty_memory()
    summary["user"]["personalContext"]["summary"] = "简洁中文"
    facts = [{"id": "one", "content": "喜欢咖啡", "category": "preference", "confidence": 0.9},
             {"id": "two", "content": "使用 Python", "category": "knowledge", "confidence": 0.8}]
    _store(USER, summary, facts)
    directory = _directory(USER)
    saved = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
    files = list((directory / "agents" / "__default__" / "facts").glob("*/*.md"))
    assert saved["user"]["personalContext"]["summary"] == "简洁中文" and "facts" not in saved
    assert len(files) == 2 and all(path.read_text(encoding="utf-8").startswith("---\n") for path in files)
    metadata = yaml.safe_load(files[0].read_text(encoding="utf-8").split("---", 2)[1])
    assert metadata["schemaVersion"] == 2 and metadata["revision"] == 1 and metadata["status"] == "active"
    assert metadata["topics"] == [] and metadata["consolidatedFrom"] == []
    assert metadata["source"] == {"type": "unknown", "threadId": None}
    assert {item["content"] for item in _read_files(USER)[1]} == {"喜欢咖啡", "使用 Python"}
    assert not (memory_dir / "memory-index.sqlite").exists()


def test_watermark_and_fact_versions_follow_deerflow_format(memory_dir):
    _store(USER, empty_memory(), [], 0)
    assert not (_directory(USER) / "memory.json").exists()
    _store(USER, empty_memory(), [{"id": "one", "content": "喜欢咖啡", "confidence": 0.9, "source": "main"}])
    summary, facts = _read_files(USER)
    assert facts[0]["source"] == {"type": "conversation", "threadId": "main"}
    _store(USER, summary, facts, summary["revision"])
    assert _read_files(USER)[0]["revision"] == summary["revision"]
    facts[0]["content"] = "喜欢茶"
    _store(USER, summary, facts, summary["revision"])
    updated, saved = _read_files(USER)
    assert updated["revision"] == summary["revision"] + 1 and saved[0]["revision"] == 2
    _advance(USER, "answer")
    path = _directory(USER) / "agents" / "__default__" / ".metadata" / "watermarks.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"main": ["id", "answer"]}


def test_invalid_fact_does_not_damage_existing_memory(memory_dir):
    _store(USER, empty_memory(), [{"id": "one", "content": "保留", "confidence": 0.9}])
    before = _read_files(USER)
    with pytest.raises(ValueError):
        _store(USER, before[0], [{"id": "../unsafe", "content": "无效", "confidence": 0.9}])
    assert _read_files(USER) == before
    assert not (_directory(USER) / ".memory.journal.json").exists()


def test_stale_memory_revision_cannot_overwrite_newer_update(memory_dir):
    _store(USER, empty_memory(), [{"id": "one", "content": "旧值", "confidence": 0.9}])
    stale, facts = _read_files(USER)
    _store(USER, stale, [{"id": "one", "content": "新值", "confidence": 0.9}], stale["revision"])
    with pytest.raises(ValueError, match="记忆已更新"):
        _store(USER, stale, facts, stale["revision"])
    assert _read_files(USER)[1][0]["content"] == "新值"


@pytest.mark.parametrize("phase", ["prepared", "committed"])
def test_storage_recovers_after_process_exit(memory_dir, phase):
    summary = empty_memory()
    summary["user"]["personalContext"]["summary"] = "原摘要"
    _store(USER, summary, [{"id": "one", "content": "原事实", "confidence": 0.9}])
    code = r'''
import json, os, sys
from pathlib import Path
from app.config import settings
from app.agent import memory_store
settings.data_dir = Path(sys.argv[1])
user, phase = sys.argv[2:]
summary, facts = memory_store._read_files(user)
summary["user"]["personalContext"]["summary"] = "新摘要"
facts[0]["content"] = "新事实"
write = memory_store._atomic_write
def crash(path, value):
    write(path, value)
    if phase == "prepared" and path == memory_store._directory(user) / "memory.json":
        os._exit(81)
    if phase == "committed" and path.name == ".memory.journal.json" and json.loads(value)["state"] == "committed":
        os._exit(82)
memory_store._atomic_write = crash
memory_store._store(user, summary, facts, summary["revision"])
'''
    child = subprocess.run([sys.executable, "-c", code, str(memory_dir), USER, phase], check=False)
    assert child.returncode == (81 if phase == "prepared" else 82)
    restored, facts = _read_files(USER)
    expected = "原" if phase == "prepared" else "新"
    assert restored["user"]["personalContext"]["summary"] == expected + "摘要"
    assert facts[0]["content"] == expected + "事实"
    assert not (_directory(USER) / ".memory.journal.json").exists()


def test_recovery_does_not_overwrite_memory_when_backup_is_missing(memory_dir):
    _store(USER, empty_memory(), [{"id": "one", "content": "保留", "confidence": 0.9}])
    directory = _directory(USER)
    summary_path = directory / "memory.json"
    before = summary_path.read_text(encoding="utf-8")
    fact_path = next((directory / "agents" / "__default__" / "facts").glob("*/*.md"))
    backup = directory / ".recovery" / "abc"
    backup.mkdir(parents=True)
    (backup / "memory.json").write_text(before, encoding="utf-8")
    journal = {"operationId": "abc", "state": "prepared", "agentName": "__default__",
               "expectedRevision": 1, "nextRevision": 2, "factIds": ["one"],
               "oldEntries": {"one": {"path": fact_path.relative_to(directory).as_posix()}}}
    (directory / ".memory.journal.json").write_text(json.dumps(journal), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        _read_files(USER)
    assert summary_path.read_text(encoding="utf-8") == before
    assert "保留" in fact_path.read_text(encoding="utf-8")


def test_conversation_filters_internal_tool_upload_and_trivial_messages():
    records = [
        {"id": "before", "role": "assistant", "content": "旧回复"},
        {"id": "input:old", "role": "user", "content": "启用前"},
        {"id": "old-answer", "role": "assistant", "content": "旧答复"},
        {"id": "input:new", "role": "user", "content": "用户文本和附件路径"},
        {"id": "tool-call", "role": "assistant", "content": [{"type": "tool_use", "id": "x"}]},
        {"id": "tool-result", "role": "user", "content": [{"type": "tool_result", "content": "工具结果"}]},
        {"id": "bus:1", "role": "user", "content": "内部消息"},
        {"id": "failed:1", "role": "assistant", "content": "回复失败"},
        {"id": "answer", "role": "assistant", "content": [{"type": "text", "text": "最终答复"}]},
        {"id": "input:upload", "role": "user", "content": "上传路径"},
        {"id": "upload-answer", "role": "assistant", "content": "上传完成"},
        {"id": "input:ack", "role": "user", "content": "好的！"},
        {"id": "ack-answer", "role": "assistant", "content": "明白"},
    ]
    events = {"old": {"user_seq": 3, "user_text": "启用前"},
              "new": {"user_seq": 4, "user_text": "保留正文"},
              "upload": {"user_seq": 5, "user_text": ""},
              "ack": {"user_seq": 6, "user_text": "好的！"}}
    assert [item["id"] for item in _conversation(records, events, {})][:2] == ["input:old", "old-answer"]
    messages = _conversation(records, events, {"main": ["id", "old-answer"]})
    assert [(item["id"], item["content"]) for item in messages] == [
        ("input:new", "保留正文"), ("answer", "最终答复")]
    assert _conversation(records, events, {"main": ["id", "answer"]}) == []


def test_scope_filter_and_safe_fact_replacement():
    old = {"id": "old", "content": "喜欢喝茶", "category": "preference", "confidence": 0.9}
    removal = {"id": "old", "scope": "user", "reason": "用户明确更正", "replacementFactIndex": 0}
    result = _result(newFacts=[_fact("喜欢咖啡", scope="project"), _fact("仅本次", durability="temporary"),
                               _fact("允许删除", authority="transactional")], factsToRemove=[removal])
    _, kept = _apply(empty_memory(), [old], result)
    assert [item["content"] for item in kept] == ["喜欢喝茶"]
    result["newFacts"][0] = _fact("喜欢咖啡")
    _, kept = _apply(empty_memory(), [old], result)
    assert [item["content"] for item in kept] == ["喜欢咖啡"]
    assert _apply(empty_memory(), [old], _result(factsToRemove=[{**removal, "scope": "project"}]))[1] == [old]


def test_invalid_new_fact_blocks_deletion_and_empty_summary_does_not_clear():
    old = {"id": "old", "content": "旧事实", "confidence": 0.9}
    summary = empty_memory()
    summary["user"]["workContext"]["summary"] = "保留摘要"
    removal = {"id": "old", "scope": "user", "reason": "用户更正"}
    update = _result(newFacts=[_fact("新事实", confidence="invalid")], factsToRemove=[removal])
    with pytest.raises(ValueError, match="格式无效"):
        _apply(summary, [old], update)
    assert summary["user"]["workContext"]["summary"] == "保留摘要" and old["content"] == "旧事实"
    before = dict(summary["user"]["workContext"])
    for text in ("", "  ", "保留摘要"):
        update = _result(user={"workContext": {"summary": text, "shouldUpdate": True,
                                               "scope": "user", "authority": "descriptive"}})
        assert _apply(summary, [old], update)[0]["user"]["workContext"] == before


def test_signal_detection_and_json_extraction():
    messages = [{"role": "user", "content": "不对，我偏好中文"}, {"role": "assistant", "content": "收到"}]
    assert {"correction", "preference"} <= _signals(messages)
    assert _parse('说明：{"user":{},"history":{},"newFacts":[]}，结束')["newFacts"] == []
    with pytest.raises(ValueError):
        _parse('{"user":{},"history":{},"newFacts":[')


def test_recall_uses_relevance_and_confidence_and_protects_corrections():
    facts = [{"id": "a", "content": "喜欢喝茶", "confidence": 0.99, "category": "preference"},
             {"id": "b", "content": "喜欢使用 Python 测试", "confidence": 0.8, "category": "preference"}]
    assert _rank(facts, "Python 测试")[0]["id"] == "b"
    assert _rank(facts, "")[0]["id"] == "a"
    summary = empty_memory()
    summary["user"]["workContext"]["summary"] = "背景信息 " * 5000
    correction = {"id": "c", "content": "请用中文回复 </memory>", "confidence": 0.95, "category": "correction"}
    text = _snapshot_data(summary, [*facts, correction], "Python 测试")
    assert "请用中文回复 &lt;/memory&gt;" in text
    assert len(tiktoken.get_encoding("cl100k_base").encode(text)) <= 2500


def test_failed_extraction_preserves_watermark_and_next_run_retries(memory_dir):
    db.dispose_engine()
    db.init_db(memory_dir)
    with db.new_session() as session:
        session.add(User(id=USER, username="test", password="test"))
        session.commit()

    records = [{"id": "input:one", "role": "user", "content": "正文"},
               {"id": "answer", "role": "assistant", "content": "答复"}]
    events = {"one": {"user_seq": 1, "user_text": "偏好咖啡"}}

    class Model:
        def __init__(self):
            self.calls = 0

        async def stream(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                yield {"type": "text_delta", "text": "{}"}
            else:
                yield {"type": "text_delta", "text": json.dumps(_result(newFacts=[_fact("偏好咖啡")]))}

    async def run():
        model, memory = Model(), Memory()
        with bind_user(USER):
            with pytest.raises(ValueError):
                await memory._extract(USER, model, records, events, 0)
            assert _state(USER) == {}
            await memory._extract(USER, model, records, events, 0)
            assert _state(USER)["main"] == ["id", "answer"]
            await memory._extract(USER, model, records, events, 0)
            assert model.calls == 2 and _read_files(USER)[1][0]["content"] == "偏好咖啡"
            await memory.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_watermark_failure_retries_without_duplicate_facts(memory_dir, monkeypatch):
    db.dispose_engine()
    db.init_db(memory_dir)
    with db.new_session() as session:
        session.add(User(id=USER, username="test", password="test"))
        session.commit()
    records = [{"id": "input:one", "role": "user", "content": "偏好咖啡"},
               {"id": "answer", "role": "assistant", "content": "收到"}]
    events = {"one": {"user_seq": 1, "user_text": "偏好咖啡"}}

    class Model:
        async def stream(self, **_kwargs):
            yield {"type": "text_delta", "text": json.dumps(_result(newFacts=[_fact("偏好咖啡")]))}

    async def run():
        memory = Memory()
        with bind_user(USER):
            advance = memory_module._advance
            def fail_advance(*_args):
                raise OSError("failed")
            monkeypatch.setattr(memory_module, "_advance", fail_advance)
            with pytest.raises(OSError):
                await memory._extract(USER, Model(), records, events, 0)
            assert _state(USER) == {}
            monkeypatch.setattr(memory_module, "_advance", advance)
            await memory._extract(USER, Model(), records, events, 0)
            assert _state(USER)["main"] == ["id", "answer"]
            assert [fact["content"] for fact in _read_files(USER)[1]] == ["偏好咖啡"]
            await memory.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_extraction_reapplies_corrections_after_revision_conflict(memory_dir, monkeypatch):
    db.dispose_engine()
    db.init_db(memory_dir)
    with db.new_session() as session:
        session.add(User(id=USER, username="test", password="test"))
        session.commit()
    records = [{"id": "input:one", "role": "user", "content": "不对，我偏好中文"},
               {"id": "answer", "role": "assistant", "content": "我会用中文回复"}]
    events = {"one": {"user_text": "不对，我偏好中文"}}
    original_store, attempts = memory_module._store, []

    def conflict_once(user, summary, facts, revision):
        attempts.append(revision)
        if len(attempts) == 1:
            current, current_facts = _read_files(user)
            current["user"]["workContext"]["summary"] = "并发更新"
            original_store(user, current, current_facts, current["revision"])
            raise MemoryRevisionConflict("记忆已更新，请重试")
        original_store(user, summary, facts, revision)

    class Model:
        calls = 0

        async def stream(self, **kwargs):
            self.calls += 1
            assert "检测到纠错" in kwargs["system"] and "sourceError" in kwargs["system"]
            assert "七类摘要正文" in kwargs["system"] and "1000 token" in kwargs["system"]
            assert kwargs["tools"] == []
            fact = _fact("偏好中文", category="correction", sourceError="此前使用英文")
            yield {"type": "text_delta", "text": "结果：" + json.dumps(_result(newFacts=[fact]))}

    async def run():
        monkeypatch.setattr(memory_module, "_store", conflict_once)
        model, memory = Model(), Memory()
        with bind_user(USER):
            await memory._extract(USER, model, records, events, 0)
            summary, facts = _read_files(USER)
            assert summary["user"]["workContext"]["summary"] == "并发更新"
            assert facts[0]["sourceError"] == "此前使用英文" and facts[0]["category"] == "correction"
            assert model.calls == 1 and attempts == [0, 1] and _state(USER)["main"] == ["id", "answer"]
            await memory.close()

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_debounce_has_maximum_wait_while_messages_keep_arriving(monkeypatch):
    monkeypatch.setattr(memory_module, "_DEBOUNCE", 0.09)
    monkeypatch.setattr(memory_module, "_MAX_WAIT", 0.18)

    async def run():
        memory, processed = Memory(), asyncio.Event()
        async def extract(_user, _client, records, _events, _generation, _signals):
            processed.set()

        monkeypatch.setattr(memory, "_extract", extract)
        with bind_user(USER):
            async def send():
                for index in range(20):
                    memory.enqueue(None, [{"id": str(index)}], {}, 0, role="main")
                    await asyncio.sleep(0.04)

            sender = asyncio.create_task(send())
            await asyncio.wait_for(processed.wait(), 1)
            assert not sender.done()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
            await memory.close()

    asyncio.run(run())


def test_different_users_are_processed_serially_and_shutdown_drains(monkeypatch):
    monkeypatch.setattr(memory_module, "_DEBOUNCE", 0.01)
    monkeypatch.setattr(memory_module, "_BATCH_DELAY", 0)
    monkeypatch.setattr(memory_module, "_SHUTDOWN_WAIT", 1)

    async def run():
        memory, first_started, release = Memory(), asyncio.Event(), asyncio.Event()
        started = []
        async def extract(user, _client, _records, _events, _generation, _signals):
            started.append(user)
            if len(started) == 1:
                first_started.set()
                await release.wait()

        monkeypatch.setattr(memory, "_extract", extract)
        with bind_user("a" * 32):
            memory.enqueue(None, [], {}, 0, role="main")
        with bind_user("b" * 32):
            memory.enqueue(None, [], {}, 0, role="main")
        await asyncio.wait_for(first_started.wait(), 1)
        assert started == ["a" * 32]
        release.set()
        await memory.close()
        assert started == ["a" * 32, "b" * 32] and not memory.pending

    asyncio.run(run())


def test_queued_extraction_keeps_users_isolated(memory_dir, monkeypatch):
    monkeypatch.setattr(memory_module, "_DEBOUNCE", 0.01)
    monkeypatch.setattr(memory_module, "_BATCH_DELAY", 0)
    db.dispose_engine()
    db.init_db(memory_dir)
    users = ("a" * 32, "b" * 32)
    with db.new_session() as session:
        session.add_all([User(id=user, username=user, password="test") for user in users])
        session.commit()

    class Model:
        async def stream(self, **kwargs):
            conversation = json.loads(kwargs["messages"][0]["content"])["messages"]
            if conversation[0]["content"].startswith("不对"):
                assert "检测到纠错" in kwargs["system"]
            yield {"type": "text_delta", "text": json.dumps(_result(newFacts=[_fact(conversation[-2]["content"])]))}

    async def run():
        memory = Memory()
        for index, user in enumerate(users):
            text = f"用户偏好 {index}"
            first = f"不对，{text}" if index == 0 else text
            records = [{"id": "input:one", "role": "user", "content": first},
                       {"id": "answer", "role": "assistant", "content": "收到"}]
            events = {"one": {"user_text": first}}
            with bind_user(user):
                memory.enqueue(Model(), records, events, 0, role="main")
                if index == 0:
                    records = records.copy()
                    for number in range(3):
                        records.extend([{"id": f"input:{number}", "role": "user", "content": text},
                                        {"id": f"answer:{number}", "role": "assistant", "content": "收到"}])
                        events[str(number)] = {"user_text": text}
                    memory.enqueue(Model(), records, events, 0, role="main")
        await memory.close()
        for index, user in enumerate(users):
            assert [fact["content"] for fact in _read_files(user)[1]] == [f"用户偏好 {index}"]
            assert _state(user)["main"] == ["id", "answer:2" if index == 0 else "answer"]

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()


def test_shutdown_stops_waiting_after_deadline(monkeypatch):
    monkeypatch.setattr(memory_module, "_DEBOUNCE", 0)
    monkeypatch.setattr(memory_module, "_SHUTDOWN_WAIT", 0.05)

    async def run():
        memory, started, cancelled = Memory(), asyncio.Event(), asyncio.Event()
        async def extract(_user, _client, _records, _events, _generation, _signals):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(memory, "_extract", extract)
        with bind_user(USER):
            memory.enqueue(None, [], {}, 0, role="main")
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(memory.close(), 1)
        assert cancelled.is_set()

    asyncio.run(run())


def test_full_queue_still_accepts_signal_without_overtaking(monkeypatch):
    monkeypatch.setattr(memory_module, "_MAX_PENDING", 1)
    monkeypatch.setattr(memory_module, "_DEBOUNCE", 100)

    async def run():
        memory, started = Memory(), []
        async def extract(user, _client, _records, _events, _generation, _signals):
            started.append(user)

        monkeypatch.setattr(memory, "_extract", extract)
        with bind_user("a" * 32):
            memory.enqueue(None, [], {}, 0, role="main")
        with bind_user("b" * 32):
            memory.enqueue(None, [], {}, 0, role="main")
        with bind_user("c" * 32):
            memory.enqueue(None, [{"id": "input:one", "role": "user", "content": "不对"}],
                           {"one": {"user_text": "不对"}}, 0, role="main")
        assert list(memory.pending) == ["a" * 32, "c" * 32]
        await memory.close()
        assert started == ["a" * 32, "c" * 32]

    asyncio.run(run())
