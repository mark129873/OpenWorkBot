import sys

import pytest

from app import db
from app.agent import agent_bus, tools
from app.config import settings
from app.sessions import chat_store, session_permissions


def test_file_pagination_and_authorization_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "workspace_root", lambda: tmp_path)
    (tmp_path / "a.txt").write_text("第一行\n" + "长" * 2001 + "\n末行\n", encoding="utf-8")
    assert "继续读取：offset=2" in tools.tool_read_file("a.txt", limit=1)
    page = tools.tool_read_file("a.txt", offset=2, limit=1)
    assert "长" * 2000 in page and "该行超过 2000" in page and "offset=3" in page
    assert "末行" in tools.tool_read_file("a.txt", offset=3)
    assert tools.tool_read_file("a.txt", offset=0).startswith("Error:")
    assert session_permissions.scope_for("read_file", {"path": "a.txt", "offset": 1}) == (
        session_permissions.scope_for("read_file", {"path": "a.txt", "offset": 3, "limit": 1}))
    assert session_permissions.scope_for("read_file", {"path": "a.txt", "other": 1}) != (
        session_permissions.scope_for("read_file", {"path": "a.txt"}))


def test_bounded_output_keeps_head_and_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "workspace_root", lambda: tmp_path)
    popen = tools.subprocess.Popen
    monkeypatch.setattr(tools.subprocess, "Popen", lambda *_args, **kwargs: popen([
        sys.executable, "-c", "import sys; sys.stdout.buffer.write(('开头' + '中' * 400000 + '结尾').encode('utf-8'))"], **kwargs))
    result = tools.capture_bash("ignored")
    assert result.startswith("开头") and result.endswith("结尾") and "中段已截断" in result
    assert len(result.encode("utf-8")) <= 1024 * 1024
    assert len(tools.output_preview(result)) <= 30000


def test_chat_records_and_cycle_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    db.dispose_engine()
    db.init_db(tmp_path)
    try:
        with db.new_session() as session:
            chat_store.initialize(session, tmp_path)
            chat_store.add_visible(session, "user", "你好", message_id="user:one")
            agent_bus.send(session, source="a", target="main", task_id="task", content="报告", message_id="m1")
            session.commit()
        with db.new_session() as session:
            msg = chat_store.records(session, "agent_message", delivery="pending")[0]
            agent_bus.deliver(session, msg)
            agent_bus.deliver(session, msg)
            session.commit()
            assert chat_store.visible(session)[0]["text"] == "你好"
            assert chat_store.history(session, "main") == [{"role": "user", "content": "报告"}]
            with pytest.raises(ValueError):
                agent_bus.send(session, source="a", target="main", task_id="task", content="别的", message_id="m1")
        assert agent_bus.find_cycles([("A", "B"), ("B", "C"), ("C", "A")]) == [["A", "B", "C", "A"]]
    finally:
        db.dispose_engine()
