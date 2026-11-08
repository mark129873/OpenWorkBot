import pytest

from app import db
from app.agent import agent_bus
from app.agent.worker import _read_page
from app.config import settings
from app.sessions import chat_store, session_permissions


def test_file_pagination_and_authorization_scope():
    content = "first\n" + "长" * 2001 + "\n" + "x\n" * 2000
    page = _read_page(content, 2, None)
    assert "该行超过 2000" in page and "start_line=2002" in page
    path = "/mnt/user-data/workspace/a.txt"
    first = session_permissions.scope_for("read_file", {"path": path, "start_line": 1}, canonical_path=path)
    second = session_permissions.scope_for("read_file", {"path": path, "start_line": 3}, canonical_path=path)
    assert first == second
    assert session_permissions.scope_for("read_file", {"path": path}, canonical_path=None) is None


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
