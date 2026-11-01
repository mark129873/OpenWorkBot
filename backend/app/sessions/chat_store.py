"""主聊天及后台上下文的持久化操作；调用方负责事务与并发。"""

import json

from sqlalchemy import func, select

from app.models import Message, Part, Session, new_id, now_ms
from app.sessions.execution_state import project_directory, project_id


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read(row) -> dict:
    return json.loads(row.data) if row else {}


def initialize(db, data_dir) -> None:
    if db.get(Session, "main") is None:
        db.add(Session(id="main", project_id=project_id(data_dir), directory=project_directory(data_dir),
                       title="主聊天"))
        db.flush()
    if db.get(Message, "main:control") is None:
        db.add(Message(id="main:control", session_id="main", data=dump({"role": "control"})))
        db.flush()
    if get(db, "meta") is None:
        put(db, "meta", {"type": "meta", "revision": 0, "user_seq": 0})
    db.commit()


def get(db, key: str) -> dict | None:
    row = db.get(Part, key)
    return read(row) if row else None


def put(db, key: str, value: dict) -> None:
    row = db.get(Part, key)
    if row is None:
        db.add(Part(id=key, message_id="main:control", session_id="main", data=dump(value)))
    else:
        row.data = dump(value)
        row.time_updated = now_ms()
    db.flush()


def records(db, kind: str, *, context_id: str | None = None,
            delivery: str | None = None) -> list[dict]:
    query = select(Part).where(Part.message_id == "main:control",
                               func.json_extract(Part.data, "$.type") == kind)
    if context_id is not None:
        query = query.where(func.json_extract(Part.data, "$.context_id") == context_id)
    if delivery is not None:
        query = query.where(func.json_extract(Part.data, "$.delivery") == delivery)
    return [read(row) for row in db.scalars(query.order_by(Part.sequence, Part.id))]


def add_visible(db, role: str, text: str, *, message_id: str | None = None,
                status: str = "completed") -> str:
    message_id = message_id or new_id()
    db.add(Message(id=message_id, session_id="main", data=dump({
        "role": role, "visible": True, "status": status,
        "order": get(db, "meta")["revision"] + 1,
    })))
    db.flush()
    db.add(Part(id=f"text:{message_id}", message_id=message_id, session_id="main",
                data=dump({"type": "text", "text": text})))
    db.flush()
    return message_id


def update_visible(db, message_id: str, *, text: str | None = None,
                   status: str | None = None) -> None:
    message = db.get(Message, message_id)
    part = db.get(Part, f"text:{message_id}")
    if message is None or part is None:
        raise ValueError("消息不存在")
    metadata, body = read(message), read(part)
    if status is not None:
        metadata["status"] = status
    if text is not None:
        body["text"] = text
    message.data, part.data = dump(metadata), dump(body)
    message.time_updated = part.time_updated = now_ms()
    db.flush()


def visible(db) -> list[dict]:
    rows = db.execute(select(Message, Part).join(Part, Part.message_id == Message.id).where(
        Message.session_id == "main", func.json_extract(Message.data, "$.visible") == 1,
        func.json_extract(Part.data, "$.type") == "text").order_by(Message.sequence, Message.id)).all()
    return [{"id": message.id, "role": read(message)["role"], "text": read(part)["text"],
             "status": read(message).get("status", "completed"), "time": message.time_created,
             "order": read(message).get("order", 0)}
            for message, part in rows]


def append_history(db, context_id: str, role: str, content, *, key: str | None = None) -> str:
    key = key or new_id()
    existing = get(db, f"history:{key}")
    value = {"type": "agent_context", "id": key, "context_id": context_id,
             "role": role, "content": content}
    if existing is not None and existing != value:
        raise ValueError("相同上下文消息 ID 的内容不一致")
    if existing is None:
        put(db, f"history:{key}", value)
    return key


def history(db, context_id: str) -> list[dict]:
    return [{"role": row["role"], "content": row["content"]}
            for row in records(db, "agent_context", context_id=context_id)]


def bump(db, *, user_event: bool = False) -> dict:
    meta = get(db, "meta")
    meta["revision"] += 1
    if user_event:
        meta["user_seq"] += 1
    put(db, "meta", meta)
    return meta
