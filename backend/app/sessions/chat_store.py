"""主聊天及后台上下文的持久化操作；调用方负责事务与并发。"""

import json
from copy import deepcopy

from sqlalchemy import delete, select, update

from app.models import AgentMessage, Message, Part, Session, SharedState, TurnUsage, new_id, now_ms
from app.user_context import require_user, scoped, unscoped


def dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read(row) -> dict:
    if not row:
        return {}
    value = {**deepcopy(row.data), "type": "agent_message" if isinstance(row, AgentMessage) else row.type}
    if row.context_id is not None:
        value["context_id"] = row.context_id
    return value


def initialize(db, data_dir) -> None:
    session_id = scoped("main")
    if db.get(Session, session_id) is None:
        db.add(Session(id=session_id, user_id=require_user()))
        db.flush()
    db.commit()


def clear(db) -> None:
    session_id = scoped("main")
    db.execute(delete(Part).where(Part.session_id == session_id))
    db.execute(delete(AgentMessage).where(AgentMessage.session_id == session_id))
    db.execute(delete(SharedState).where(SharedState.session_id == session_id))
    db.execute(delete(Message).where(Message.session_id == session_id))
    db.execute(delete(TurnUsage).where(TurnUsage.session_id == session_id))


def get(db, key: str) -> dict | None:
    if key == "meta":
        row = db.get(Session, scoped("main"))
        if row is None:
            return None
        return {**deepcopy(row.runtime_data), "type": "meta", "revision": row.revision, "user_seq": row.user_seq,
                "assistant_id": row.assistant_id, "reset": deepcopy(row.reset_state),
                "last_reset": deepcopy(row.last_reset)}
    row = db.get(AgentMessage if key.startswith("bus:") else Part, scoped(key))
    return read(row) if row else None


def put(db, key: str, value: dict) -> None:
    if key == "meta":
        row = db.get(Session, scoped("main"))
        row.revision = value["revision"]
        row.user_seq = value["user_seq"]
        row.assistant_id = value.get("assistant_id") or row.assistant_id
        row.reset_state = deepcopy(value.get("reset"))
        row.last_reset = deepcopy(value.get("last_reset"))
        row.runtime_data = deepcopy({k: v for k, v in value.items()
                                     if k not in {"type", "revision", "user_seq", "assistant_id", "reset", "last_reset"}})
        row.time_updated = now_ms()
    else:
        model = AgentMessage if value.get("type") == "agent_message" else Part
        if key.startswith("bus:") != (model is AgentMessage):
            raise ValueError("内部消息键或类型无效")
        row = db.get(model, scoped(key))
        payload = deepcopy({k: v for k, v in value.items() if k not in {"type", "context_id"}})
        if row is None:
            fields = ({"task_id": value.get("task_id"), "kind": value.get("kind"), "delivery": value["delivery"]}
                      if model is AgentMessage else {"type": value["type"]})
            row = model(key=scoped(key), session_id=scoped("main"), sequence=_next(db, "next_part_sequence"),
                        context_id=value.get("context_id"), data=payload, **fields)
            db.add(row)
        else:
            if model is AgentMessage:
                row.task_id, row.kind, row.delivery = value.get("task_id"), value.get("kind"), value["delivery"]
            else:
                row.type = value["type"]
            row.context_id = value.get("context_id")
            row.data = payload
            row.time_updated = now_ms()
    db.flush()


def _next(db, name: str) -> int:
    column = getattr(Session, name)
    return db.execute(update(Session).where(Session.id == scoped("main"))
                      .values({name: column + 1}).returning(column)
                      .execution_options(synchronize_session=False)).scalar_one()


def next_display_order(db) -> int:
    return _next(db, "next_display_order")


def records(db, kind: str, *, context_id: str | None = None,
            delivery: str | None = None) -> list[dict]:
    model = AgentMessage if kind == "agent_message" else Part
    query = select(model).where(model.session_id == scoped("main"))
    if model is Part:
        query = query.where(Part.type == kind)
    if context_id is not None:
        query = query.where(model.context_id == context_id)
    if delivery is not None:
        column = AgentMessage.delivery if model is AgentMessage else Part.data["delivery"].as_string()
        query = query.where(column == delivery)
    return [read(row) for row in db.scalars(query.order_by(model.sequence, model.key))]


def add_visible(db, role: str, text: str, *, message_id: str | None = None,
                status: str = "completed", attachments: list | None = None) -> str:
    message_id = message_id or new_id()
    db.add(Message(id=scoped(message_id), session_id=scoped("main"),
                   sequence=_next(db, "next_message_sequence"), display_order=next_display_order(db),
                   role=role, text=text, status=status, attachments=attachments or []))
    db.flush()
    return message_id


def update_visible(db, message_id: str, *, text: str | None = None,
                   status: str | None = None) -> None:
    message = db.get(Message, scoped(message_id))
    if message is None:
        raise ValueError("消息不存在")
    if status is not None:
        message.status = status
    if text is not None:
        message.text = text
    message.time_updated = now_ms()
    db.flush()


def _visible_rows(rows) -> list[dict]:
    return [{"id": unscoped(message.id), "role": message.role, "text": message.text,
             "attachments": message.attachments, "status": message.status, "time": message.time_created,
             "order": message.display_order} for message in rows]


def visible(db) -> list[dict]:
    rows = db.scalars(select(Message).where(Message.session_id == scoped("main"))
                      .order_by(Message.sequence)).all()
    return _visible_rows(rows)


def visible_page(db, limit: int, before: int | None = None) -> tuple[list[dict], int | None]:
    query = select(Message).where(Message.session_id == scoped("main"))
    if before is not None:
        query = query.where(Message.sequence < before)
    rows = db.scalars(query.order_by(Message.sequence.desc()).limit(limit + 1)
                      .execution_options(populate_existing=True)).all()
    page = list(reversed(rows[:limit]))
    return _visible_rows(page), page[0].sequence if len(rows) > limit and page else None


def stage_context(db, context_id: str, role: str, content, *, key: str | None = None) -> str:
    key = key or new_id()
    existing = get(db, f"context-pending:{key}")
    value = {"type": "context_pending", "id": key, "context_id": context_id,
             "role": role, "content": content}
    if existing is not None and existing != value:
        raise ValueError("相同上下文消息 ID 的内容不一致")
    if existing is None:
        put(db, f"context-pending:{key}", value)
    return key


def clear_pending(db, context_id: str, identities: list[str]) -> None:
    for identity in identities:
        row = db.get(Part, scoped(f"context-pending:{identity}"))
        if row and row.type == "context_pending" and row.context_id == context_id:
            db.delete(row)


def bump(db, *, user_event: bool = False) -> dict:
    meta = get(db, "meta")
    meta["revision"] += 1
    if user_event:
        meta["user_seq"] += 1
    put(db, "meta", meta)
    return meta
