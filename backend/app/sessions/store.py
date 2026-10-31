"""按 ID 更新；仅首次插入分配顺序，避免时间线漂移。"""

import json

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session as DbSession

from app.models import Message, Part, Session, Todo, TurnUsage, new_id, now_ms


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _load(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def touch_session(db: DbSession, session_id: str) -> None:
    row = db.get(Session, session_id)
    if row is not None:
        row.time_updated = now_ms()


def _next_sequence(db: DbSession, model, session_id: str) -> int:
    current = db.execute(
        select(func.max(model.sequence)).where(model.session_id == session_id)
    ).scalar()
    return (current if current is not None else -1) + 1


def ensure_session(
    db: DbSession, session_id: str, model: str, first_user_text: str
) -> Session:
    row = db.get(Session, session_id)
    if row is not None:
        return row
    from app.config import settings
    from app.sessions.execution_state import project_id, project_directory
    row = Session(
        project_id=project_id(settings.data_dir), directory=project_directory(settings.data_dir),
        id=session_id,
        title=(first_user_text or "新会话").replace("\n", " ")[:30],
        time_created=now_ms(),
        time_updated=now_ms(),
    )
    db.add(row)
    db.flush()
    db.commit()
    return row


def get_session(db: DbSession, session_id: str) -> Session | None:
    row = db.get(Session, session_id)
    if row is None or row.time_archived is not None:
        return None
    return row


def is_session_archived(db: DbSession, session_id: str) -> bool:
    """区分软删与未落库草稿，防止旧请求恢复已删会话。"""
    row = db.get(Session, session_id)
    return row is not None and row.time_archived is not None


def list_sessions(db: DbSession, limit: int = 200) -> list[Session]:
    stmt = (
        select(Session)
        .where(Session.time_archived.is_(None))
        .order_by(Session.time_updated.desc())
        .limit(limit)
    )
    return list(db.execute(stmt).scalars())


def soft_delete_session(db: DbSession, session_id: str) -> bool:
    """用户侧永久删除，数据库仅标记归档。"""
    row = db.get(Session, session_id)
    if row is None or row.time_archived is not None:
        return False
    row.time_archived = now_ms()
    db.commit()
    return True


def turn_id(row) -> str:
    return _load(row.data).get("anchor", {}).get("turnId", "")


def message_role(row: Message) -> str:
    return _load(row.data).get("role", "")


def part_kind(row: Part) -> str:
    kind = _load(row.data).get("type", "")
    return "tool_call" if kind == "tool" else kind


def part_data(row: Part) -> dict:
    """返回工具投影；修改后须显式保存。"""
    data = _load(row.data)
    if data.get("type") == "tool":
        state = data.get("state", {})
        return {**state, "output": state.get("output", state.get("error", "")), "name": data.get("tool", ""),
                "tool_call_id": data.get("callID", row.id)}
    return data


def _tag(data: dict, label: str) -> dict:
    result = dict(data)
    result.pop("turn_id", None)
    result["anchor"] = {**result.get("anchor", {}), "turnId": label}
    return result


def upsert_message(db: DbSession, session_id: str, message_id: str,
                   role: str, data: dict, turn_id: str, *, commit: bool = True) -> Message:
    """同会话更新保序，改归属时重新分配序号。"""
    if "text" in data:
        raise ValueError("正文必须通过 text part 保存")
    payload = _tag({**data, "role": role}, turn_id)
    payload.setdefault("time", {"created": now_ms()})
    row = db.get(Message, message_id)
    if row is None:
        row = Message(id=message_id, session_id=session_id,
                      sequence=_next_sequence(db, Message, session_id), data=_dump(payload),
                      time_created=payload["time"]["created"])
        db.add(row)
    else:
        if row.session_id != session_id:
            row.sequence = _next_sequence(db, Message, session_id)
            row.session_id = session_id
        payload["time"]["created"] = row.time_created
        row.data = _dump(payload)
    row.time_updated = payload["time"].get("completed", now_ms()) if role == "assistant" else row.time_created
    touch_session(db, session_id)
    db.flush()
    if commit:
        db.commit()
    return row


def upsert_part(db: DbSession, session_id: str, message_id: str, part_id: str,
                kind: str, data: dict, turn_id: str = "", *, commit: bool = True) -> Part:
    """写入前校验父消息与会话，防止跨会话混入。"""
    parent = db.get(Message, message_id)
    if parent is None or parent.session_id != session_id:
        raise ValueError("部件所属消息不存在或不属于该会话")
    if turn_id and _load(parent.data).get("anchor", {}).get("turnId", "") != turn_id:
        raise ValueError("部件轮次与所属消息不一致")
    if kind == "tool_call":
        old = db.get(Part, part_id)
        old_state = _load(old.data).get("state", {}) if old else {}
        status = data.get("status", "pending")
        status = "error" if status in ("failed", "denied") else status
        state = {"status": status, "input": data.get("input") or {}}
        if status == "pending":
            state["raw"] = _dump(state["input"])
        else:
            state["time"] = {"start": old_state.get("time", {}).get("start", now_ms())}
            # 停止也是终态，保留已收到的子任务输出。
            if status in ("completed", "error", "stopped"):
                state["time"]["end"] = now_ms()
                if status == "error":
                    state["error"] = data.get("error", data.get("output", "")) or ""
                else:
                    state.update(output=data.get("output") or "", title=data.get("name", ""), metadata={})
        payload = {"type": "tool", "callID": data.get("tool_call_id", part_id),
                   "tool": data.get("name", ""), "state": state}
    else:
        if kind not in ("text", "compaction", "timeline"):
            raise ValueError(f"未实现的 ZCode 部件类型：{kind}")
        payload = {**data, "type": kind}
    row = db.get(Part, part_id)
    if row is None:
        current = db.execute(select(func.max(Part.sequence)).where(Part.message_id == message_id)).scalar()
        row = Part(id=part_id, message_id=message_id, session_id=session_id,
                   sequence=(current if current is not None else -1) + 1, data=_dump(payload))
        db.add(row)
    else:
        if row.message_id != message_id or row.session_id != session_id:
            current = db.scalar(select(func.max(Part.sequence)).where(Part.message_id == message_id))
            row.sequence = (current if current is not None else -1) + 1
            row.message_id, row.session_id = message_id, session_id
        row.data = _dump(payload)
    row.time_updated = now_ms()
    touch_session(db, session_id)
    db.flush()
    if commit:
        db.commit()
    return row


def save_user_message(db: DbSession, session_id: str, message_id: str,
                      text: str, turn_id: str, model: str = "") -> Message:
    """用户元信息与正文原子提交。"""
    try:
        row = upsert_message(db, session_id, message_id, "user", {}, turn_id, commit=False)
        # 固定用户正文 ID 保证重试幂等。
        upsert_part(db, session_id, message_id, message_id, "text", {"text": text}, turn_id, commit=False)
        db.commit()
        return row
    except Exception:
        db.rollback()
        raise


def find_last_user_message(db: DbSession, session_id: str) -> Message | None:
    return db.execute(select(Message).where(
        Message.session_id == session_id, func.json_extract(Message.data, "$.role") == "user",
        func.coalesce(func.json_extract(Message.data, "$.synthetic"), 0) == 0
    ).order_by(Message.sequence.desc()).limit(1)).scalar()


def rollback_turn(db: DbSession, session_id: str, turn_id: str, from_sequence: int) -> int:
    """重跑保留用户正文，清理助手内容和本轮事实。"""
    result = db.execute(delete(Message).where(
        Message.session_id == session_id,
        func.json_extract(Message.data, "$.role") == "assistant",
        Message.sequence >= from_sequence,
    ))
    db.execute(delete(Message).where(
        Message.session_id == session_id,
        func.json_extract(Message.data, "$.synthetic") == 1,
        func.json_extract(Message.data, "$.anchor.turnId") == turn_id,
    ))
    db.execute(delete(TurnUsage).where(TurnUsage.session_id == session_id, TurnUsage.turn_id == turn_id))
    from app.sessions.approvals import forget_turn
    forget_turn(session_id, turn_id)
    # 从保留的成功工具输入恢复任务板，清除已删轮次副作用。
    rows = db.scalars(select(Part).join(Message, Part.message_id == Message.id).where(
        Part.session_id == session_id).order_by(Message.sequence, Part.sequence)).all()
    items = []
    for part in rows:
        data = part_data(part)
        if data.get("name") == "todo_write" and data.get("status") == "completed":
            from app.agent.todo import parse_items
            parsed, error = parse_items(data.get("input", {}).get("items"))
            if not error:
                items = parsed
    replace_todos(db, session_id, items, commit=False)
    db.commit()
    return result.rowcount or 0


def retag_message(db: DbSession, row: Message, label: str) -> None:
    """重跑仅改消息标签，正文随 message_id 归属。"""
    row.data = _dump(_tag(_load(row.data), label))
    row.time_updated = now_ms()
    db.commit()


def list_turns(db: DbSession, session_id: str) -> list[TurnUsage]:
    return list(db.scalars(select(TurnUsage).where(TurnUsage.session_id == session_id).order_by(TurnUsage.started_at)))


def turn_fact(row: TurnUsage) -> dict:
    return {"turn_id": row.turn_id, "state": {"completed": "success", "error": "failed", "cancelled": "stopped"}.get(row.status, row.status),
            "started_at": row.started_at, "ended_at": row.completed_at, "active_ms": row.duration_ms,
            "tokens_used": row.output_tokens, "input_tokens": row.input_tokens,
            "cache_read_tokens": row.cache_read_input_tokens, "cache_creation_tokens": row.cache_creation_input_tokens}


def read_todos(db: DbSession, session_id: str) -> list[dict]:
    return [{"content": r.content, "status": r.status} for r in db.scalars(
        select(Todo).where(Todo.session_id == session_id).order_by(Todo.position))]


def replace_todos(db: DbSession, session_id: str, items: list[dict], *, commit=True) -> None:
    try:
        db.execute(delete(Todo).where(Todo.session_id == session_id))
        now = now_ms()
        db.add_all([Todo(session_id=session_id, position=i, content=t["content"], status=t["status"],
                         time_created=now, time_updated=now) for i, t in enumerate(items)])
        touch_session(db, session_id)
        if commit:
            db.commit()
    except Exception:
        db.rollback()
        raise


def session_info(db: DbSession, row: Session) -> dict:
    messages = db.scalars(select(Message).where(Message.session_id == row.id).order_by(Message.sequence)).all()
    model = next((_load(m.data).get("modelId", "") for m in reversed(messages)
                  if message_role(m) == "assistant" and _load(m.data).get("modelId")), "")
    turns = list_turns(db, row.id)
    return {"id": row.id, "title": row.title, "model": model,
            "created_at": row.time_created, "updated_at": row.time_updated,
            "tokens_used": sum(t.output_tokens for t in turns), "input_tokens": sum(t.input_tokens for t in turns)}


def save_compaction(db, session_id, label, before_sequence, summary, tokens_before, tokens_after):
    """摘要隐藏为合成用户消息，边界保存消息 ID。"""
    boundary = db.scalar(select(Message).where(Message.session_id == session_id,
        Message.sequence <= before_sequence).order_by(Message.sequence.desc()).limit(1))
    mid = new_id()
    try:
        upsert_message(db, session_id, mid, "user", {
            "synthetic": True, "summary": {"title": "Compact summary", "body": summary},
            "semantics": {"origin": "agent_runtime", "kind": "compact_summary", "uiVisibility": "hidden",
                          "providerVisibility": "visible", "transcriptVisibility": "hidden"}}, label, commit=False)
        upsert_part(db, session_id, mid, new_id(), "text", {"text": summary, "synthetic": True}, label, commit=False)
        upsert_part(db, session_id, mid, new_id(), "compaction", {
            "auto": True, "tail_start_id": boundary.id if boundary else None,
            "preCompactTokenCount": tokens_before, "postCompactTokenCount": tokens_after}, label, commit=False)
        db.commit()
        return mid
    except Exception:
        db.rollback()
        raise
