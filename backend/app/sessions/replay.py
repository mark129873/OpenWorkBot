from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.config import settings
from app.models import Message, Part, Session
from app.sessions.approvals import approval_events, get_request
from app.sessions.store import _load, list_turns, turn_fact, message_role, turn_id as message_turn_id, part_kind, part_data, session_info


def _message_text(parts: list[Part]) -> str:
    return "".join(_load(p.data).get("text", "") for p in parts if part_kind(p) == "text")


def _part_data(row: Part) -> dict:
    return part_data(row)


def _load_messages_and_parts(db: DbSession, session_id: str) -> tuple[list[Message], dict[str, list[Part]]]:
    messages = list(
        db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.sequence.asc())
        ).scalars()
    )
    parts_by_message: dict[str, list[Part]] = {}
    if messages:
        stmt = (
            select(Part)
            .where(Part.session_id == session_id, Part.message_id.in_([m.id for m in messages]))
            .order_by(Part.sequence.asc())
        )
        for part in db.execute(stmt).scalars():
            parts_by_message.setdefault(part.message_id, []).append(part)
    return messages, parts_by_message


def load_replay(
    db: DbSession, session_id: str, running_turn_ids: set[str] | None = None
) -> dict | None:
    """无收口记录且不在运行集合的轮次视为已停止。"""
    running_turn_ids = running_turn_ids or set()

    session_row = db.get(Session, session_id)
    if session_row is None or session_row.time_archived is not None:
        return None

    turn_facts = {r.turn_id: turn_fact(r) for r in list_turns(db, session_id)}
    approvals = approval_events(session_id)
    compactions = []
    messages, parts_by_message = _load_messages_and_parts(db, session_id)
    context_used = next((_load(m.data)['tokens']['input'] for m in reversed(messages)
                         if message_role(m) == 'assistant' and 'tokens' in _load(m.data)), 0)
    for msg in messages:
        for part in parts_by_message.get(msg.id, []):
            data = _load(part.data)
            if data.get('type') == 'compaction':
                compactions.append({'turn_id': message_turn_id(msg), 'time': part.time_created,
                                    'tokens_before': data.get('preCompactTokenCount'),
                                    'tokens_after': data.get('postCompactTokenCount')})

    turns: dict[str, dict] = {}
    turn_order: list[str] = []
    for msg in messages:
        if _load(msg.data).get("synthetic"):
            continue
        turn_id = message_turn_id(msg)
        if turn_id not in turns:
            fact = turn_facts.get(turn_id, {})
            turns[turn_id] = {
                "turn_id": turn_id,
                "state": ("running" if turn_id in running_turn_ids else
                          (fact.get("state", "stopped") if fact.get("state") != "running" else "stopped")),
                "started_at": fact.get("started_at", msg.time_created),
                "ended_at": fact.get("ended_at"),
                "active_ms": fact.get("active_ms"),
                "user_message": None,
                "work_items": [],
                "final_text": None,
                "_text_parts": [],
                "_first_seq": msg.sequence,
            }
            turn_order.append(turn_id)

        bucket = turns[turn_id]
        if message_role(msg) == "user" and bucket["user_message"] is None:
            bucket["user_message"] = {
                "id": msg.id,
                "text": _message_text(parts_by_message.get(msg.id, [])),
                "time": msg.time_created,
            }
            continue

        for part in parts_by_message.get(msg.id, []):
            data = _part_data(part)
            item_time = part.time_created
            if part_kind(part) == "text":
                bucket["_text_parts"].append(data.get("text", ""))
                bucket["_last_text_id"] = part.id
                bucket["work_items"].append(
                    {"kind": "text", "id": part.id, "text": data.get("text", ""), "time": item_time}
                )
            elif part_kind(part) == "tool_call":
                bucket["work_items"].append(
                    {
                        "kind": "tool_call",
                        "id": part.id,
                        "name": data.get("name", ""),
                        "input": data.get("input"),
                        "status": "failed" if data.get("status") == "error" else data.get("status", "completed"),
                        "output": data.get("output", ""),
                        "time": item_time,
                    }
                )
                if data.get('name') == 'subtask':
                    bucket['work_items'].append({'kind': 'subtask', 'id': part.id,
                        'goal': (data.get('input') or {}).get('goal', ''),
                        'status': 'failed' if data.get('status') == 'error' else data.get('status', 'running'),
                        'output': data.get('output', ''), 'time': item_time})
                elif data.get('name') == 'todo_write' and data.get('status') == 'completed':
                    from app.agent.todo import parse_items
                    items, error = parse_items((data.get('input') or {}).get('items'))
                    if not error:
                        bucket['work_items'].append({'kind': 'todo', 'id': part.id, 'items': items, 'time': item_time})
        error = _load(msg.data).get('error')
        if error:
            bucket['work_items'].append({'kind': 'error', 'id': msg.id,
                'message': error.get('data', {}).get('message', error.get('name', '')), 'time': msg.time_updated})

    # 同一请求只取最新审批事实，避免残留等待卡。
    latest_by_request: dict[str, dict] = {}
    for data in approvals:
        latest_by_request[data.get("request_id", "")] = data
    for data in latest_by_request.values():
        bucket = turns.get(data.get("turn_id", ""))
        if bucket is not None:
            bucket["work_items"].append(
                {
                    "kind": "approval",
                    "id": data.get("request_id", ""),
                    "tool": data.get("tool", ""),
                    "input": data.get("input"),
                    "reason": data.get("reason", ""),
                    "status": data.get("status", "requested"),
                    "time": data.get("time", 0),
                }
            )
    for data in compactions:
        bucket = turns.get(data.get("turn_id", ""))
        if bucket is not None:
            bucket["work_items"].append(
                {
                    "kind": "compaction",
                    "id": data.get("time", 0),
                    "tokens_before": data.get("tokens_before"),
                    "tokens_after": data.get("tokens_after"),
                    "time": data.get("time", 0),
                }
            )

    # 每轮最后一条正文移到工作块外作为最终回复。
    result_turns = []
    for turn_id in turn_order:
        bucket = turns[turn_id]
        text_parts = bucket.pop("_text_parts")
        last_text_id = bucket.pop("_last_text_id", None)
        final_text = text_parts[-1] if text_parts else None
        items = bucket["work_items"]
        if final_text is not None and last_text_id:
            items = [x for x in items if not (x["kind"] == "text" and x["id"] == last_text_id)]
        bucket["final_text"] = final_text
        bucket["work_items"] = sorted(items, key=lambda x: x.get("time", 0))
        result_turns.append(bucket)

    pending = next(
        (a for a in latest_by_request.values() if a.get("status") == "requested" and get_request(a["request_id"])), None
    )

    return {
        "session": {
            **session_info(db, session_row),
            "context_used": context_used,
            "context_window": settings.context_window,
        },
        "turns": result_turns,
        "pending_approval": pending,
    }


def load_history(db: DbSession, session_id: str) -> list[dict]:
    """摘要边界之后重建历史；tool_use 后紧接 tool_result。"""
    messages, parts_by_message = _load_messages_and_parts(db, session_id)

    # 摘要为隐藏用户消息；tail_start_id 指向最后被摘要的消息。
    cutoff_seq = -1
    summary_text = ''
    for msg in messages:
        if not _load(msg.data).get('synthetic'):
            continue
        for part in parts_by_message.get(msg.id, []):
            data = _load(part.data)
            if data.get('type') == 'compaction':
                boundary = next((m for m in messages if m.id == data.get('tail_start_id')), None)
                cutoff_seq = boundary.sequence if boundary else -1
                summary_text = _message_text(parts_by_message.get(msg.id, []))
    result: list[dict] = []
    if summary_text:
        result.append({'role': 'user', 'content': f'<此前对话摘要>\n{summary_text}\n</此前对话摘要>'})

    pending_tool_results: list[dict] = []

    def flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            result.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for msg in messages:
        if _load(msg.data).get("synthetic") or msg.sequence <= cutoff_seq:
            continue
        if message_role(msg) == "user":
            flush_tool_results()
            text = _message_text(parts_by_message.get(msg.id, []))
            result.append({"role": "user", "content": text})
            continue

        flush_tool_results()
        content: list[dict] = []
        for part in parts_by_message.get(msg.id, []):
            data = _part_data(part)
            if part_kind(part) == "text" and data.get("text"):
                content.append({"type": "text", "text": data["text"]})
            elif part_kind(part) == "tool_call":
                content.append(
                    {
                        "type": "tool_use",
                        "id": data.get("tool_call_id", part.id),
                        "name": data.get("name", ""),
                        "input": data.get("input") or {},
                    }
                )
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": data.get("tool_call_id", part.id),
                        "content": data.get("output", ""),
                        **({"is_error": True} if data.get("status") == "error" else {}),
                    }
                )
        if content:
            result.append({"role": "assistant", "content": content})

    flush_tool_results()
    return result
