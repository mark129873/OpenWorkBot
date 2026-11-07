import hashlib

from sqlalchemy import func, select

from app.agent import agent_bus
from app.models import Part, new_id
from app.sessions import chat_store as store
from app.user_context import scoped


NAMES = {"state_write", "state_read", "state_list", "agent_subscribe", "agent_unsubscribe", "agent_publish"}
_TERMINAL = {"completed", "failed", "stopped"}


def _topic_key(topic: str, actor: str) -> str:
    return f"subscription:{hashlib.sha256(topic.encode('utf-8')).hexdigest()}:{actor}"


def _text(args: dict, key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 不能为空")
    return value


def _active_contexts(session) -> set[str]:
    tasks = [item["context_id"] for item in store.records(session, "task_state")
             if item["status"] not in _TERMINAL and not item.get("cleanup")]
    helpers = [item["context_id"] for item in store.records(session, "agent_run")
               if item["status"] not in _TERMINAL]
    return {"main", *tasks, *helpers}


def _remember_reader(session, state: dict, actor: str) -> None:
    if actor != "main" and actor not in state.get("readers", []):
        state.setdefault("readers", []).append(actor)
        store.put(session, f"state:{state['id']}", state)


def _public(state: dict) -> dict:
    return {key: state.get(key) for key in ("id", "task_id", "kind", "content", "version", "updated_by")}


def call(session, name: str, args: dict, *, actor: str, task_id: str | None, request_id: str) -> str:
    if actor not in _active_contexts(session):
        return "Error: 当前 Agent 已结束"
    if name in {"agent_subscribe", "agent_unsubscribe"}:
        topic = _text(args, "topic")
        key = _topic_key(topic, actor)
        if name == "agent_subscribe":
            store.put(session, key, {"type": "subscription", "topic": topic, "context_id": actor, "task_id": task_id})
        else:
            row = session.get(Part, scoped(key))
            if row:
                session.delete(row)
        return "订阅已更新"
    if name == "agent_publish":
        topic, content = _text(args, "topic"), _text(args, "content")
        key = f"publication:{request_id}"
        previous = store.get(session, key)
        if previous:
            if previous.get("processed"):
                return "已处理"
            if previous["topic"] != topic or previous["content"] != content:
                raise ValueError("相同发布 ID 的内容不一致")
            return store.dump({"recipients": previous["recipients"]})
        active = _active_contexts(session)
        recipients = sorted({item["context_id"] for item in store.records(session, "subscription")
                             if item["topic"] == topic and item["context_id"] in active})
        for target in recipients:
            agent_bus.send(session, source=actor, target=target, task_id=task_id or "", content=content,
                           kind="broadcast", message_id=f"{request_id}:{target}")
        store.put(session, key, {"type": "publication", "task_id": task_id, "topic": topic,
                                 "content": content, "recipients": recipients})
        return store.dump({"recipients": recipients})
    if name == "state_write":
        key = f"state-action:{request_id}"
        previous = store.get(session, key)
        if previous:
            if previous.get("processed"):
                return "已处理"
            if previous["args"] != args:
                raise ValueError("相同状态请求 ID 的参数不一致")
            return previous["output"]
        state_id = args.get("state_id") or new_id()
        expected = args.get("expected_version")
        kind, content = _text(args, "kind"), _text(args, "content")
        if not isinstance(state_id, str) or not state_id or type(expected) is not int or expected < 0:
            raise ValueError("状态 ID 或版本无效")
        state = store.get(session, f"state:{state_id}")
        if state and state["version"] != expected or not state and expected != 0:
            return store.dump({"updated": False, "conflict": True, "state": _public(state) if state else None})
        owner_task = state.get("task_id") if state else args.get("task_id", task_id)
        if not state and owner_task is not None and not store.get(session, f"task:{owner_task}"):
            raise ValueError("任务不存在")
        state = {"type": "shared_state", "id": state_id, "task_id": owner_task, "kind": kind,
                 "content": content, "version": expected + 1, "updated_by": actor,
                 "readers": state.get("readers", []) if state else []}
        store.put(session, f"state:{state_id}", state)
        output = store.dump({"updated": True, "conflict": False, "state": _public(state)})
        store.put(session, key, {"type": "coord_action", "task_id": task_id, "args": args, "output": output})
        return output
    if name == "state_read":
        state = store.get(session, f"state:{_text(args, 'state_id')}")
        if not state:
            return "Error: 共享状态不存在"
        _remember_reader(session, state, actor)
        return store.dump(_public(state))
    if name == "state_list":
        offset, limit = args.get("offset", 0), args.get("limit", 10)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("分页参数无效")
        query = select(Part).where(Part.message_id == scoped("main:control"),
                                   func.json_extract(Part.data, "$.type") == "shared_state")
        for field in ("task_id", "kind"):
            if field in args:
                query = query.where(func.json_extract(Part.data, f"$.{field}") == _text(args, field))
        rows = list(session.scalars(query.order_by(Part.sequence, Part.id).offset(offset).limit(limit + 1)))
        states = [store.read(row) for row in rows[:limit]]
        for state in states:
            _remember_reader(session, state, actor)
        return store.dump({"items": [_public(state) for state in states],
                           "next_offset": offset + limit if len(rows) > limit else None})
    return f"Error: 未知协作工具 {name}"
