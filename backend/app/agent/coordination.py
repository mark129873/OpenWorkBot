import hashlib

from app.agent import agent_bus
from app.models import Part
from app.sessions import chat_store as store
from app.user_context import scoped


NAMES = {"agent_subscribe", "agent_unsubscribe", "agent_publish"}
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
    return f"Error: 未知协作工具 {name}"
