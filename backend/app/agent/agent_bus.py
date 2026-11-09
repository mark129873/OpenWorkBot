"""消息以数据库为准，队列只用于唤醒运行时。"""

from app.models import new_id
from app.sessions import chat_store as store


def send(db, *, source: str, target: str, task_id: str, content: str,
         kind: str = "message", message_id: str | None = None, run_id: str | None = None) -> str:
    message_id = message_id or new_id()
    key = f"bus:{message_id}"
    value = {"type": "agent_message", "id": message_id, "source": source, "context_id": target,
             "task_id": task_id, "run_id": run_id, "kind": kind, "content": content, "delivery": "pending"}
    old = store.get(db, key)
    if old:
        if any(old.get(field) != value[field] for field in value if field != "delivery"):
            raise ValueError("相同消息 ID 的内容不一致")
    else:
        store.put(db, key, value)
    return message_id


def deliver(db, message: dict) -> None:
    store.stage_context(db, message["context_id"], "user", message["content"], key=f"bus:{message['id']}")
    message["delivery"] = "delivered"
    store.put(db, f"bus:{message['id']}", message)


def processed(db, message: dict) -> None:
    message["delivery"] = "processed"
    store.put(db, f"bus:{message['id']}", message)


def find_cycles(edges: list[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
    cycles: set[tuple[str, ...]] = set()

    def visit(node: str, path: list[str]) -> None:
        if node in path:
            cycle = path[path.index(node):]
            first = min(range(len(cycle)), key=cycle.__getitem__)
            cycles.add(tuple(cycle[first:] + cycle[:first]))
            return
        for target in graph.get(node, ()):
            visit(target, path + [node])

    for start in graph:
        visit(start, [])
    return [list(cycle) + [cycle[0]] for cycle in sorted(cycles)]


def wait_graph(db) -> list[tuple[str, str]]:
    edges = []
    for task in store.records(db, "task_state"):
        if task["status"] not in {"completed", "failed", "stopped"}:
            edges.append((f"task:{task['id']}", f"context:{task['context_id']}"))
        if task.get("dependency"):
            edges.append((f"task:{task['id']}", f"task:{task['dependency']}"))
    for waiting in store.records(db, "file_wait"):
        if waiting.get("status") == "waiting":
            source = (f"context:{waiting['context_id']}" if waiting.get("context_id")
                      else f"task:{waiting['task_id']}")
            edges.append((source, f"task:{waiting['owner']}"))
    for waiting in store.records(db, "agent_wait"):
        if waiting.get("status") == "waiting":
            edges.append((f"context:{waiting['context_id']}", f"context:{waiting['target']}"))
    return edges
