import asyncio
from typing import Any, Callable, Awaitable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime as GraphRuntime

from app import db
from app.sessions import chat_store as store
from app.user_context import require_user


class AgentState(TypedDict, total=False):
    history: list[dict]
    memory: str
    incoming: dict | None
    deliveries: dict[str, dict]


class AgentContext(TypedDict):
    action: Callable[[], Awaitable[Any]] | None


def _receive(state: AgentState) -> dict:
    message = state.get("incoming")
    if not message:
        return {}
    deliveries = dict(state.get("deliveries") or {})
    previous = deliveries.get(message["id"])
    if previous is not None and previous != message:
        raise ValueError("相同投递 ID 的消息不一致")
    deliveries[message["id"]] = message
    return {"deliveries": deliveries, "incoming": None}


async def _act(state: AgentState, runtime: GraphRuntime[AgentContext]) -> dict:
    action = runtime.context.get("action")
    if action is not None:
        await action()
    return {}


class AgentGraph:
    def __init__(self, checkpointer) -> None:
        self.write_lock = getattr(checkpointer, "write_lock", asyncio.Lock())
        builder = StateGraph(AgentState, context_schema=AgentContext)
        builder.add_node("receive", _receive)
        builder.add_node("act", _act)
        builder.add_edge(START, "receive")
        builder.add_edge("receive", "act")
        builder.add_edge("act", END)
        self.graph = builder.compile(checkpointer=checkpointer)
        self.locks: dict[str, asyncio.Lock] = {}

    def thread_id(self, context_id: str) -> str:
        with db.new_session() as session:
            meta = store.get(session, "meta")
            if not meta:
                raise RuntimeError("助手尚未初始化")
            thread_id = f"{require_user()}:{meta['assistant_id']}:{context_id}"
            key = f"graph:{context_id}"
            record = store.get(session, key)
            if not record:
                store.put(session, key, {"type": "graph_thread", "context_id": context_id,
                                         "thread_id": thread_id})
                session.commit()
            elif record["thread_id"] != thread_id:
                raise ValueError("Agent 线程归属不匹配")
        return thread_id

    async def invoke(self, context_id: str, *, incoming: dict | None = None,
                     history: list[dict] | None = None, memory: str | None = None,
                     action: Callable[[], Awaitable[Any]] | None = None) -> None:
        async with self.write_lock:
            thread_id = self.thread_id(context_id)
        config = {"configurable": {"thread_id": thread_id}}
        data: AgentState = {"incoming": incoming}
        if history is not None:
            data["history"] = history
        if memory is not None:
            data["memory"] = memory
        async with self.locks.setdefault(thread_id, asyncio.Lock()):
            await self.graph.ainvoke(data, config, context={"action": action}, durability="sync")

    async def has_delivery(self, context_id: str, message: dict) -> bool:
        async with self.write_lock:
            thread_id = self.thread_id(context_id)
        async with self.locks.setdefault(thread_id, asyncio.Lock()):
            state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        return (state.values.get("deliveries") or {}).get(message["id"]) == message

    async def delete_thread(self, thread_id: str) -> None:
        async with self.locks.setdefault(thread_id, asyncio.Lock()):
            await self.graph.checkpointer.adelete_thread(thread_id)
