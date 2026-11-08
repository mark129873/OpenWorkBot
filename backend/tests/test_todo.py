import asyncio

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.langgraph_loop import LangGraphLoop
from app.agent.todo import remind
from app.agent.tools import TOOLS


def test_todos_are_graph_state_and_isolated_by_thread():
    async def run():
        spec = next(tool for tool in TOOLS if tool["name"] == "write_todos")
        loop = LangGraphLoop([spec], checkpointer=InMemorySaver())
        seen = []

        async def model(state, context):
            seen.append(state.get("todos"))
            if context.get("sent"):
                return AIMessage(content="完成")
            context["sent"] = True
            return AIMessage(content="", tool_calls=[{"id": "one", "name": "write_todos",
                "args": {"todos": context["todos"]}, "type": "tool_call"}])

        async def tool(call, context):
            return "已更新 TODO。"

        items = [{"content": "", "status": "in_progress"}, {"content": "第二项", "status": "in_progress"}]
        first = await loop.invoke({"model_call": model, "tool_call": tool, "todos": items}, thread_id="a")
        assert first["todos"] == items
        second = await loop.invoke({"model_call": model, "tool_call": tool, "todos": []}, thread_id="a")
        assert second["todos"] == [] and seen[-2] == items
        third = await loop.invoke({"model_call": model, "tool_call": tool, "todos": items}, thread_id="b")
        assert third["todos"] == items and seen[-2] in (None, [])
    asyncio.run(run())


def test_todo_reminder_uses_current_state_without_changing_history():
    state = {"todos": [{"content": "当前事项", "status": "pending"}]}
    history = [{"role": "user", "content": "继续"}]
    result = remind(history, state)
    assert len(result) == 2 and "当前事项" in result[-1]["content"]
    assert history == [{"role": "user", "content": "继续"}]
    history.append({"role": "assistant", "content": [{"type": "tool_use", "name": "write_todos"}]})
    assert remind(history, state) is history


def test_duplicate_todo_calls_fail_without_running_later_tools():
    async def run():
        spec = next(tool for tool in TOOLS if tool["name"] == "write_todos")
        other = {"name": "other", "description": "test", "input_schema": {
            "type": "object", "properties": {}, "required": []}}
        loop = LangGraphLoop([spec, other])
        calls = []

        async def model(state, context):
            if context.get("sent"):
                return AIMessage(content="完成")
            context["sent"] = True
            return AIMessage(content="", tool_calls=[
                {"id": "a", "name": "write_todos", "args": {"todos": []}, "type": "tool_call"},
                {"id": "b", "name": "other", "args": {}, "type": "tool_call"},
                {"id": "c", "name": "write_todos", "args": {"todos": []}, "type": "tool_call"}])

        async def tool(call, context):
            calls.append(call["id"])
            return "ok"

        result = await asyncio.wait_for(loop.invoke({"model_call": model, "tool_call": tool}, thread_id="t"), 3)
        assert calls == [] and not result.get("todos")
        assert [message.status for message in result["messages"] if message.type == "tool"] == ["error"] * 3
    asyncio.run(run())
