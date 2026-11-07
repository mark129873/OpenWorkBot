import asyncio

from langchain_core.messages import AIMessage

from app.agent.langgraph_loop import LangGraphLoop
from app.sandbox.lease import SandboxLeaseManager
from app.user_context import require_user
from test_sandbox_lease import Provider


def test_native_graph_controls_model_and_ordered_tools_and_sandbox_lifecycle():
    async def run():
        provider = Provider()
        loop = LangGraphLoop([{"name": "read_file", "description": "Read file", "input_schema": {
            "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}],
            leases=SandboxLeaseManager(provider))
        events = []

        async def model(state, context):
            events.append("model")
            if any(message.type == "tool" for message in state["messages"]):
                return AIMessage(content="完成")
            return AIMessage(content="", tool_calls=[
                {"id": "one", "name": "read_file", "args": {"path": "a"}, "type": "tool_call"},
                {"id": "two", "name": "read_file", "args": {"path": "b"}, "type": "tool_call"}])

        async def tool(call, context):
            assert context["sandbox"] is not None
            events.append(call["args"]["path"])
            return "ok"

        context = {"user_id": require_user(), "thread_id": "thread", "owner_id": "run",
                   "model_call": model, "tool_call": tool}
        result = await loop.invoke(context, thread_id="thread")
        assert result["messages"][-1].content == "完成"
        assert events == ["model", "a", "b", "model"]
        assert len(provider.acquired) == 1 and len(provider.released) == 1
    asyncio.run(run())


def test_existing_tool_group_runs_without_an_extra_model_call():
    async def run():
        from langgraph.checkpoint.memory import InMemorySaver

        loop = LangGraphLoop([{"name": "operation", "description": "Operation", "input_schema": {
            "type": "object", "properties": {}, "required": []}}], checkpointer=InMemorySaver())
        calls = []

        async def before(state, context):
            if context.get("injected"):
                return {"jump_to": "end"}
            context["injected"] = True
            return {"messages": [AIMessage(content="", tool_calls=[
                {"id": "one", "name": "operation", "args": {}, "type": "tool_call"}])], "jump_to": "tools"}

        async def model(state, context):
            raise AssertionError("existing actions must not call the model")

        async def tool(call, context):
            calls.append(call["id"])
            return "ok"

        await loop.invoke({"before_model": before, "model_call": model, "tool_call": tool}, thread_id="thread")
        assert calls == ["one"]
    asyncio.run(run())


def test_unknown_tool_still_reaches_the_execution_journal():
    async def run():
        loop = LangGraphLoop([{"name": "known", "description": "Known", "input_schema": {
            "type": "object", "properties": {}, "required": []}}])
        calls = []

        async def model(state, context):
            if calls:
                return AIMessage(content="完成")
            return AIMessage(content="", tool_calls=[{"id": "one", "name": "unknown", "args": {}, "type": "tool_call"}])

        async def tool(call, context):
            calls.append(call["name"])
            return "Error: 未知工具"

        await loop.invoke({"model_call": model, "tool_call": tool}, thread_id="thread")
        assert calls == ["unknown"]
    asyncio.run(run())


def test_ordering_is_scoped_to_each_model_message():
    async def run():
        loop = LangGraphLoop([{"name": "operation", "description": "Operation", "input_schema": {
            "type": "object", "properties": {}, "required": []}}])
        calls = []

        async def model(state, context):
            if len(calls) == 2:
                return AIMessage(content="完成")
            return AIMessage(content="", tool_calls=[{"id": "same-id", "name": "operation", "args": {}, "type": "tool_call"}])

        async def tool(call, context):
            calls.append(call["id"])
            return "ok"

        await loop.invoke({"model_call": model, "tool_call": tool}, thread_id="thread")
        assert calls == ["same-id", "same-id"]
    asyncio.run(run())
