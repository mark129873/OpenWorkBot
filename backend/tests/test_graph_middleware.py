import asyncio

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.agent.graph_middleware import OrderedToolMiddleware


class Model(BaseChatModel):
    @property
    def _llm_type(self):
        return "ordering-test"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(message.type == "tool" for message in messages):
            result = AIMessage(content="完成")
        else:
            result = AIMessage(content="", tool_calls=[
                {"id": "first", "name": "operation", "args": {"name": "first"}, "type": "tool_call"},
                {"id": "second", "name": "operation", "args": {"name": "second"}, "type": "tool_call"},
            ])
        return ChatResult(generations=[ChatGeneration(message=result)])


def test_multiple_langgraph_tool_calls_keep_model_order():
    async def run():
        events = []

        @tool
        async def operation(name: str) -> str:
            """Record a tool call."""
            events.append((name, "start"))
            await asyncio.sleep(0.02)
            events.append((name, "end"))
            return name

        graph = create_agent(Model(), [operation], middleware=[OrderedToolMiddleware()])
        result = await graph.ainvoke({"messages": [("user", "run")]}, context={"run": "test"})
        assert result["messages"][-1].content == "完成"
        assert events == [("first", "start"), ("first", "end"), ("second", "start"), ("second", "end")]
    asyncio.run(run())
