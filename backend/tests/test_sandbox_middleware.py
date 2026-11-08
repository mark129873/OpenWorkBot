import asyncio

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.sandbox.lease import SandboxLeaseManager
from app.sandbox.middleware import SandboxMiddleware
from app.user_context import require_user
from test_sandbox_lease import Provider


class Model(BaseChatModel):
    @property
    def _llm_type(self):
        return "sandbox-test"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if any(message.type == "tool" for message in messages):
            message = AIMessage(content="完成")
        else:
            message = AIMessage(content="", tool_calls=[{
                "id": "read-1", "name": "read_file", "args": {"path": "a.txt"}, "type": "tool_call",
            }])
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
async def read_file(path: str) -> str:
    """Read a test file."""
    return path


@tool("read_file")
def read_file_sync(path: str) -> str:
    """Read a test file."""
    return path


def test_langgraph_tool_lifecycle_acquires_and_releases_sandbox():
    async def run():
        provider = Provider()
        middleware = SandboxMiddleware(leases=SandboxLeaseManager(provider))
        context = {"user_id": require_user(), "thread_id": "thread", "owner_id": "run"}
        graph = create_agent(Model(), [read_file], middleware=[middleware])
        try:
            result = await graph.ainvoke({"messages": [("user", "读取文件")]}, context=context)
            assert result["messages"][-1].content == "完成"
            assert provider.acquired == [("thread", require_user())]
            assert provider.released == [f"sandbox-{require_user()}"]
            assert "sandbox" not in context and "sandbox_id" not in context
        finally:
            await middleware.release(context)
    asyncio.run(run())


@pytest.mark.parametrize("lazy_init", [True, False])
def test_synchronous_middleware_acquires_and_releases(lazy_init):
    provider = Provider()
    middleware = SandboxMiddleware(lazy_init=lazy_init, leases=SandboxLeaseManager(provider))
    context = {"user_id": require_user(), "thread_id": "thread", "owner_id": "run"}
    graph = create_agent(Model(), [read_file_sync], middleware=[middleware])
    result = graph.invoke({"messages": [("user", "读取文件")]}, context=context)
    assert result["messages"][-1].content == "完成"
    assert provider.acquired == [("thread", require_user())]
    assert provider.released == [f"sandbox-{require_user()}"]


def test_middleware_rejects_per_agent_skill_isolation():
    with pytest.raises(ValueError, match="共享"):
        SandboxMiddleware(available_skills={"one"})
