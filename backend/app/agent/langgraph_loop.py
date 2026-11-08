from contextvars import ContextVar

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.agent.graph_middleware import OrderedToolMiddleware
from app.sandbox.middleware import SandboxMiddleware


_model_context = ContextVar("openworkbot_model_context")


class _Model(BaseChatModel):
    @property
    def _llm_type(self):
        return "openworkbot"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("模型调用缺少运行上下文")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        callback, state, context = _model_context.get()
        message = await callback(state, context)
        return ChatResult(generations=[ChatGeneration(message=message)])


class _LoopMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["tools", "end"])
    async def abefore_model(self, state, runtime):
        before = runtime.context.get("before_model")
        return await before(state, runtime.context) if before else None

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        after = runtime.context.get("after_model")
        return await after(state, runtime.context) if after else None

    async def awrap_model_call(self, request, handler):
        context = request.runtime.context
        token = _model_context.set((context["model_call"], request.state, context))
        try:
            return await handler(request)
        finally:
            _model_context.reset(token)

    async def awrap_tool_call(self, request, handler):
        output = await request.runtime.context["tool_call"](request.tool_call, request.runtime.context)
        return ToolMessage(content=output, tool_call_id=request.tool_call["id"], name=request.tool_call["name"],
                           status="error" if output.startswith("Error:") else "success")


async def _placeholder(**kwargs):
    raise RuntimeError("工具调用缺少运行上下文")


class LangGraphLoop:
    def __init__(self, specs, *, checkpointer=None, leases=None):
        unique = {spec["name"]: spec for spec in specs}
        tools = [StructuredTool(name=spec["name"], description=spec["description"],
                                args_schema=spec["input_schema"], coroutine=_placeholder)
                 for spec in unique.values()]
        middleware = [OrderedToolMiddleware()]
        self.sandbox_middleware = SandboxMiddleware(leases=leases) if leases else None
        if self.sandbox_middleware:
            middleware.append(self.sandbox_middleware)
        middleware.append(_LoopMiddleware())
        self.graph = create_agent(_Model(), tools, middleware=middleware, checkpointer=checkpointer)

    async def invoke(self, context, *, thread_id):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}
        succeeded = False
        try:
            durability = {"durability": "sync"} if self.graph.checkpointer else {}
            result = await self.graph.ainvoke({"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(content="开始")]},
                                              config, context=context, **durability)
            succeeded = True
            return result
        finally:
            retain = context.get("retain_sandbox") and (succeeded or context.pop("retain_on_error", False))
            if self.sandbox_middleware and not retain:
                await self.sandbox_middleware.release(context)
