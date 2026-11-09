import logging
from contextvars import ContextVar

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from app.agent.context import ConversationState, confirmed, messages_from_records
from app.agent.graph_middleware import OrderedToolMiddleware
from app.agent.todo import TodoMiddleware
from app.sandbox.middleware import SandboxMiddleware


_model_context = ContextVar("openworkbot_model_context")
logger = logging.getLogger(__name__)


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
        acknowledge = context.get("acknowledge_pending")
        if acknowledge:
            records = context.pop("pending_context_records", [])
            await acknowledge(confirmed(records, state.get("messages", [])))
        message = await callback(state, context)
        return ChatResult(generations=[ChatGeneration(message=message)])


class _LoopMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["tools", "end"])
    async def abefore_model(self, state, runtime):
        before = runtime.context.get("before_model")
        result = (await before(state, runtime.context) if before else None) or {}
        collect = runtime.context.get("collect_pending")
        if collect:
            records = collect()
            runtime.context["pending_context_records"] = records
            incoming = messages_from_records(records)
            if incoming:
                result["messages"] = incoming + result.get("messages", [])
        repair = runtime.context.get("repair_messages")
        if repair and result.get("jump_to") not in {"tools", "end"}:
            combined = add_messages(state.get("messages", []), result.get("messages", []))
            messages, changed = repair(combined)
            if changed:
                result["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages]
        return result or None

    @hook_config(can_jump_to=["model", "end"])
    async def aafter_model(self, state, runtime):
        after = runtime.context.get("after_model")
        result = (await after(state, runtime.context) if after else None) or {}
        last = state.get("messages", [])[-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and last.additional_kwargs.get("memory_status") == "pending":
            status = runtime.context.pop("memory_status", "completed")
            result["messages"] = [last.model_copy(update={"additional_kwargs": {**last.additional_kwargs,
                                                                                 "memory_status": status}})] + result.get("messages", [])
        compacted = runtime.context.pop("compacted_context", None)
        if compacted:
            result["messages"] = [RemoveMessage(id=identity) for identity in compacted["ids"]] + result.get("messages", [])
            result["summary_text"] = compacted["summary"]
        return result or None

    async def awrap_model_call(self, request, handler):
        context = request.runtime.context
        token = _model_context.set((context["model_call"], request.state, context))
        try:
            return await handler(request)
        finally:
            _model_context.reset(token)

    async def awrap_tool_call(self, request, handler):
        output = await request.runtime.context["tool_call"](request.tool_call, request.runtime.context)
        group_id = request.runtime.context.get("group_id")
        identity = f"result:{group_id}:{request.tool_call['id']}" if group_id else None
        if request.tool_call["name"] == "write_todos" and not output.startswith("Error:"):
            result = await handler(request)
            for message in result.update.get("messages", []):
                if isinstance(message, ToolMessage):
                    message.id = identity
            apply_todos = request.runtime.context.get("apply_todos")
            if apply_todos:
                error = await apply_todos(result.update["todos"], request.tool_call)
                if error:
                    return ToolMessage(content=error, tool_call_id=request.tool_call["id"], id=identity,
                                       name=request.tool_call["name"], status="error")
                return result.update["messages"][0]
            return result
        return ToolMessage(content=output, tool_call_id=request.tool_call["id"], id=identity, name=request.tool_call["name"],
                           status="error" if output.startswith("Error:") else "success")


async def _placeholder(**kwargs):
    raise RuntimeError("工具调用缺少运行上下文")


class LangGraphLoop:
    def __init__(self, specs, *, checkpointer=None, leases=None):
        unique = {spec["name"]: spec for spec in specs}
        tools = [StructuredTool(name=spec["name"], description=spec["description"],
                                args_schema=spec["input_schema"], coroutine=_placeholder)
                 for spec in unique.values() if spec["name"] != "write_todos"]
        middleware = [OrderedToolMiddleware()]
        self.sandbox_middleware = SandboxMiddleware(leases=leases) if leases else None
        if self.sandbox_middleware:
            middleware.append(self.sandbox_middleware)
        if "write_todos" in unique:
            middleware.append(TodoMiddleware())
        middleware.append(_LoopMiddleware())
        self.graph = create_agent(_Model(), tools, middleware=middleware, checkpointer=checkpointer,
                                  state_schema=ConversationState)

    async def invoke(self, context, *, thread_id):
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}
        succeeded = False
        try:
            durability = {"durability": "sync"} if self.graph.checkpointer else {}
            initial = [] if context.get("collect_pending") else [HumanMessage(content="开始")]
            result = await self.graph.ainvoke({"messages": initial},
                                              config, context=context, **durability)
            if context.get("collect_pending") and context.get("acknowledge_pending"):
                try:
                    records = context["collect_pending"]()
                    await context["acknowledge_pending"](confirmed(records, result.get("messages", [])))
                except Exception:
                    logger.exception("检查点消息暂存确认失败，将在下次处理")
            succeeded = True
            return result
        finally:
            retain = context.get("retain_sandbox") and (succeeded or context.pop("retain_on_error", False))
            if self.sandbox_middleware and not retain:
                await self.sandbox_middleware.release(context)
