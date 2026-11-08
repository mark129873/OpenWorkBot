import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.sandbox.lease import SandboxLeaseManager
from app.user_context import require_user


SANDBOX_TOOLS = frozenset({"bash", "read_file", "write_file", "str_replace", "glob", "ls", "grep"})


class SandboxMiddleware(AgentMiddleware):
    def __init__(self, lazy_init: bool = True, *, available_skills: set[str] | None = None,
                 owns_agent_skill_projection: bool = True, leases: SandboxLeaseManager | None = None) -> None:
        super().__init__()
        if available_skills is not None:
            raise ValueError("本项目的 Skills 由所有 Agent 共享")
        self._lazy_init = lazy_init
        self._owns_agent_skill_projection = owns_agent_skill_projection
        self.leases = leases

    def _leases(self, context) -> SandboxLeaseManager:
        leases = self.leases or context.get("sandbox_leases")
        if not isinstance(leases, SandboxLeaseManager):
            raise RuntimeError("沙箱租约管理器不可用")
        return leases

    async def _initialize(self, context) -> None:
        if not isinstance(context, dict) or context.get("user_id") != require_user():
            raise PermissionError("沙箱用户身份无效")
        user_id, thread_id, owner_id = context["user_id"], context.get("thread_id"), context.get("owner_id")
        if not isinstance(thread_id, str) or not isinstance(owner_id, str):
            raise ValueError("沙箱运行身份不完整")
        leases = self._leases(context)
        sandbox_id = await leases.acquire(owner_id, thread_id, user_id=user_id)
        sandbox = leases.provider.get_scoped(sandbox_id, thread_id=thread_id, user_id=user_id)
        if sandbox is None:
            raise RuntimeError("沙箱已不可复用")
        context["sandbox_id"] = sandbox_id
        context["sandbox"] = sandbox

    def before_agent(self, state, runtime):
        if not self._lazy_init:
            asyncio.run(self._initialize(runtime.context))

    async def abefore_agent(self, state, runtime):
        if not self._lazy_init:
            await self._initialize(runtime.context)

    def before_model(self, state, runtime):
        self._check_current(runtime.context)

    async def abefore_model(self, state, runtime):
        self._check_current(runtime.context)

    def _check_current(self, context) -> None:
        sandbox_id = context.get("sandbox_id")
        if sandbox_id and self._leases(context).provider.get_scoped(
                sandbox_id, thread_id=context.get("thread_id"), user_id=context.get("user_id")) is None:
            raise RuntimeError("沙箱已不可复用")

    def wrap_tool_call(self, request, handler):
        if request.tool_call["name"] in SANDBOX_TOOLS:
            asyncio.run(self._initialize(request.runtime.context))
        return handler(request)

    async def awrap_tool_call(self, request, handler: Callable[..., Awaitable[ToolMessage | Command]]):
        if request.tool_call["name"] in SANDBOX_TOOLS:
            await self._initialize(request.runtime.context)
        return await handler(request)

    def after_agent(self, state, runtime):
        if not runtime.context.get("retain_sandbox"):
            asyncio.run(self.release(runtime.context))

    async def aafter_agent(self, state, runtime):
        if not runtime.context.get("retain_sandbox"):
            await self.release(runtime.context)

    async def release(self, context) -> None:
        if not isinstance(context, dict) or context.get("user_id") != require_user():
            raise PermissionError("沙箱用户身份无效")
        owner_id = context.get("owner_id")
        sandbox = context.pop("sandbox", None)
        context.pop("sandbox_id", None)
        try:
            if sandbox is not None and isinstance(owner_id, str):
                operation = asyncio.create_task(asyncio.to_thread(sandbox.release_command_scope, owner_id))
                try:
                    await asyncio.shield(operation)
                except asyncio.CancelledError:
                    await operation
                    raise
        finally:
            if isinstance(owner_id, str):
                await self._leases(context).release(owner_id, user_id=context["user_id"])
