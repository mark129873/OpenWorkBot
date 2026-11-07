import asyncio
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from app.sandbox.lease import SandboxLeaseManager
from app.user_context import require_user


SANDBOX_TOOLS = frozenset({"bash", "read_file", "write_file", "edit_file", "glob", "delete_file"})


class SandboxMiddleware(AgentMiddleware):
    def __init__(self, leases: SandboxLeaseManager) -> None:
        super().__init__()
        self.leases = leases

    async def awrap_tool_call(self, request, handler: Callable[..., Awaitable[ToolMessage | Command]]):
        if request.tool_call["name"] not in SANDBOX_TOOLS:
            return await handler(request)
        context = request.runtime.context
        if not isinstance(context, dict) or context.get("user_id") != require_user():
            raise PermissionError("沙箱用户身份无效")
        user_id, thread_id, owner_id = context["user_id"], context.get("thread_id"), context.get("owner_id")
        if not isinstance(thread_id, str) or not isinstance(owner_id, str):
            raise ValueError("沙箱运行身份不完整")
        sandbox_id = await self.leases.acquire(owner_id, thread_id, user_id=user_id)
        sandbox = self.leases.provider.get_scoped(sandbox_id, thread_id=thread_id, user_id=user_id)
        if sandbox is None:
            raise RuntimeError("沙箱已不可复用")
        context["sandbox_id"] = sandbox_id
        context["sandbox"] = sandbox
        return await handler(request)

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
                await self.leases.release(owner_id, user_id=context["user_id"])
