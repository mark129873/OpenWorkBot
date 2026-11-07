import asyncio

from app.sandbox.sandbox_provider import SandboxProvider


class SandboxLeaseManager:
    def __init__(self, provider: SandboxProvider) -> None:
        self.provider = provider
        self._locks: dict[str, asyncio.Lock] = {}
        self._owners: dict[tuple[str, str], tuple[str, str]] = {}
        self._leases: dict[str, set[tuple[str, str]]] = {}
        self._sandbox_users: dict[str, str] = {}

    async def acquire(self, owner_id: str, thread_id: str, *, user_id: str) -> str:
        if not owner_id or not thread_id or not user_id:
            raise ValueError("沙箱租约身份不完整")
        key = (user_id, owner_id)
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            previous = self._owners.get(key)
            if previous:
                if previous[0] != thread_id:
                    raise PermissionError("沙箱租约线程不匹配")
                if self.provider.get_scoped(previous[1], thread_id=thread_id, user_id=user_id) is None:
                    raise RuntimeError("沙箱已不可复用")
                return previous[1]
            operation = asyncio.create_task(self.provider.acquire_async(thread_id, user_id=user_id))
            try:
                sandbox_id = await asyncio.shield(operation)
            except asyncio.CancelledError:
                sandbox_id = await operation
                if not self._leases.get(sandbox_id):
                    await asyncio.to_thread(self.provider.release, sandbox_id)
                raise
            sandbox_user = self._sandbox_users.get(sandbox_id)
            if sandbox_user is not None and sandbox_user != user_id:
                raise PermissionError("沙箱不属于当前用户")
            if self.provider.get_scoped(sandbox_id, thread_id=thread_id, user_id=user_id) is None:
                if not self._leases.get(sandbox_id):
                    await asyncio.to_thread(self.provider.release, sandbox_id)
                raise PermissionError("沙箱身份校验失败")
            self._owners[key] = (thread_id, sandbox_id)
            self._leases.setdefault(sandbox_id, set()).add(key)
            self._sandbox_users[sandbox_id] = user_id
            return sandbox_id

    async def release(self, owner_id: str, *, user_id: str) -> None:
        key = (user_id, owner_id)
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            previous = self._owners.pop(key, None)
            if previous is None:
                return
            _, sandbox_id = previous
            owners = self._leases[sandbox_id]
            owners.remove(key)
            if owners:
                return
            del self._leases[sandbox_id]
            operation = asyncio.create_task(asyncio.to_thread(self.provider.release, sandbox_id))
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                await operation
                self._sandbox_users.pop(sandbox_id, None)
                raise
            self._sandbox_users.pop(sandbox_id, None)
