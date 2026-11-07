import asyncio
import threading

import pytest

from app.sandbox.lease import SandboxLeaseManager
from app.sandbox.sandbox_provider import SandboxProvider


class Provider(SandboxProvider):
    def __init__(self):
        self.acquired = []
        self.released = []
        self.ready = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.same_id = False

    def acquire(self, thread_id=None, *, user_id=None):
        self.ready.set()
        assert self.proceed.wait(5)
        self.acquired.append((thread_id, user_id))
        return "same" if self.same_id else f"sandbox-{user_id}"

    def get(self, sandbox_id):
        return self

    def get_scoped(self, sandbox_id, *, thread_id, user_id):
        return self if sandbox_id == ("same" if self.same_id else f"sandbox-{user_id}") else None

    def release_command_scope(self, scope_id):
        pass

    def release(self, sandbox_id):
        self.released.append(sandbox_id)


def test_owners_share_user_sandbox_until_last_release():
    async def run():
        provider = Provider()
        leases = SandboxLeaseManager(provider)
        a, b, other = await asyncio.gather(
            leases.acquire("run-a", "thread-a", user_id="one"),
            leases.acquire("run-b", "thread-b", user_id="one"),
            leases.acquire("run-c", "thread-c", user_id="two"),
        )
        assert a == b != other
        await leases.release("run-a", user_id="one")
        assert a not in provider.released
        await leases.release("run-b", user_id="one")
        await leases.release("run-c", user_id="two")
        assert set(provider.released) == {a, other}
    asyncio.run(run())


def test_sandbox_identity_cannot_be_shared_across_users():
    async def run():
        provider = Provider()
        provider.same_id = True
        leases = SandboxLeaseManager(provider)
        await leases.acquire("a", "thread-a", user_id="one")
        with pytest.raises(PermissionError, match="不属于当前用户"):
            await leases.acquire("b", "thread-b", user_id="two")
        assert not provider.released
        await leases.release("a", user_id="one")
    asyncio.run(run())


def test_cancelled_acquire_releases_created_sandbox():
    async def run():
        provider = Provider()
        provider.proceed.clear()
        leases = SandboxLeaseManager(provider)
        operation = asyncio.create_task(leases.acquire("run", "thread", user_id="one"))
        assert await asyncio.to_thread(provider.ready.wait, 5)
        operation.cancel()
        provider.proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert provider.released == ["sandbox-one"]
    asyncio.run(run())
