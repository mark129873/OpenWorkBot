import asyncio

import pytest

from app.community.aio_sandbox.aio_sandbox_provider import AioSandboxProvider
from app.community.aio_sandbox.local_backend import LocalContainerBackend
from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.sandbox.lease import SandboxLeaseManager
from app.user_context import bind_user, require_user


class Backend:
    instance = "instance"

    def __init__(self):
        self.created = []
        self.destroyed = []
        self.alive = True
        self.reconciled = 0
        self.grants = []
        self.decisions = []

    _network_target = staticmethod(LocalContainerBackend._network_target)

    def seed_network_grants(self, info, grants):
        self.grants.append((info.user_id, grants))

    def decide_network_event(self, info, request_id, decision):
        self.decisions.append((info.user_id, request_id, decision))

    def reconcile(self):
        self.reconciled += 1

    def create(self, thread_id, sandbox_id, *, user_id):
        info = SandboxInfo(sandbox_id, user_id, "generation", "container", "proxy", "network", "egress", "url", "token")
        self.created.append(info)
        return info

    def is_alive(self, info):
        return self.alive

    def destroy(self, info):
        self.destroyed.append(info)


class Sandbox:
    def close(self):
        pass


def test_provider_shares_user_sandbox_and_checks_scope(tmp_path, monkeypatch):
    backend = Backend()
    provider = AioSandboxProvider(backend, tmp_path)
    monkeypatch.setattr(provider, "_make_sandbox", lambda info: Sandbox())
    try:
        user = require_user()
        first = provider.acquire("thread-a", user_id=user)
        assert provider.acquire("thread-b", user_id=user) == first
        assert len(backend.created) == 1 and backend.reconciled == 1
        assert provider.get_scoped(first, thread_id="thread-a", user_id=user) is not None
        assert provider.get_scoped(first, thread_id="not-acquired", user_id=user) is None
        with bind_user("b" * 32):
            assert provider.get_scoped(first, thread_id="thread-a", user_id="b" * 32) is None
            second = provider.acquire("other", user_id=require_user())
            assert second != first
            provider.release(second)
        provider.release(first)
    finally:
        provider.close()
    assert len(backend.destroyed) == 2


def test_quarantine_waits_for_all_leases_before_recycle(tmp_path, monkeypatch):
    async def run():
        backend = Backend()
        provider = AioSandboxProvider(backend, tmp_path)
        monkeypatch.setattr(provider, "_make_sandbox", lambda info: Sandbox())
        leases = SandboxLeaseManager(provider)
        user = require_user()
        sandbox = await leases.acquire("a", "thread-a", user_id=user)
        assert await leases.acquire("b", "thread-b", user_id=user) == sandbox
        provider.quarantine_sandbox(sandbox, "命令停止结果未知")
        with pytest.raises(RuntimeError, match="沙箱已隔离"):
            await leases.acquire("c", "thread-c", user_id=user)
        await leases.release("a", user_id=user)
        assert not backend.destroyed
        await leases.release("b", user_id=user)
        assert len(backend.destroyed) == 1
        assert provider.quarantine.get(sandbox) is None
        provider.close()
    asyncio.run(run())


def test_session_network_grants_survive_recreation_and_are_user_scoped(tmp_path, monkeypatch):
    backend = Backend()
    provider = AioSandboxProvider(backend, tmp_path, idle_timeout=0)
    monkeypatch.setattr(provider, "_make_sandbox", lambda info: Sandbox())
    user = require_user()
    try:
        sandbox = provider.acquire("first", user_id=user)
        provider.decide_network(user, "generation", ["event"], "example.com", 443, "allowSession")
        assert backend.decisions == [(user, "event", "allow_sandbox")]
        assert backend.grants == [(user, [("example.com", 443, None)])]
        provider.release(sandbox)
        provider.cleanup_idle()
        provider.acquire("second", user_id=user)
        assert backend.grants[-1] == (user, [("example.com", 443, None)])
        with bind_user("b" * 32):
            other = provider.acquire("other", user_id=require_user())
            assert len(backend.grants) == 2
            provider.release(other)
        provider.release(sandbox)
        provider.reset_user(user)
        provider.acquire("third", user_id=user)
        assert len(backend.grants) == 2
        provider.release(sandbox)
    finally:
        provider.close()


def test_reset_checks_old_resources_and_does_not_ignore_quarantine(tmp_path):
    backend = Backend()
    provider = AioSandboxProvider(backend, tmp_path)
    user = require_user()
    try:
        provider.quarantine.block(provider._sandbox_id(user), user, "generation", "结果未知")
        with pytest.raises(RuntimeError, match="资源状态未知"):
            provider.reset_user(user)
        assert backend.reconciled == 1
    finally:
        provider.close()
