import asyncio
import csv
import contextvars
import threading
from types import SimpleNamespace

import pytest

from app.community.aio_sandbox.aio_sandbox_provider import AioSandboxProvider
from app.community.aio_sandbox import local_backend
from app.community.aio_sandbox.local_backend import AIO_IMAGE, LocalContainerBackend
from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.sandbox.lease import SandboxLeaseManager
from app.user_context import bind_user, require_user


class Backend:
    instance = "instance"

    def __init__(self):
        self.created = []
        self.destroyed = []
        self.reconciled = 0

    def reconcile(self):
        self.reconciled += 1

    def create(self, thread_id, sandbox_id, *, user_id):
        info = SandboxInfo(sandbox_id=sandbox_id, sandbox_url="url", user_id=user_id, generation="generation",
                           container_id="container", proxy_id="proxy", network_id="network", egress_network_id="egress",
                           relay_token="token", request_headers={"X-DeerFlow-Relay-Token": "token"})
        self.created.append(info)
        return info

    def is_alive(self, info):
        return True

    def destroy(self, info):
        self.destroyed.append(info)


class Sandbox:
    def __init__(self, identity):
        self.id = identity
        self.closed = False

    def close(self):
        self.closed = True


def make_provider(tmp_path, monkeypatch, **kwargs):
    backend = Backend()
    provider = AioSandboxProvider(backend, tmp_path, **kwargs)
    monkeypatch.setattr(provider, "_make_sandbox", lambda info, handle: Sandbox(handle))
    return backend, provider


def test_handles_share_physical_container_without_cross_release(tmp_path, monkeypatch):
    backend, provider = make_provider(tmp_path, monkeypatch)
    user = require_user()
    try:
        first = provider.acquire("thread-a", user_id=user)
        second = provider.acquire("thread-b", user_id=user)
        assert first != second and len(backend.created) == 1
        assert provider.acquire("thread-a", user_id=user) == first
        assert provider.get(first).id == first
        assert provider.get_scoped(first, thread_id="thread-b", user_id=user) is None
        provider.release(first)
        assert provider.get(second).id == second and not backend.destroyed
        with bind_user("b" * 32):
            assert provider.get(second) is None
            with pytest.raises(PermissionError):
                provider.release(second)
            other = provider.acquire("other")
            assert len(backend.created) == 2
            provider.release(other)
        provider.release(second)
    finally:
        provider.close()
    assert len(backend.destroyed) == 2 and backend.reconciled == 1


def test_default_identity_and_distinct_threadless_handles(tmp_path, monkeypatch):
    backend, provider = make_provider(tmp_path, monkeypatch)
    context = contextvars.Context()
    try:
        first = context.run(provider.acquire)
        second = context.run(provider.acquire)
        assert first != second and len(backend.created) == 1
        assert backend.created[0].user_id == "default"
        assert context.run(provider.get, first).id == first
        with pytest.raises(PermissionError):
            context.run(provider.acquire, "thread", user_id="b" * 32)
        context.run(provider.release, first)
        context.run(provider.release, second)
    finally:
        provider.close()


def test_quarantine_waits_for_all_leases_before_recycle(tmp_path, monkeypatch):
    async def run():
        backend, provider = make_provider(tmp_path, monkeypatch)
        leases = SandboxLeaseManager(provider)
        user = require_user()
        first = await leases.acquire("a", "thread-a", user_id=user)
        second = await leases.acquire("b", "thread-b", user_id=user)
        assert first != second and len(backend.created) == 1
        provider.quarantine_sandbox(first, "命令停止结果未知")
        with pytest.raises(RuntimeError, match="沙箱已隔离"):
            await leases.acquire("c", "thread-c", user_id=user)
        await leases.release("a", user_id=user)
        assert not backend.destroyed
        await leases.release("b", user_id=user)
        assert len(backend.destroyed) == 1
        assert provider.quarantine.get(provider._sandbox_id(user)) is None
        provider.close()
    asyncio.run(run())


def test_reset_and_static_network_policy(tmp_path, monkeypatch):
    backend, provider = make_provider(tmp_path, monkeypatch)
    user = require_user()
    try:
        handle = provider.acquire("thread")
        assert provider.sandbox_network_mode() == "allowlist"
        assert provider.consume_network_policy_events(handle) == []
        assert not provider.decide_network_policy_request(handle, "event", "allow_sandbox")
        with pytest.raises(RuntimeError, match="运行中"):
            provider.reset_user(user)
        provider.release(handle)
        provider.reset_user(user)
        assert len(backend.destroyed) == 1
        provider.quarantine.block(provider._sandbox_id(user), user, "generation", "结果未知")
        with pytest.raises(RuntimeError, match="资源状态未知"):
            provider.reset_user(user)
    finally:
        provider.close()


def test_sandbox_info_does_not_serialize_relay_credentials():
    info = SandboxInfo("id", "http://localhost", container_id="container",
                       request_headers={"X-DeerFlow-Relay-Token": "secret"}, relay_token="secret")
    serialized = info.to_dict()
    assert "secret" not in repr(serialized) and "secret" not in repr(info)
    restored = SandboxInfo.from_dict(serialized)
    assert restored.sandbox_id == "id" and restored.container_id == "container"
    assert restored.request_headers == {} and not restored.relay_token


def test_default_workspace_and_extra_mount_boundary(tmp_path):
    backend = LocalContainerBackend(tmp_path / "data", tmp_path / "workspaces", tmp_path)
    assert backend._workspace("default") == tmp_path / "workspaces" / "default"
    with pytest.raises(ValueError, match="不支持额外挂载"):
        backend.create(None, "a" * 32, extra_mounts=[("host", "/other", True)], user_id=require_user())


def test_sandbox_mounts_only_current_user_skills_readonly(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    backend = LocalContainerBackend(tmp_path / "data", tmp_path / "workspaces", root)
    calls = []

    def docker(*args, **kwargs):
        calls.append(args)
        if args[0] == "port":
            return "127.0.0.1:12345"
        return "resource"

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(backend, "_docker", docker)
    monkeypatch.setattr(backend, "_check_engine", lambda: False)
    monkeypatch.setattr(backend, "_ensure_image", lambda image: None)
    monkeypatch.setattr(local_backend.httpx, "Client", Client)
    for user in ("a" * 32, "b" * 32):
        with bind_user(user):
            backend.create("thread", user, user_id=user)
    creations = [call for call in calls if call[0] == "create" and AIO_IMAGE in call]
    assert len(creations) == 2
    for call, user in zip(creations, ("a" * 32, "b" * 32)):
        mounts = [next(csv.reader([call[index + 1]])) for index, value in enumerate(call) if value == "--mount"]
        skill_mount = next(mount for mount in mounts if "dst=/mnt/skills" in mount)
        assert f"src={(root / user).resolve()}" in skill_mount
        assert "readonly" in skill_mount


def test_shutdown_during_creation_cleans_new_resources(tmp_path, monkeypatch):
    backend, provider = make_provider(tmp_path, monkeypatch)
    entered, finish = threading.Event(), threading.Event()
    create = backend.create
    errors = []

    def slow_create(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return create(*args, **kwargs)

    def acquire():
        with bind_user("a" * 32):
            try:
                provider.acquire("thread")
            except Exception as exc:
                errors.append(exc)

    monkeypatch.setattr(backend, "create", slow_create)
    thread = threading.Thread(target=acquire)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (provider.close(), closed.set()))
    thread.start()
    try:
        assert entered.wait(5)
        closer.start()
        assert not closed.wait(0.1)
    finally:
        finish.set()
        thread.join(5)
        if closer.ident is not None:
            closer.join(5)
    assert closed.is_set() and not closer.is_alive()
    assert not thread.is_alive() and len(errors) == 1
    assert "正在关闭" in str(errors[0]) and backend.created == backend.destroyed
