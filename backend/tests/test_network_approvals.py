import asyncio
import threading

import pytest

from app import db
from app.agent.runtime import Conflict, Runtime
from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.config import settings
from app.sessions import chat_store as store
from app.user_context import require_user


class Provider:
    def __init__(self):
        self.info = SandboxInfo("sandbox", require_user(), "generation", "container", "proxy", "network",
                                "egress", "url", "token")
        self.events = [{"request_id": "one", "host": "example.com", "port": 443},
                       {"request_id": "two", "host": "example.org", "port": 443}]
        self.decisions = []
        self.resets = []
        self.fail = False

    def network_events(self, user_id):
        assert user_id == require_user()
        return self.info, list(self.events)

    def acknowledge_network_event(self, info, request_id):
        with db.new_session() as session:
            assert store.get(session, f"approval:network:{info.generation}:{request_id}")
        self.events = [event for event in self.events if event["request_id"] != request_id]

    def decide_network(self, *args):
        self.decisions.append(args)
        if self.fail:
            raise RuntimeError("connection lost")

    def reset_user(self, user_id, *, clear_grants=True):
        assert user_id == require_user()
        self.resets.append(user_id)

    def forget_network_grants(self, user_id):
        assert user_id == require_user()


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "modelio")
    db.dispose_engine()
    db.init_db(tmp_path)
    yield
    db.dispose_engine()


def test_network_approvals_are_separate_and_reuse_existing_choices(isolated_db):
    async def run():
        provider = Provider()
        runtime = Runtime(client=object(), sandbox_provider=provider)
        await runtime.start()
        try:
            await runtime.network_approvals.poll()
            snapshot = await runtime.snapshot()
            approvals = snapshot["approvals"]
            assert {item["input"]["host"] for item in approvals} == {"example.com", "example.org"}
            assert all(item["task_id"] is None and item["kind"] == "sandbox_network" for item in approvals)
            first = next(item for item in approvals if item["input"]["host"] == "example.com")
            assert [option["label"] for option in first["options"]] == ["允许 5 分钟", "本会话内始终允许", "拒绝"]
            result = await runtime.decide(first["request_id"], "allowOnce", assistant_id=snapshot["assistant_id"])
            assert result == {"ok": True, "replayed": False}
            assert provider.decisions == [(require_user(), "generation", ["one"], "example.com", 443, "allowOnce")]
            assert (await runtime.decide(first["request_id"], "allowOnce", assistant_id=snapshot["assistant_id"]))["replayed"]
            assert len((await runtime.snapshot())["approvals"]) == 1
        finally:
            await runtime.close()
    asyncio.run(run())


def test_uncertain_network_decision_is_not_replayed(isolated_db):
    async def run():
        provider = Provider()
        runtime = Runtime(client=object(), sandbox_provider=provider)
        await runtime.start()
        try:
            await runtime.network_approvals.poll()
            snapshot = await runtime.snapshot()
            request_id = snapshot["approvals"][0]["request_id"]
            provider.fail = True
            with pytest.raises(Conflict, match="结果未知"):
                await runtime.decide(request_id, "allowSession", assistant_id=snapshot["assistant_id"])
            with pytest.raises(Conflict):
                await runtime.decide(request_id, "allowSession", assistant_id=snapshot["assistant_id"])
            assert len(provider.decisions) == 1
            assert any("结果未知" in message["text"] for message in (await runtime.snapshot())["messages"])
        finally:
            await runtime.close()
    asyncio.run(run())


def test_assistant_reset_clears_network_session(isolated_db):
    async def run():
        provider = Provider()
        runtime = Runtime(client=object(), sandbox_provider=provider)
        await runtime.start()
        try:
            await runtime.network_approvals.poll()
            snapshot = await runtime.snapshot()
            approval = snapshot["approvals"][0]
            await runtime.decide(approval["request_id"], "allowSession", assistant_id=snapshot["assistant_id"])
            await runtime.reset(snapshot["assistant_id"], "network-reset")
            async with asyncio.timeout(8):
                while (await runtime.snapshot())["assistant_id"] == snapshot["assistant_id"]:
                    await asyncio.sleep(0.02)
            assert provider.resets == [require_user()]
            with pytest.raises(Conflict):
                await runtime.decide(approval["request_id"], "allowSession", assistant_id=snapshot["assistant_id"])
        finally:
            await runtime.close()
    asyncio.run(run())


def test_cancelled_reset_keeps_network_session_grants(isolated_db):
    class BlockingProvider(Provider):
        def __init__(self):
            super().__init__()
            self.started, self.release, self.finished = threading.Event(), threading.Event(), threading.Event()
            self.forgot = False

        def reset_user(self, user_id, *, clear_grants=True):
            assert not clear_grants
            self.started.set()
            assert self.release.wait(5)
            self.finished.set()

        def forget_network_grants(self, user_id):
            self.forgot = True

    async def run():
        provider = BlockingProvider()
        runtime = Runtime(client=object(), sandbox_provider=provider)
        await runtime.start()
        try:
            assistant_id = (await runtime.snapshot())["assistant_id"]
            await runtime.reset(assistant_id, "cancel-network-reset")
            assert await asyncio.to_thread(provider.started.wait, 5)
            await runtime.cancel_reset(assistant_id, "cancel-network-reset")
            provider.release.set()
            assert await asyncio.to_thread(provider.finished.wait, 5)
            assert not provider.forgot
            assert (await runtime.snapshot())["assistant_id"] == assistant_id
        finally:
            provider.release.set()
            await runtime.close()
    asyncio.run(run())
