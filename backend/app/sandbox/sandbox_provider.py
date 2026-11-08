import asyncio
from abc import ABC, abstractmethod

from app.sandbox.sandbox import Sandbox


class SandboxProvider(ABC):
    uses_thread_data_mounts = False
    needs_upload_permission_adjustment = True
    supports_agent_skill_isolation = False
    supports_lazy_agent_skill_isolation = False

    @abstractmethod
    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        raise NotImplementedError

    async def acquire_async(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        return await asyncio.to_thread(self.acquire, thread_id, user_id=user_id)

    @abstractmethod
    def get(self, sandbox_id: str) -> Sandbox | None:
        raise NotImplementedError

    @abstractmethod
    def get_scoped(self, sandbox_id: str, *, thread_id: str, user_id: str) -> Sandbox | None:
        raise NotImplementedError

    @abstractmethod
    def release(self, sandbox_id: str) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def sync_agent_skills(self, sandbox_id: str, *, thread_id: str, user_id: str, projection) -> None:
        raise NotImplementedError("本项目的 Skills 由所有 Agent 共享")

    async def sync_agent_skills_async(self, sandbox_id: str, *, thread_id: str, user_id: str, projection) -> None:
        await asyncio.to_thread(self.sync_agent_skills, sandbox_id, thread_id=thread_id,
                                user_id=user_id, projection=projection)

    def sandbox_network_mode(self) -> str:
        return "allowlist"

    def sandbox_network_temporary_grant_ttl(self) -> int:
        return 300

    def consume_network_policy_events(self, sandbox_id: str) -> list[dict]:
        return []

    async def consume_network_policy_events_async(self, sandbox_id: str) -> list[dict]:
        return await asyncio.to_thread(self.consume_network_policy_events, sandbox_id)

    def deny_pending_network_policy_events(self, sandbox_id: str) -> bool:
        return True

    async def deny_pending_network_policy_events_async(self, sandbox_id: str) -> bool:
        return await asyncio.to_thread(self.deny_pending_network_policy_events, sandbox_id)

    def decide_network_policy_request(self, sandbox_id: str, request_id: str, decision: str) -> bool:
        return False

    async def decide_network_policy_request_async(self, sandbox_id: str, request_id: str, decision: str) -> bool:
        return await asyncio.to_thread(self.decide_network_policy_request, sandbox_id, request_id, decision)
