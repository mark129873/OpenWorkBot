import asyncio
from abc import ABC, abstractmethod

from app.sandbox.sandbox import Sandbox


class SandboxProvider(ABC):
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
