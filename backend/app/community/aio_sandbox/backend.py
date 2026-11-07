from abc import ABC, abstractmethod

from app.community.aio_sandbox.sandbox_info import SandboxInfo


class SandboxBackend(ABC):
    @abstractmethod
    def create(self, thread_id: str, sandbox_id: str, extra_mounts=None, *, user_id: str) -> SandboxInfo:
        raise NotImplementedError

    @abstractmethod
    def destroy(self, info: SandboxInfo) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_alive(self, info: SandboxInfo) -> bool:
        raise NotImplementedError
