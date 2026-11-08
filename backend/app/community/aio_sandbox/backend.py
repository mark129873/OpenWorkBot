from abc import ABC, abstractmethod

from app.community.aio_sandbox.sandbox_info import SandboxInfo


class SandboxBackend(ABC):
    @abstractmethod
    def create(self, thread_id: str | None, sandbox_id: str, extra_mounts=None,
               *, user_id: str | None = None) -> SandboxInfo:
        raise NotImplementedError

    @abstractmethod
    def destroy(self, info: SandboxInfo) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_alive(self, info: SandboxInfo) -> bool:
        raise NotImplementedError

    @abstractmethod
    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        raise NotImplementedError

    def is_absent(self, sandbox_id: str) -> bool:
        return False

    def complete_absent_teardown(self, sandbox_id: str) -> None:
        if not self.is_absent(sandbox_id):
            raise RuntimeError("沙箱资源尚未清理")

    def inspect_runtime(self, sandbox_id: str) -> SandboxInfo | None:
        raise RuntimeError("沙箱资源状态不可用")

    def list_running(self) -> list[SandboxInfo]:
        return []
