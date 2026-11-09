import asyncio

from sqlalchemy import select

from app import db
from app.agent.runtime import Runtime
from app.agent.memory import Memory
from app.agent.rag_client import RagClient
from app.community.aio_sandbox.aio_sandbox_provider import AioSandboxProvider
from app.community.aio_sandbox.local_backend import LocalContainerBackend
from app.config import settings
from app.models import User
from app.sandbox.lease import SandboxLeaseManager
from app.user_context import bind_user


class Gateway:
    def __init__(self, checkpointer) -> None:
        self.checkpointer = checkpointer
        self.memory = Memory()
        self.rag = RagClient()
        backend = LocalContainerBackend(settings.data_dir, settings.workspaces_dir, settings.user_skills_dir,
                                        settings.sandbox_allow_domains)
        self.sandbox_provider = AioSandboxProvider(backend, settings.data_dir)
        self.sandbox_leases = SandboxLeaseManager(self.sandbox_provider)
        self.runtimes: dict[str, Runtime] = {}
        self.lock = asyncio.Lock()

    async def runtime(self, user_id: str) -> Runtime:
        async with self.lock:
            runtime = self.runtimes.get(user_id)
            if runtime is None:
                with bind_user(user_id):
                    runtime = Runtime(user_id=user_id, checkpointer=self.checkpointer, memory=self.memory, rag=self.rag,
                                      sandbox_provider=self.sandbox_provider, sandbox_leases=self.sandbox_leases)
                    await runtime.start()
                self.runtimes[user_id] = runtime
            return runtime

    async def start(self) -> None:
        with db.new_session() as session:
            users = list(session.scalars(select(User.id).where(User.disabled == 0)))
        for user_id in users:
            await self.runtime(user_id)

    async def close(self) -> None:
        error = None
        for user_id, runtime in self.runtimes.items():
            with bind_user(user_id):
                try:
                    await runtime.close()
                except Exception as exc:
                    error = error or exc
        self.runtimes.clear()
        try:
            try:
                await asyncio.to_thread(self.sandbox_provider.close)
            except Exception as exc:
                error = error or exc
        finally:
            try:
                await self.memory.close()
            finally:
                await self.rag.close()
        if error:
            raise RuntimeError("运行时或沙箱清理失败") from error
