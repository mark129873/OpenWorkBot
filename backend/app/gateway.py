import asyncio

from sqlalchemy import select

from app import db
from app.agent.runtime import Runtime
from app.agent.memory import Memory
from app.agent.rag_client import RagClient
from app.models import User
from app.user_context import bind_user


class Gateway:
    def __init__(self, checkpointer) -> None:
        self.checkpointer = checkpointer
        self.memory = Memory()
        self.rag = RagClient()
        self.runtimes: dict[str, Runtime] = {}
        self.lock = asyncio.Lock()

    async def runtime(self, user_id: str) -> Runtime:
        async with self.lock:
            runtime = self.runtimes.get(user_id)
            if runtime is None:
                with bind_user(user_id):
                    runtime = Runtime(user_id=user_id, checkpointer=self.checkpointer, memory=self.memory, rag=self.rag)
                    await runtime.start()
                self.runtimes[user_id] = runtime
            return runtime

    async def start(self) -> None:
        with db.new_session() as session:
            users = list(session.scalars(select(User.id).where(User.disabled == 0)))
        for user_id in users:
            await self.runtime(user_id)

    async def close(self) -> None:
        for user_id, runtime in self.runtimes.items():
            with bind_user(user_id):
                await runtime.close()
        self.runtimes.clear()
        await self.memory.close()
        await self.rag.close()
