import asyncio

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class SqliteSaver(AsyncSqliteSaver):
    def __init__(self, conn):
        super().__init__(conn)
        self.write_lock = asyncio.Lock()

    async def aput(self, *args, **kwargs):
        async with self.write_lock:
            return await super().aput(*args, **kwargs)

    async def aput_writes(self, *args, **kwargs):
        async with self.write_lock:
            return await super().aput_writes(*args, **kwargs)

    async def adelete_thread(self, *args, **kwargs):
        async with self.write_lock:
            return await super().adelete_thread(*args, **kwargs)
