import asyncio
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres import PostgresSaver as SyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app import db
from app.config import settings


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


async def _run_io(call, *args, **kwargs):
    task = asyncio.create_task(asyncio.to_thread(call, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


async def _open_manager(manager):
    task = asyncio.create_task(asyncio.to_thread(manager.__enter__))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled() and task.exception() is None:
            await _run_io(manager.__exit__, None, None, None)
        raise


# 同步驱动在线程中运行，兼容 Windows 的 Proactor 事件循环。
class PostgresSaver(SyncPostgresSaver):
    def __init__(self, conn, pipe=None, serde=None):
        super().__init__(conn, pipe=pipe, serde=serde)
        self.write_lock = asyncio.Lock()

    async def aget_tuple(self, *args, **kwargs):
        return await _run_io(self.get_tuple, *args, **kwargs)

    async def alist(self, *args, **kwargs):
        rows = await _run_io(lambda: list(self.list(*args, **kwargs)))
        for row in rows:
            yield row

    async def aput(self, *args, **kwargs):
        async with self.write_lock:
            return await _run_io(self.put, *args, **kwargs)

    async def aput_writes(self, *args, **kwargs):
        async with self.write_lock:
            return await _run_io(self.put_writes, *args, **kwargs)

    async def adelete_thread(self, *args, **kwargs):
        async with self.write_lock:
            return await _run_io(self.delete_thread, *args, **kwargs)


@asynccontextmanager
async def open_saver(data_dir):
    if settings.database_backend == "sqlite":
        async with SqliteSaver.from_conn_string(str(data_dir / "app.db")) as saver:
            await saver.conn.execute("PRAGMA journal_mode=WAL")
            await saver.conn.execute("PRAGMA busy_timeout=5000")
            await saver.setup()
            yield saver
    elif settings.database_backend == "postgresql":
        manager = PostgresSaver.from_conn_string(db.postgres_url())
        saver = await _open_manager(manager)
        try:
            await _run_io(saver.setup)
            yield saver
        finally:
            await _run_io(manager.__exit__, None, None, None)
    else:
        raise ValueError("DATABASE_BACKEND 必须是 sqlite 或 postgresql")
