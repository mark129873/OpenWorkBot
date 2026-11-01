import asyncio
import itertools
import logging
import uuid

from app.catalog import Catalog
from app.config import Settings
from app.documents import chunks, snapshot
from app.models import Models
from app.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Knowledge:
    def __init__(self, config: Settings, *, catalog=None, models=None, vectors=None):
        self.config = config
        self.catalog = catalog or Catalog(config.data_dir)
        self.models = models or Models(config)
        self.vectors = vectors or VectorStore(config, self.catalog)
        self.writes = asyncio.Lock()
        self.catalog_lock = asyncio.Lock()

    async def ingest(self, path: str) -> dict:
        async with self.writes:
            source, temporary = await asyncio.to_thread(snapshot, self.config.workspace, path,
                                                        self.config.data_dir / "staging")
            relative = source.relative_to(self.config.workspace).as_posix()
            version = uuid.uuid4().hex
            # shortcut: 中断暂存版本可能残留，需要回收时增加受控清理。
            self.catalog.stage(version, relative)
            count = 0
            try:
                iterator = iter(chunks(temporary, source.suffix))
                while batch := await asyncio.to_thread(lambda: list(itertools.islice(iterator, 32))):
                    vectors = await self.models.embed(batch)
                    await self.vectors.add(version, source.name, count, batch, vectors)
                    count += len(batch)
                if not count:
                    raise ValueError("未提取到文本；扫描件或空文件不能入库")
                async with self.catalog_lock:
                    document_id, old = self.catalog.activate(uuid.uuid4().hex, relative, source.name,
                                                             version, temporary.stat().st_size)
                    warning = ""
                    if old:
                        try:
                            await self.vectors.delete(old)
                        except Exception:
                            logger.warning("旧索引清理失败")
                            warning = "旧版本索引清理失败，当前检索仅使用新版本"
                return {"status": "success", "document_id": document_id, "path": relative,
                        "chunks": count, "warning": warning}
            except Exception:
                if version not in self.catalog.versions():
                    try:
                        await self.vectors.delete(version)
                    except Exception:
                        logger.warning("未完成的索引待清理")
                    else:
                        self.catalog.unstage(version)
                raise
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning("暂存文件仍被占用")

    async def list_documents(self) -> dict:
        return {"status": "success", "documents": self.catalog.list()}

    async def remove(self, document_id: str) -> dict:
        async with self.writes:
            async with self.catalog_lock:
                document = self.catalog.get(document_id)
                if not document:
                    raise ValueError("知识库中没有该文件")
                self.catalog.set_status(document_id, "deleting")
                await self.vectors.delete(document["version"])
                self.catalog.remove(document_id)
                return {"status": "success", "document_id": document_id, "path": document["path"]}

    async def search(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("检索问题不能为空")
        from app.graph import build_graph
        graph = build_graph(self)
        result = await graph.ainvoke({"query": query.strip(), "attempt": 0, "seen": [], "candidates": []})
        return {"status": result["status"], "answer": result["answer"], "sources": result.get("sources", [])}
