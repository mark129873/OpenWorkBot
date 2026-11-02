import asyncio
import itertools
import logging
import uuid

from app.catalog import Catalog
from app.chunking import chunks
from app.config import Settings
from app.documents import read_document, remove_structure, remove_structures, save_structure, snapshot
from app.models import Models
from app.structure import clean_document, parse_document
from app.vector_store import VectorStore
from app import rag_parameters as params, tracing

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
            count = 0
            staged = False
            activated = False
            structure_path = None
            try:
                existing = self.catalog.get_by_path(relative)
                document_id = existing["id"] if existing else uuid.uuid4().hex
                # shortcut: 中断暂存版本可能残留，需要回收时增加受控清理。
                self.catalog.stage(version, relative)
                staged = True
                try:
                    document, edge_ids = await asyncio.to_thread(parse_document, temporary, source.suffix,
                                                                 source.name, document_id)
                except Exception as exc:
                    if source.suffix.lower() == ".pdf":
                        raise ValueError("PDF 解析失败") from exc
                    raise
                saving = asyncio.create_task(asyncio.to_thread(save_structure, self.config.data_dir,
                                                                document_id, version, document))
                try:
                    structure_path = await asyncio.shield(saving)
                except asyncio.CancelledError:
                    structure_path = await saving
                    raise
                await asyncio.to_thread(clean_document, document, edge_ids)
                iterator = iter(chunks(document))
                while batch := await asyncio.to_thread(lambda: list(itertools.islice(iterator, 32))):
                    vectors = await self.models.embed([chunk["search_text"] for chunk in batch])
                    await self.vectors.add(version, source.name, count, batch, vectors)
                    count += len(batch)
                if not count:
                    raise ValueError("未提取到文本；扫描件或空文件不能入库")
                async with self.catalog_lock:
                    document_id, old = self.catalog.activate(document_id, relative, source.name,
                                                             version, temporary.stat().st_size)
                    activated = True
                    warnings = []
                    if old:
                        try:
                            await self.vectors.delete(old)
                        except Exception:
                            logger.warning("旧索引清理失败")
                            warnings.append("旧版本索引清理失败，当前检索仅使用新版本")
                        try:
                            await asyncio.to_thread(remove_structure, self.config.data_dir, document_id, old)
                        except Exception:
                            logger.warning("旧结构化文件清理失败")
                            warnings.append("旧版本结构化文件清理失败")
                return {"status": "success", "document_id": document_id, "path": relative,
                        "chunks": count, "warning": "；".join(warnings)}
            except Exception:
                if staged and not activated:
                    try:
                        await self.vectors.delete(version)
                    except Exception:
                        logger.warning("未完成的索引待清理")
                    else:
                        self.catalog.unstage(version)
                raise
            finally:
                if structure_path and not activated:
                    try:
                        structure_path.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("未完成的结构化文件待清理")
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
                await asyncio.to_thread(remove_structures, self.config.data_dir, document_id)
                self.catalog.remove(document_id)
                return {"status": "success", "document_id": document_id, "path": document["path"]}

    async def parent_documents(self, ranked: list[dict]) -> list[dict]:
        selected = {}
        for chunk in ranked:
            version = chunk["document_id"]
            if version not in selected:
                if len(selected) >= params.EVIDENCE_TOP_K:
                    continue
                selected[version] = []
            selected[version].append({key: chunk[key] for key in ("id", "heading_path", "pages") if key in chunk})
        if not selected:
            return []
        async with self.catalog_lock:
            versions = {row["version"]: row for row in self.catalog.list() if row["status"] == "ready"}
            evidence = []
            for version, matched in selected.items():
                row = versions.get(version)
                if not row:
                    raise RuntimeError("父文档缺失或内容无效")
                try:
                    content = await asyncio.to_thread(read_document, self.config.data_dir, row["id"], version)
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    raise RuntimeError("父文档缺失或内容无效") from exc
                evidence.append({"id": version, "document_id": version, "filename": row["filename"],
                                 "content": content, "matched_chunks": matched})
            return evidence

    async def search(self, query: str, include_evidence: bool = False) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("检索问题不能为空")
        from app.graph import build_graph
        graph = build_graph(self)
        state = {"query": query.strip(), "attempt": 0, "seen": [], "candidates": []}
        config = tracing.graph_config() or {}
        config["recursion_limit"] = max(25, (params.SUPPLEMENTAL_ROUNDS + 1) * 5 + 5)
        if "callbacks" in config:
            config["metadata"] = {"prompt_version": tracing.prompt_version(),
                                  "rewrite_query_limit": params.REWRITE_QUERY_LIMIT,
                                  "queries_per_round": params.QUERIES_PER_ROUND,
                                  "rerank_candidates": params.RERANK_CANDIDATES,
                                  "evidence_top_k": params.EVIDENCE_TOP_K,
                                  "supplemental_rounds": params.SUPPLEMENTAL_ROUNDS,
                                  "search_retries": params.SEARCH_RETRIES,
                                  "json_retries": params.JSON_RETRIES,
                                  "model_request_retries": params.MODEL_REQUEST_RETRIES}
        result = await graph.ainvoke(state, config=config)
        response = {"status": result["status"], "answer": result["answer"], "sources": result.get("sources", [])}
        if include_evidence:
            response["evidence"] = result.get("evidence", [])
        return response
