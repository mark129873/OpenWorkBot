import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from app.catalog import Catalog
from app.config import Settings
from app import rag_parameters as params, tracing


class VectorStore:
    def __init__(self, config: Settings, catalog: Catalog):
        self.config = config
        self.catalog = catalog
        self.client = None
        self.lock = asyncio.Lock()

    def _connect(self):
        if not self.config.milvus_uri or not self.config.milvus_token:
            raise RuntimeError("Milvus Cloud 未配置")
        if self.client is None:
            self.client = MilvusClient(uri=self.config.milvus_uri, token=self.config.milvus_token)
        return self.client

    def _identity(self) -> str:
        value = f"{self.config.milvus_uri}\0{self.config.collection}"
        return hashlib.sha256(value.encode()).hexdigest()

    def _model_identity(self) -> str:
        path = Path(self.config.embedding_model).resolve()
        info = path.stat()
        return hashlib.sha256(f"{path}\0{info.st_size}\0{info.st_mtime_ns}".encode()).hexdigest()

    def _check(self, *, dimension: int | None = None) -> bool:
        client = self._connect()
        exists = client.has_collection(self.config.collection)
        owned = self.catalog.setting("collection")
        if exists and owned != self._identity():
            raise RuntimeError("Milvus 集合已存在且不属于此知识库，请先处理旧集合")
        if not exists and owned and self.catalog.versions():
            raise RuntimeError("知识库集合不存在，无法读取已保存的索引")
        if exists:
            model_id = self.catalog.setting("embedding_model")
            if model_id and model_id != self._model_identity():
                raise RuntimeError("Embedding 模型已改变，需要重新建立索引")
            return True
        if dimension is None:
            return False
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("document_id", DataType.VARCHAR, max_length=64)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("search_text", DataType.VARCHAR, max_length=65535,
                         enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("filename", DataType.VARCHAR, max_length=512)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(name="content_bm25", input_field_names=["search_text"],
                                     output_field_names=["sparse"], function_type=FunctionType.BM25))
        indexes = client.prepare_index_params()
        indexes.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        indexes.add_index(field_name="document_id", index_type="INVERTED")
        client.create_collection(self.config.collection, schema=schema, index_params=indexes,
                                 consistency_level="Strong")
        self.catalog.set_setting("collection", self._identity())
        self.catalog.set_setting("embedding_model", self._model_identity())
        return True

    async def check(self) -> bool:
        async with self.lock:
            return await asyncio.to_thread(self._check)

    async def add(self, version: str, filename: str, start: int,
                  chunks: list[dict], vectors: list[list[float]]) -> None:
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("文档分块与向量数量不匹配")
        rows = [{"chunk_id": uuid.uuid4().hex, "document_id": version,
                 "content": chunk["content"], "search_text": chunk["search_text"],
                 "chunk_index": start + index, "filename": filename, "dense": vectors[index],
                 "metadata": {"heading_path": chunk["heading_path"], "pages": chunk["pages"]}}
                for index, chunk in enumerate(chunks)]
        async with self.lock:
            await asyncio.to_thread(self._check, dimension=len(vectors[0]))
            await asyncio.to_thread(self.client.insert, self.config.collection, rows)

    async def search(self, query: str, vector: list[float], versions: list[str]) -> list[dict]:
        with tracing.observation("milvus_hybrid_search", as_type="retriever", input={"query": query},
                                 metadata={"collection": self.config.collection, "dense_limit": params.DENSE_TOP_K,
                                           "bm25_limit": params.BM25_TOP_K, "rrf_k": params.RRF_K,
                                           "limit": params.FUSED_TOP_K}) as span:
            async with self.lock:
                exists = await asyncio.to_thread(self._check)
                if not exists or not versions:
                    tracing.update(span, output=[])
                    return []
            expression = f"document_id in {json.dumps(versions)}"
            requests = [
                AnnSearchRequest(data=[vector], anns_field="dense", param={"metric_type": "COSINE"},
                                 limit=params.DENSE_TOP_K, filter=expression),
                AnnSearchRequest(data=[query], anns_field="sparse", param={"metric_type": "BM25"},
                                 limit=params.BM25_TOP_K, filter=expression),
            ]
            results = await asyncio.to_thread(self.client.hybrid_search, self.config.collection,
                                              reqs=requests, ranker=RRFRanker(params.RRF_K), limit=params.FUSED_TOP_K,
                                              output_fields=["document_id", "content", "search_text", "filename",
                                                             "chunk_index", "metadata"],
                                              consistency_level="Strong")
            hits = []
            for hit in results[0]:
                entity = dict(hit.get("entity", {}))
                metadata = entity.pop("metadata", {})
                hits.append({"id": str(hit["chunk_id"]), **entity, **metadata,
                             "score": float(hit["distance"])})
            tracing.update(span, output=hits)
            return hits

    async def delete(self, version: str) -> None:
        async with self.lock:
            if await asyncio.to_thread(self._check):
                await asyncio.to_thread(self.client.delete, self.config.collection,
                                        filter=f"document_id == {json.dumps(version)}")
