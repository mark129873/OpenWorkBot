import asyncio
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.config import Settings
from app.knowledge import Knowledge
from app.vector_store import VectorStore


def config(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = tmp_path / "embedding.gguf"
    model.write_bytes(b"model")
    return Settings(workspace, tmp_path / "data", "http://milvus.invalid", "secret", "test_collection",
                    "http://embedding.invalid", str(model), "http://rerank.invalid", "rerank",
                    "key", "http://llm.invalid", "model")


class FakeModels:
    def __init__(self):
        self.batches = []
        self.plans = 0
        self.answers = 0

    async def embed(self, texts):
        self.batches.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    async def rerank(self, query, items):
        return items[:10]

    async def structured(self, system, prompt, default):
        if "规划器" in system:
            self.plans += 1
            return {"original": self.plans == 1, "rewrite": self.plans > 1}
        if "查询生成器" in system:
            return {"queries": ["补充查询"]}
        return {"sufficient": False, "can_retry": True, "missing": ["更多资料"]}

    async def text(self, system, prompt):
        self.answers += 1
        return "依据已检索内容回答。"


class FakeVectors:
    def __init__(self):
        self.rows = {}
        self.fail_add = False
        self.fail_search = False

    async def add(self, version, filename, start, texts, vectors):
        if self.fail_add:
            raise RuntimeError("injected failure")
        self.rows.setdefault(version, []).extend(
            {"id": f"{version}-{start + index}", "document_id": version,
             "filename": filename, "content": text, "score": 1.0}
            for index, text in enumerate(texts))

    async def search(self, query, vector, versions):
        if self.fail_search:
            raise RuntimeError("offline")
        return [row for version in versions for row in self.rows.get(version, [])]

    async def delete(self, version):
        self.rows.pop(version, None)


@pytest.mark.asyncio
async def test_ingest_update_failure_and_remove_preserve_source(tmp_path):
    cfg = config(tmp_path)
    source = cfg.workspace / "note.txt"
    source.write_text("资料甲" * 10000, encoding="utf-8")
    models, vectors = FakeModels(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    first = await service.ingest("note.txt")
    previous = service.catalog.get(first["document_id"])["version"]
    assert max(models.batches) <= 32
    assert len(vectors.rows[previous]) > 32

    source.write_text("资料乙", encoding="utf-8")
    vectors.fail_add = True
    with pytest.raises(RuntimeError):
        await service.ingest("note.txt")
    assert service.catalog.get(first["document_id"])["version"] == previous
    assert previous in vectors.rows

    vectors.fail_add = False
    updated = await service.ingest("note.txt")
    assert updated["document_id"] == first["document_id"]
    assert service.catalog.get(first["document_id"])["version"] != previous
    assert previous not in vectors.rows
    await service.remove(first["document_id"])
    assert source.read_text(encoding="utf-8") == "资料乙"
    assert (await service.list_documents())["documents"] == []


@pytest.mark.asyncio
async def test_cancelled_update_keeps_old_version_after_restart(tmp_path):
    cfg = config(tmp_path)
    source = cfg.workspace / "note.txt"
    source.write_text("原资料", encoding="utf-8")
    vectors = FakeVectors()
    service = Knowledge(cfg, models=FakeModels(), vectors=vectors)
    document = await service.ingest("note.txt")
    previous = service.catalog.get(document["document_id"])["version"]
    source.write_text("新资料" * 10000, encoding="utf-8")

    class SlowModels(FakeModels):
        def __init__(self):
            super().__init__()
            self.waiting = asyncio.Event()

        async def embed(self, texts):
            if self.batches:
                self.waiting.set()
                await asyncio.Event().wait()
            return await super().embed(texts)

    service.models = SlowModels()
    pending = asyncio.create_task(service.ingest("note.txt"))
    await service.models.waiting.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    restarted = Knowledge(cfg, models=FakeModels(), vectors=vectors)
    assert restarted.catalog.versions() == [previous]
    assert previous in vectors.rows and len(vectors.rows) == 2
    assert (await restarted.search("资料"))["sources"][0]["filename"] == "note.txt"


@pytest.mark.asyncio
async def test_one_evidence_check_node_retries_at_most_once(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")
    models, vectors = FakeModels(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    result = await service.search("项目是什么")
    assert models.plans == 2
    assert models.answers == 1
    assert result["status"] == "insufficient"
    assert "a.md" in result["answer"] and result["sources"][0]["filename"] == "a.md"


@pytest.mark.asyncio
async def test_search_error_is_not_empty_knowledge(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")
    models, vectors = FakeModels(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    vectors.fail_search = True
    with pytest.raises(RuntimeError, match="检索失败"):
        await service.search("问题")


@pytest.mark.asyncio
async def test_empty_query_plan_still_searches_original_question(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class EmptyPlan(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"original": False, "rewrite": False, "subquery": False, "expand": False}
            return {"sufficient": True, "can_retry": False}

    models, vectors = EmptyPlan(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("问题"))["status"] == "success"


@pytest.mark.asyncio
async def test_empty_and_outside_files_are_rejected(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "empty.txt").write_text("", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    service = Knowledge(cfg, models=FakeModels(), vectors=FakeVectors())
    with pytest.raises(ValueError, match="未提取到文本"):
        await service.ingest("empty.txt")
    with pytest.raises(ValueError, match="工作区"):
        await service.ingest(str(outside))
    assert service.catalog.list() == []


@pytest.mark.asyncio
async def test_pdf_without_extractable_text_is_rejected(tmp_path):
    cfg = config(tmp_path)
    pdf = PdfWriter()
    pdf.add_blank_page(width=100, height=100)
    pdf.write(cfg.workspace / "scan.pdf")
    service = Knowledge(cfg, models=FakeModels(), vectors=FakeVectors())
    with pytest.raises(ValueError, match="扫描件"):
        await service.ingest("scan.pdf")
    assert service.catalog.list() == []


def test_existing_cloud_collection_is_not_adopted(tmp_path):
    cfg = config(tmp_path)
    service = Knowledge(cfg, models=FakeModels(), vectors=FakeVectors())
    store = VectorStore(cfg, service.catalog)

    class Existing:
        def has_collection(self, name):
            return True

    store.client = Existing()
    with pytest.raises(RuntimeError, match="不属于此知识库"):
        store._check()
