import asyncio
import json
import threading
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app import rag_parameters as params
from app.config import Settings
from app.graph import build_graph, query_limits
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
        return items

    async def structured(self, system, prompt, default):
        if "规划器" in system:
            self.plans += 1
            return {"rewrite": True}
        if any(label in system for label in ("查询改写器", "子问题拆分器", "查询扩展器")):
            return {"queries": [f"补充查询{self.plans}"]}
        return {"sufficient": False, "can_retry": True, "missing": ["更多资料"]}

    async def text(self, system, prompt):
        self.answers += 1
        return "依据已检索内容回答。"


class FakeVectors:
    def __init__(self):
        self.rows = {}
        self.fail_add = False
        self.fail_search = False

    async def add(self, version, filename, start, chunks, vectors):
        if self.fail_add:
            raise RuntimeError("injected failure")
        self.rows.setdefault(version, []).extend(
            {"id": f"{version}-{start + index}", "document_id": version,
             "filename": filename, **chunk, "score": 1.0}
            for index, chunk in enumerate(chunks))

    async def search(self, query, vector, versions):
        if self.fail_search:
            raise RuntimeError("offline")
        return [row for version in versions for row in self.rows.get(version, [])]

    async def delete(self, version):
        self.rows.pop(version, None)


class RecordingVectors(FakeVectors):
    def __init__(self):
        super().__init__()
        self.queries = []

    async def search(self, query, vector, versions):
        self.queries.append(query)
        return await super().search(query, vector, versions)


@pytest.mark.asyncio
async def test_ingest_update_failure_and_remove_preserve_source(tmp_path):
    cfg = config(tmp_path)
    source = cfg.workspace / "note.txt"
    source.write_text("资料甲" * 10000, encoding="utf-8")
    models, vectors = FakeModels(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    first = await service.ingest("note.txt")
    previous = service.catalog.get(first["document_id"])["version"]
    structure_dir = cfg.data_dir / "documents" / first["document_id"]
    previous_file = structure_dir / f"{previous}.json"
    assert json.loads(previous_file.read_text(encoding="utf-8"))["doc_id"] == first["document_id"]
    assert max(models.batches) <= 32
    assert len(vectors.rows[previous]) > 32

    source.write_text("资料乙", encoding="utf-8")
    vectors.fail_add = True
    with pytest.raises(RuntimeError):
        await service.ingest("note.txt")
    assert service.catalog.get(first["document_id"])["version"] == previous
    assert previous in vectors.rows
    assert list(structure_dir.glob("*.json")) == [previous_file]

    vectors.fail_add = False
    updated = await service.ingest("note.txt")
    assert updated["document_id"] == first["document_id"]
    assert service.catalog.get(first["document_id"])["version"] != previous
    assert previous not in vectors.rows
    assert not previous_file.exists()
    assert len(list(structure_dir.glob("*.json"))) == 1
    await service.remove(first["document_id"])
    assert source.read_text(encoding="utf-8") == "资料乙"
    assert (await service.list_documents())["documents"] == []
    assert not structure_dir.exists()


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
    assert [path.name for path in (cfg.data_dir / "documents" / document["document_id"]).glob("*.json")] == [f"{previous}.json"]
    assert (await restarted.search("资料"))["sources"] == []


@pytest.mark.asyncio
async def test_cancel_during_structure_write_leaves_no_partial_version(tmp_path, monkeypatch):
    from app import knowledge as knowledge_module

    cfg = config(tmp_path)
    (cfg.workspace / "note.txt").write_text("资料", encoding="utf-8")
    started, release = threading.Event(), threading.Event()
    original = knowledge_module.save_structure

    def slow_save(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(knowledge_module, "save_structure", slow_save)
    service = Knowledge(cfg, models=FakeModels(), vectors=FakeVectors())
    task = asyncio.create_task(service.ingest("note.txt"))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert service.catalog.list() == []
    assert not list((cfg.data_dir / "documents").rglob("*.json"))


@pytest.mark.asyncio
async def test_ingest_indexes_heading_and_keeps_raw_structure(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "note.md").write_text("# 制度\n\n## 材料\n\n材料原文\n", encoding="utf-8")
    models, vectors = FakeModels(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    result = await service.ingest("note.md")
    version = service.catalog.get(result["document_id"])["version"]
    structure = json.loads((cfg.data_dir / "documents" / result["document_id"] / f"{version}.json").read_text(encoding="utf-8"))
    assert structure["source"]["filename"] == "note.md"
    assert [node["text"] for node in structure["tree"]] == ["制度", "材料", "材料原文"]
    assert vectors.rows[version][0]["content"] == "材料原文"
    assert vectors.rows[version][0]["search_text"] == "制度\n材料\n材料原文"
    assert vectors.rows[version][0]["heading_path"] == ["制度", "材料"]


@pytest.mark.asyncio
async def test_update_reports_old_structure_cleanup_failure(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    source = cfg.workspace / "note.txt"
    source.write_text("原资料", encoding="utf-8")
    vectors = FakeVectors()
    service = Knowledge(cfg, models=FakeModels(), vectors=vectors)
    first = await service.ingest("note.txt")
    old = service.catalog.get(first["document_id"])["version"]
    source.write_text("新资料", encoding="utf-8")

    def fail(*_args):
        raise OSError("cleanup failed")

    monkeypatch.setattr("app.knowledge.remove_structure", fail)
    result = await service.ingest("note.txt")
    assert result["status"] == "success" and "结构化文件清理失败" in result["warning"]
    assert service.catalog.get(first["document_id"])["version"] != old
    assert old not in vectors.rows
    assert len(list((cfg.data_dir / "documents" / first["document_id"]).glob("*.json"))) == 2


@pytest.mark.asyncio
async def test_one_evidence_check_node_retries_at_most_once(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")
    models, vectors = FakeModels(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    result = await service.search("项目是什么", include_evidence=True)
    assert models.plans == 2
    assert models.answers == 1
    assert result["status"] == "insufficient"
    assert result["answer"] == "依据已检索内容回答。" and result["sources"] == []
    assert result["evidence"][0]["content"] == "项目资料"


@pytest.mark.asyncio
async def test_configured_supplemental_rounds_are_reachable(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "SUPPLEMENTAL_ROUNDS", 6)
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")
    models, vectors = FakeModels(), FakeVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    result = await service.search("项目是什么")
    assert models.plans == 7 and models.answers == 1
    assert result["status"] == "insufficient"


@pytest.mark.asyncio
async def test_configured_query_limits_control_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "QUERIES_PER_ROUND", 2)
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")

    class TunedModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True, "multi_query": True}
            if "查询改写器" in system:
                return {"queries": ["改写1", "改写2", "改写3"]}
            if "多查询生成器" in system:
                return {"queries": ["多查1", "多查2"]}
            return {"sufficient": True, "can_retry": False}

    models, vectors = TunedModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    result = await build_graph(service).ainvoke({"query": "问题", "attempt": 0, "seen": [], "candidates": []})
    assert result["rewrites"] == ["改写1"]
    assert vectors.queries == ["改写1", "多查1"]


@pytest.mark.parametrize(("plan", "expected"), [
    ({"rewrite": True, "multi_query": True, "expand": True, "subquery": True},
     {"rewrite": 1, "multi_query": 3, "expand": 2, "subquery": 2}),
    ({"multi_query": True}, {"multi_query": 8}),
    ({"expand": True, "subquery": True}, {"expand": 4, "subquery": 4}),
    ({"rewrite": True}, {"rewrite": 1}),
    ({}, {}),
])
def test_dynamic_query_limits(plan, expected, monkeypatch):
    monkeypatch.setattr(params, "QUERIES_PER_ROUND", 8)
    monkeypatch.setattr(params, "REWRITE_QUERY_LIMIT", 1)
    assert query_limits(plan) == expected


@pytest.mark.asyncio
async def test_four_strategies_share_query_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "QUERIES_PER_ROUND", 6)
    monkeypatch.setattr(params, "REWRITE_QUERY_LIMIT", 1)
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")

    class FourModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True, "multi_query": True, "expand": True, "subquery": True}
            for label, prefix in (("查询改写器", "改写"), ("多查询生成器", "多查"),
                                  ("查询扩展器", "扩展"), ("子问题拆分器", "拆分")):
                if label in system:
                    return {"queries": [f"{prefix}{index}" for index in range(20)]}
            return {"sufficient": True, "can_retry": False}

    models, vectors = FourModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    result = await build_graph(service).ainvoke({"query": "问题", "attempt": 0, "seen": [], "candidates": []})
    assert result["limits"] == {"rewrite": 1, "multi_query": 2, "expand": 2, "subquery": 1}
    assert [len(result[field]) for field in ("rewrites", "multi_queries", "expansions", "subqueries")] == [1, 2, 2, 1]
    assert len(vectors.queries) == 6 and "问题" not in vectors.queries


@pytest.mark.asyncio
async def test_retrieval_passes_ranked_candidates_to_rerank_stage(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")

    class RankedModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True}
            if "查询改写器" in system:
                return {"queries": ["检索问题"]}
            return {"sufficient": True, "can_retry": False}

        async def rerank(self, query, items):
            assert query == "原问题" and len(items) == 20
            assert [item["_rank_index"] for item in items] == list(range(20))
            assert [item["score"] for item in items] == list(range(39, 19, -1))
            assert all(item["content"] for item in items)
            return items

    class ManyVectors(FakeVectors):
        async def search(self, query, vector, versions):
            return [{"id": str(index), "document_id": versions[0], "filename": "a.md",
                     "content": f"内容{index}", "score": index} for index in range(40)]

    models, vectors = RankedModels(), ManyVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    result = await build_graph(service).ainvoke({"query": "原问题", "attempt": 0, "seen": [], "candidates": []})
    assert len(result["candidates"]) == 40 and len(result["rerank_candidates"]) == 20
    assert len(result["evidence"]) == 1 and result["status"] == "success"
    assert result["evidence"][0]["content"] == "项目资料"
    assert len(result["evidence"][0]["matched_chunks"]) == 20


@pytest.mark.asyncio
async def test_parent_documents_are_selected_after_reranking_all_chunks(tmp_path):
    cfg = config(tmp_path)

    class Models(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True}
            if "查询改写器" in system:
                return {"queries": ["检索问题"]}
            assert "文档0全部正文" in prompt and "文档4全部正文" in prompt
            return {"sufficient": True, "can_retry": False}

    class Vectors(FakeVectors):
        async def search(self, query, vector, versions):
            hits = []
            for index, version in enumerate(self.rows):
                for chunk in range(11 if index == 0 else 1):
                    hits.append({"id": f"{index}-{chunk}", "document_id": version,
                                 "filename": f"{index}.md", "content": "子块",
                                 "score": 100 - len(hits), "pages": [index + 1]})
            return hits

    vectors = Vectors()
    service = Knowledge(cfg, models=Models(), vectors=vectors)
    for index in range(6):
        (cfg.workspace / f"{index}.md").write_text(f"# 标题{index}\n\n文档{index}全部正文", encoding="utf-8")
        await service.ingest(f"{index}.md")
    state = await build_graph(service).ainvoke({"query": "原问题", "attempt": 0, "seen": [], "candidates": []})
    assert [item["filename"] for item in state["evidence"]] == [f"{index}.md" for index in range(5)]
    assert state["evidence"][0]["content"] == "标题0\n文档0全部正文"
    assert len(state["evidence"][0]["matched_chunks"]) == 11


@pytest.mark.asyncio
async def test_retrieval_evaluation_saves_full_parent_documents(tmp_path, monkeypatch):
    from evaluations.evaluate_retrieval import evaluate_case

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    cfg = config(tmp_path)
    (cfg.workspace / "A01 资料.md").write_text("# 标题\n\n申请材料\n\n支付额度", encoding="utf-8")
    models = FakeModels()
    service = Knowledge(cfg, models=models, vectors=FakeVectors())
    await service.ingest("A01 资料.md")
    case = {"id": "BF-001", "question": "问题", "reference": "答案", "targets": ["A01 资料.md"]}
    record = await evaluate_case(build_graph(service), case, "run-1")
    assert record["status"] == "success" and record["scores"]["recall_at_5"] == 1
    assert record["evidence"][0]["content"] == "标题\n申请材料\n支付额度"
    assert models.answers == 0


@pytest.mark.asyncio
async def test_parent_rejects_missing_or_invalid_structure(tmp_path):
    cfg = config(tmp_path)
    source = cfg.workspace / "a.md"
    source.write_text("# 标题\n\n正文", encoding="utf-8")
    vectors = FakeVectors()
    service = Knowledge(cfg, models=FakeModels(), vectors=vectors)
    document = await service.ingest("a.md")
    document_id = document["document_id"]
    current = service.catalog.get(document_id)["version"]
    assert (await service.parent_documents(vectors.rows[current]))[0]["content"] == "标题\n正文"
    structure = cfg.data_dir / "documents" / document_id / f"{current}.json"
    structure.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="父文档缺失或内容无效"):
        await service.parent_documents(vectors.rows[current])
    structure.unlink()
    with pytest.raises(RuntimeError, match="父文档缺失或内容无效"):
        await service.parent_documents(vectors.rows[current])


@pytest.mark.asyncio
async def test_duplicate_queries_are_removed_before_search_across_rounds(tmp_path, monkeypatch):
    monkeypatch.setattr(params, "QUERIES_PER_ROUND", 8)
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")

    class DuplicateModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                self.plans += 1
                return {"rewrite": True, "multi_query": True, "expand": True, "subquery": True}
            if "查询改写器" in system:
                return {"queries": ["共同"]}
            if "多查询生成器" in system:
                return {"queries": ["共同", "补充", "新增" if self.plans == 2 else "补充"]}
            if "查询扩展器" in system:
                return {"queries": ["共同", "扩展"]}
            if "子问题拆分器" in system:
                return {"queries": ["共同", "拆分"]}
            return {"sufficient": self.plans == 2, "can_retry": True, "missing": ["更多资料"]}

    models, vectors = DuplicateModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    result = await service.search("原问题")
    assert result["status"] == "success" and models.plans == 2
    assert vectors.queries == ["共同", "补充", "扩展", "拆分", "新增"]


@pytest.mark.asyncio
async def test_query_nodes_run_in_parallel_and_merge_once(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class ParallelModels(FakeModels):
        def __init__(self):
            super().__init__()
            self.started = set()
            self.ready = asyncio.Event()

        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True, "multi_query": True, "subquery": True, "expand": True}
            for label, query in (("查询改写器", "改写"), ("多查询生成器", "多查"),
                                 ("子问题拆分器", "拆分"), ("查询扩展器", "扩展")):
                if label in system:
                    self.started.add(label)
                    if len(self.started) == 4:
                        self.ready.set()
                    await asyncio.wait_for(self.ready.wait(), 2)
                    return {"queries": [query, query]}
            return {"sufficient": True, "can_retry": False}

    models, vectors = ParallelModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    graph = build_graph(service)
    assert set(graph.get_graph().nodes) - {"__start__", "__end__"} == {
        "plan", "rewrite", "multi_query", "subquery", "expand", "retrieve", "rerank_stage", "grade", "answer"}
    result = await graph.ainvoke({"query": "原问题", "attempt": 0, "seen": [], "candidates": []})
    assert result["status"] == "success"
    assert len(vectors.queries) == 4 and set(vectors.queries) == {"改写", "多查", "拆分", "扩展"}


@pytest.mark.asyncio
async def test_distinct_queries_search_in_parallel(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.md").write_text("项目资料", encoding="utf-8")

    class ParallelModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True, "multi_query": True}
            if "查询改写器" in system:
                return {"queries": ["改写"]}
            if "多查询生成器" in system:
                return {"queries": ["多查询"]}
            return {"sufficient": True, "can_retry": False}

    class ParallelVectors(FakeVectors):
        def __init__(self):
            super().__init__()
            self.started = set()
            self.ready = asyncio.Event()

        async def search(self, query, vector, versions):
            self.started.add(query)
            if len(self.started) == 2:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), 2)
            return await super().search(query, vector, versions)

    models, vectors = ParallelModels(), ParallelVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.md")
    assert (await service.search("原问题"))["status"] == "success"
    assert vectors.started == {"改写", "多查询"}


@pytest.mark.asyncio
async def test_query_branch_failure_keeps_other_branches(tmp_path, caplog):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class PartialModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True, "expand": True}
            if "查询改写器" in system:
                raise RuntimeError("offline")
            if "查询扩展器" in system:
                return {"queries": ["扩展查询"]}
            return {"sufficient": True, "can_retry": False}

    models, vectors = PartialModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("原问题"))["status"] == "success"
    assert set(vectors.queries) == {"扩展查询"}
    assert "rewrites" in caplog.text and "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_rewrite_only_searches_generated_query(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class RewriteModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True}
            if "查询改写器" in system:
                return {"queries": ["改写后的问题"]}
            assert "证据评估器" in system
            return {"sufficient": True, "can_retry": False}

    models, vectors = RewriteModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("原问题"))["status"] == "success"
    assert vectors.queries == ["改写后的问题"]


@pytest.mark.asyncio
async def test_strategy_output_matching_original_is_searched_only_once(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class SameModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True, "multi_query": True}
            if "查询改写器" in system:
                return {"queries": ["原问题"]}
            if "多查询生成器" in system:
                return {"queries": ["原问题", "另一种表达"]}
            return {"sufficient": True, "can_retry": False}

    models, vectors = SameModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("原问题"))["status"] == "success"
    assert vectors.queries == ["原问题", "另一种表达"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [True, False])
async def test_failed_or_empty_query_branches_fall_back_to_original(tmp_path, failure):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class FailedModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True, "expand": True}
            if "查询改写器" in system or "查询扩展器" in system:
                if failure:
                    raise RuntimeError("offline")
                return {"queries": ["", "  ", None]}
            return {"sufficient": True, "can_retry": False}

    models, vectors = FailedModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("原问题"))["status"] == "success"
    assert vectors.queries == ["原问题"]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_query", ["首次改写", None])
async def test_supplemental_fallback_does_not_repeat_original(tmp_path, first_query):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class FailedModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                self.plans += 1
                return {"rewrite": True}
            if "查询改写器" in system:
                if self.plans == 1 and first_query:
                    return {"queries": [first_query]}
                raise RuntimeError("offline")
            return {"sufficient": self.plans == 2, "can_retry": True}

    models, vectors = FailedModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("原问题"))["status"] == "success"
    assert vectors.queries == ([first_query, "原问题"] if first_query else ["原问题"])


@pytest.mark.asyncio
async def test_fallback_search_failure_is_reported(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class FailedModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True}
            raise RuntimeError("offline")

    vectors = RecordingVectors()
    service = Knowledge(cfg, models=FailedModels(), vectors=vectors)
    await service.ingest("a.txt")
    vectors.fail_search = True
    with pytest.raises(RuntimeError, match="检索失败"):
        await service.search("原问题")
    assert vectors.queries == ["原问题", "原问题"]


@pytest.mark.asyncio
async def test_recovery_uses_only_new_selected_queries(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class RecoveryModels(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                self.plans += 1
                return {"rewrite": True, "subquery": self.plans == 1, "expand": self.plans == 2}
            if "查询改写器" in system:
                return {"queries": [f"改写{self.plans}"]}
            if "子问题拆分器" in system:
                return {"queries": ["拆分"]}
            if "查询扩展器" in system:
                return {"queries": ["扩展"]}
            return {"sufficient": self.plans == 2, "can_retry": True}

    models, vectors = RecoveryModels(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("问题"))["status"] == "success"
    assert models.plans == 2
    assert len(vectors.queries) == 4 and set(vectors.queries) == {"改写1", "拆分", "改写2", "扩展"}


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
async def test_empty_query_plan_falls_back_to_original(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "a.txt").write_text("材料", encoding="utf-8")

    class EmptyPlan(FakeModels):
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": False, "multi_query": False, "subquery": False, "expand": False}
            return {"sufficient": True, "can_retry": False}

    models, vectors = EmptyPlan(), RecordingVectors()
    service = Knowledge(cfg, models=models, vectors=vectors)
    await service.ingest("a.txt")
    assert (await service.search("问题"))["status"] == "success"
    assert vectors.queries == ["问题"]


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


@pytest.mark.asyncio
async def test_invalid_pdf_does_not_leave_index_or_structure(tmp_path):
    cfg = config(tmp_path)
    (cfg.workspace / "broken.pdf").write_bytes(b"not a PDF")
    service = Knowledge(cfg, models=FakeModels(), vectors=FakeVectors())
    with pytest.raises(ValueError, match="PDF 解析失败"):
        await service.ingest("broken.pdf")
    assert service.catalog.list() == []
    assert not list((cfg.data_dir / "documents").rglob("*.json"))
    assert not list((cfg.data_dir / "staging").iterdir())


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
