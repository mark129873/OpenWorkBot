import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from langfuse import Langfuse
from openai import AuthenticationError
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import models, rag_parameters as params, server, tracing
from app.config import Settings
from app.graph import build_graph
from app.knowledge import Knowledge
from app.vector_store import VectorStore


@pytest.mark.asyncio
async def test_disabled_langfuse_does_not_create_client_and_can_return_evidence(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setattr(tracing, "configured_client", lambda: pytest.fail("不应创建 Langfuse 客户端"))

    class FakeKnowledge:
        async def search(self, query, include_evidence=False):
            result = {"status": "success", "answer": query, "sources": []}
            if include_evidence:
                result["evidence"] = [{"filename": "A01.pdf", "content": "依据"}]
            return result

    monkeypatch.setattr(server, "knowledge", FakeKnowledge())
    assert "evidence" not in await server.rag_search("问题")
    result = await server.rag_search("问题", include_evidence=True)
    assert result["evidence"][0]["content"] == "依据"
    assert tracing.graph_config() is None


@pytest.mark.asyncio
async def test_mcp_query_traces_real_graph_state_without_changing_result(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test-trace")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test-trace")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-lf-test-trace", secret_key="sk-lf-test-trace",
                      base_url="http://127.0.0.1:1", tracer_provider=TracerProvider(),
                      span_exporter=exporter, flush_at=1)
    monkeypatch.setattr(tracing, "configured_client", lambda: client)

    class FakeModels:
        async def structured(self, system, prompt, default):
            if "规划器" in system:
                return {"rewrite": True}
            if "查询改写器" in system:
                return {"queries": ["改写问题"]}
            return {"sufficient": True}

        async def embed(self, texts):
            return [[1.0] for _ in texts]

        async def rerank(self, query, items):
            return items

        async def text(self, system, prompt):
            return "测试回答"

    class FakeVectors:
        async def search(self, query, vector, versions):
            return [{"id": "chunk-1", "document_id": "doc-1", "filename": "A01.pdf",
                     "content": "测试依据", "score": 1.0}]

    class FakeCatalog:
        def versions(self):
            return ["doc-1"]

    class FakeKnowledge:
        def __init__(self):
            self.models = FakeModels()
            self.vectors = FakeVectors()
            self.catalog = FakeCatalog()
            self.catalog_lock = asyncio.Lock()

        async def parent_documents(self, ranked):
            return ranked

        async def search(self, query):
            graph = build_graph(self)
            result = await graph.ainvoke({"query": query}, config=tracing.graph_config())
            return {"status": result["status"], "answer": result["answer"], "sources": result["sources"]}

    monkeypatch.setattr(server, "knowledge", FakeKnowledge())
    try:
        result = await server.rag_search("测试问题", "case-1")
        assert await server.rag_flush_tracing() == {"status": "success"}
        spans = exporter.get_finished_spans()
        trace_id = int(Langfuse.create_trace_id(seed="case-1"), 16)
        assert result["status"] == "success" and "测试回答" in result["answer"]
        assert result["sources"] == []
        assert any(span.name == "rag_search" and span.context.trace_id == trace_id for span in spans)
        graph_span = next(span for span in spans if span.name == "rag-langgraph")
        state = json.loads(graph_span.attributes["langfuse.observation.output"])
        assert graph_span.context.trace_id == trace_id
        assert state["evidence"][0]["content"] == "测试依据"
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_tracing_failure_does_not_break_mcp_query(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(key, "test")

    def fail():
        raise RuntimeError("secret must not be logged")

    class FakeKnowledge:
        async def search(self, query):
            return {"status": "success", "answer": query, "sources": []}

    monkeypatch.setattr(tracing, "configured_client", fail)
    monkeypatch.setattr(server, "knowledge", FakeKnowledge())
    assert await server.rag_search("问题", "case-2") == {"status": "success", "answer": "问题", "sources": []}


@pytest.mark.asyncio
async def test_metadata_failure_does_not_break_mcp_query(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(key, "test")

    class Span:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, **kwargs):
            pass

    class Client:
        def start_as_current_observation(self, **kwargs):
            return Span()

    class FakeKnowledge:
        async def search(self, query):
            return {"status": "success", "answer": query, "sources": []}

    def fail(**kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(tracing, "configured_client", Client)
    monkeypatch.setattr(tracing, "propagate_attributes", fail)
    monkeypatch.setattr(server, "knowledge", FakeKnowledge())
    assert (await server.rag_search("问题", "case-metadata"))["answer"] == "问题"


@pytest.mark.asyncio
async def test_parallel_queries_and_retries_have_correct_parents_and_evidence(tmp_path, monkeypatch):
    key = "pk-lf-test-tree"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", key)
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test-tree")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key=key, secret_key="sk-lf-test-tree", base_url="http://127.0.0.1:1",
                      tracer_provider=TracerProvider(), span_exporter=exporter, flush_at=1)
    monkeypatch.setattr(tracing, "configured_client", lambda: client)

    class FakeModels:
        def __init__(self):
            self.plans = 0
            self.grades = 0

        async def structured(self, system, prompt, default):
            with tracing.observation("llm", as_type="generation", input={"system": system, "prompt": prompt}):
                await asyncio.sleep(0)
                if "规划器" in system:
                    self.plans += 1
                    return {"rewrite": True, "multi_query": True, "subquery": True}
                if "查询改写器" in system:
                    return {"queries": [f"改写{self.plans}"]}
                if "多查询生成器" in system:
                    return {"queries": [f"多查{self.plans}"]}
                if "子问题拆分器" in system:
                    return {"queries": [f"子问题{self.plans}"]}
                self.grades += 1
                return {"sufficient": self.grades == 2, "can_retry": True, "missing": ["缺口"]}

        async def embed(self, texts):
            with tracing.observation("embedding", as_type="embedding", input=texts):
                await asyncio.sleep(0)
                return [[1.0]]

        async def rerank(self, query, items):
            with tracing.observation("rerank", as_type="span"):
                return items

        async def text(self, system, prompt):
            with tracing.observation("llm", as_type="generation", input={"system": system, "prompt": prompt}):
                return "测试回答"

    class FakeVectors:
        def __init__(self):
            self.calls = {}

        async def search(self, query, vector, versions):
            with tracing.observation("milvus_hybrid_search", as_type="retriever", input={"query": query}) as span:
                self.calls[query] = self.calls.get(query, 0) + 1
                if query == "改写1" and self.calls[query] == 1:
                    raise RuntimeError("temporary")
                hits = [{"id": query, "document_id": "doc-1", "filename": "A01.pdf",
                         "content": f"依据-{query}", "score": 1.0}]
                tracing.update(span, output=hits)
                return hits

    class FakeCatalog:
        def versions(self):
            return ["doc-1"]

    config = Settings(tmp_path, tmp_path, "", "", "collection", "", "", "", "", "", "", "")
    knowledge = Knowledge(config, catalog=FakeCatalog(), models=FakeModels(), vectors=FakeVectors())
    async def parent_documents(ranked):
        return ranked

    monkeypatch.setattr(knowledge, "parent_documents", parent_documents)
    monkeypatch.setattr(server, "knowledge", knowledge)
    try:
        result = await server.rag_search("原问题", "case-tree")
        client.flush()
        spans = exporter.get_finished_spans()
        by_id = {span.context.span_id: span for span in spans}
        parent = lambda span: by_id.get(span.parent.span_id) if span.parent else None
        root = next(span for span in spans if span.name == "rag_search")
        graph = next(span for span in spans if span.name == "rag-langgraph")
        assert parent(graph) == root
        assert all(span.context.trace_id == root.context.trace_id for span in spans)
        assert json.loads(root.attributes["langfuse.trace.metadata.operation_id"]) == "case-tree"
        assert json.loads(graph.attributes["langfuse.observation.metadata.queries_per_round"]) == params.QUERIES_PER_ROUND

        searches = [span for span in spans if span.name == "search_query"]
        assert len(searches) == 6 and all(parent(span).name == "retrieve" for span in searches)
        rerank_calls = [span for span in spans if span.name == "rerank"]
        assert len(rerank_calls) == 2 and all(parent(span).name == "rerank_stage" for span in rerank_calls)
        for span in searches:
            query = json.loads(span.attributes["langfuse.observation.input"])["query"]
            strategy = json.loads(span.attributes["langfuse.observation.metadata.strategy"])
            assert strategy == ("rewrite" if query.startswith("改写")
                                else "multi_query" if query.startswith("多查") else "subquery")
            children = [child for child in spans if parent(child) == span]
            assert all(child.name in {"embedding", "milvus_hybrid_search"} for child in children)
            assert any(child.name == "embedding" for child in children)
            assert any(child.name == "milvus_hybrid_search" for child in children)
            assert json.loads(span.attributes["langfuse.observation.output"])["attempts"] == (2 if query == "改写1" else 1)
        assert {json.loads(span.attributes["langfuse.observation.metadata.round"]) for span in searches} == {1, 2}
        assert all(parent(span).name in {"plan", "rewrite", "multi_query", "subquery", "grade", "answer"}
                   for span in spans if span.name == "llm")
        assert any(span.name == "grade_route" for span in spans)
        assert not any(span.name == "<unknown>" for span in spans)

        state = json.loads(graph.attributes["langfuse.observation.output"])
        assert state["answer"] == result["answer"]
        assert all("content" in item for item in state["evidence"])
        assert all("content" not in item for item in state["candidates"])
        assert all("content" not in item for item in state["rerank_candidates"])
        hits = [json.loads(span.attributes["langfuse.observation.output"])
                for span in spans if span.name == "milvus_hybrid_search"
                and "langfuse.observation.output" in span.attributes]
        assert all("content" in item for group in hits for item in group)
        for span in spans:
            if span.name in {"retrieve", "rerank_stage", "grade", "answer"}:
                for key in ("langfuse.observation.input", "langfuse.observation.output"):
                    if key in span.attributes:
                        value = json.loads(span.attributes[key])
                        if isinstance(value, dict):
                            assert all("content" not in item for field in ("candidates", "rerank_candidates", "evidence")
                                       for item in value.get(field, []))

        class FailedRerank(FakeModels):
            async def rerank(self, query, items):
                raise RuntimeError("rerank offline")

        monkeypatch.setattr(server, "knowledge", Knowledge(config, catalog=FakeCatalog(),
                                                             models=FailedRerank(), vectors=FakeVectors()))
        with pytest.raises(RuntimeError, match="rerank offline"):
            await server.rag_search("原问题", "case-failed-rerank")
        client.flush()
        failed_trace = int(Langfuse.create_trace_id(seed="case-failed-rerank"), 16)
        failed_hits = [json.loads(span.attributes["langfuse.observation.output"])
                       for span in exporter.get_finished_spans()
                       if span.context.trace_id == failed_trace and span.name == "milvus_hybrid_search"
                       and "langfuse.observation.output" in span.attributes]
        assert failed_hits and all("content" in item for group in failed_hits for item in group)
    finally:
        client.shutdown()


@pytest.mark.asyncio
async def test_failed_root_span_disables_nested_tracing(monkeypatch):
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(key, "test")

    class FailedClient:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("trace unavailable")

    class FakeKnowledge:
        async def search(self, query):
            assert tracing.graph_config() is None
            return {"status": "success", "answer": query, "sources": []}

    monkeypatch.setattr(tracing, "configured_client", lambda: FailedClient())
    monkeypatch.setattr(server, "knowledge", FakeKnowledge())
    assert (await server.rag_search("问题", "case-3"))["status"] == "success"


def test_partial_configuration_is_reported(monkeypatch, caplog):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    with tracing.query("问题", "case-4") as span:
        assert span is None
    assert "Langfuse 配置不完整" in caplog.text


@pytest.mark.asyncio
async def test_records_real_retrieval_and_model_operations(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    effort = "high"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test-operations")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test-operations")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-lf-test-operations", secret_key="sk-lf-test-operations",
                      base_url="http://127.0.0.1:1", tracer_provider=TracerProvider(),
                      span_exporter=exporter, flush_at=1)
    monkeypatch.setattr(tracing, "configured_client", lambda: client)
    for name, value in (("DENSE_TOP_K", 7), ("BM25_TOP_K", 8), ("FUSED_TOP_K", 9),
                        ("RRF_K", 11), ("EVIDENCE_TOP_K", 2), ("MAX_OUTPUT_TOKENS", 123)):
        monkeypatch.setattr(params, name, value)
    requests = {}

    class FakeHTTP:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            data = ({"data": [{"index": 0, "embedding": [0.1, 0.2]}]} if url.endswith("embeddings")
                    else {"results": [{"index": 0, "relevance_score": 0.9}]})
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, **kwargs):
            requests["model"] = kwargs
            usage = SimpleNamespace(input_tokens=4, output_tokens=2)
            return SimpleNamespace(status="completed", incomplete_details=None, output_text="测试回答", usage=usage)

    class FakeMilvus:
        def hybrid_search(self, *args, **kwargs):
            requests["milvus"] = kwargs
            return [[{"chunk_id": "c1", "entity": {"content": "测试依据", "search_text": "标题\n测试依据",
                                                    "metadata": {"heading_path": ["标题"], "pages": [1]},
                                                    "filename": "A01.pdf", "document_id": "v1"}, "distance": 0.9}]]

    monkeypatch.setattr(models.httpx, "AsyncClient", FakeHTTP)
    monkeypatch.setattr(models, "AsyncOpenAI", FakeOpenAI)
    config = Settings(tmp_path, tmp_path, "uri", "token", "collection", "http://embedding", "embed",
                      "http://rerank", "rerank", "key", "", "gpt-6-luna", "openai", effort, None)
    model = models.Models(config)
    vectors = VectorStore(config, None)
    vectors.client = FakeMilvus()
    monkeypatch.setattr(vectors, "_check", lambda: True)
    try:
        with tracing.query("问题", "case-5"):
            embeddings = await model.embed(["问题"])
            hits = await vectors.search("问题", embeddings[0], ["v1"])
            hits[0]["_rank_index"] = 0
            assert await model.rerank("问题", hits) == [
                {key: value for key, value in hits[0].items() if key != "search_text"}]
            assert await model.text("系统", "问题") == "测试回答"
        client.flush()
        spans = {span.name: span for span in exporter.get_finished_spans()}
        assert {"rag_search", "embedding", "milvus_hybrid_search", "rerank", "llm"} <= spans.keys()
        hit = json.loads(spans["milvus_hybrid_search"].attributes["langfuse.observation.output"])[0]
        assert hit["id"] == "c1" and hit["content"] == "测试依据"
        assert [req.limit for req in requests["milvus"]["reqs"]] == [7, 8]
        assert requests["milvus"]["limit"] == 9 and requests["milvus"]["ranker"].dict()["params"]["k"] == 11
        assert json.loads(spans["milvus_hybrid_search"].attributes["langfuse.observation.metadata.rrf_k"]) == 11
        assert json.loads(spans["rerank"].attributes["langfuse.observation.metadata.model"]) == "rerank"
        assert "content" not in json.loads(spans["rerank"].attributes["langfuse.observation.input"])["candidates"][0]
        assert requests["model"]["max_output_tokens"] == 123
        model_params = json.loads(spans["llm"].attributes["langfuse.observation.model.parameters"])
        assert model_params == {"max_output_tokens": 123, "reasoning_effort": effort, "store": False}
        assert requests["model"]["reasoning"]["effort"] == model_params["reasoning_effort"]
        assert "temperature" not in requests["model"]
        assert json.loads(spans["llm"].attributes["langfuse.observation.metadata.api"]) == "responses"
        assert json.loads(spans["llm"].attributes["langfuse.observation.metadata.finish_reason"]) == "completed"
        assert json.loads(spans["llm"].attributes["langfuse.observation.usage_details"]) == {"input": 4, "output": 2}

        class FailedOpenAI(FakeOpenAI):
            async def create(self, **kwargs):
                request = httpx.Request("POST", "http://test.invalid/responses")
                raise AuthenticationError("secret must not be recorded", response=httpx.Response(401, request=request), body={})

        monkeypatch.setattr(models, "AsyncOpenAI", FailedOpenAI)
        with tracing.query("问题", "case-model-error"):
            with pytest.raises(RuntimeError, match="401"):
                await model.text("系统", "问题")
        client.flush()
        failed_trace = int(Langfuse.create_trace_id(seed="case-model-error"), 16)
        failed = next(span for span in exporter.get_finished_spans()
                      if span.context.trace_id == failed_trace and span.name == "llm")
        assert json.loads(failed.attributes["langfuse.observation.metadata.error_type"]) == "AuthenticationError"
        assert json.loads(failed.attributes["langfuse.observation.metadata.http_status"]) == 401
        assert "secret must not be recorded" not in str(failed.attributes)
    finally:
        client.shutdown()
