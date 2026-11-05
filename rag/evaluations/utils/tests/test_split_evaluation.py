import asyncio
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from langfuse import Langfuse
from openai import AsyncOpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from ragas.metrics.collections import AnswerCorrectness, Faithfulness

from evaluations import evaluate_generation as generation
from evaluations import evaluate_retrieval as retrieval
from app import prompts, rag_parameters as params, tracing
from app.graph import build_graph
from evaluations.utils.evaluation_utils import judge_settings, save_summary, summarize


def case():
    return {"id": "BF-001", "question": "问题", "reference": "参考答案", "behavior": "直接回答",
            "targets": ["A01 target.pdf"]}


@pytest.mark.parametrize("temperature,effort,expected", [
    (None, None, {"temperature": None, "reasoning_effort": "medium"}),
    ("", "", {"temperature": None, "reasoning_effort": "medium"}),
    ("0.5", " HIGH ", {"temperature": 0.5, "reasoning_effort": "high"}),
])
def test_judge_settings_from_environment(monkeypatch, temperature, effort, expected):
    for name, value in (("EVAL_TEMPERATURE", temperature), ("EVAL_REASONING_EFFORT", effort)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert judge_settings() == expected


@pytest.mark.parametrize("value", ["bad", "nan", "-0.1", "2.1"])
def test_invalid_judge_temperature(monkeypatch, value):
    monkeypatch.setenv("EVAL_TEMPERATURE", value)
    with pytest.raises(ValueError, match="EVAL_TEMPERATURE"):
        judge_settings()


def evidence(filename="A01 target.pdf", content="原文依据"):
    return {"id": filename, "filename": filename, "content": content, "document_id": filename, "score": 1.0}


async def parent_documents(ranked):
    return ranked


class Metric:
    def __init__(self, value=0.8, **kwargs):
        self.value = value
        self.inputs = []

    async def ascore(self, **inputs):
        self.inputs.append(inputs)
        return SimpleNamespace(value=self.value)


class GenerationModels:
    def __init__(self):
        self.prompts = []

    async def structured(self, system, prompt, default):
        assert system == prompts.GRADE_SYSTEM
        self.prompts.append(prompt)
        return {"sufficient": True, "can_retry": False, "missing": []}

    async def text(self, system, prompt):
        assert system == prompts.ANSWER_SYSTEM
        self.prompts.append(prompt)
        return "根据原文回答。"

    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_retrieval_stops_before_answer_and_reports_each_stage(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setattr(tracing, "configured_client", lambda: pytest.fail("关闭时不得连接 Langfuse"))
    monkeypatch.setattr(params, "RERANK_CANDIDATES", 1)

    class Models:
        def __init__(self):
            self.plans = self.grades = 0

        async def structured(self, system, prompt, default):
            if system == prompts.PLAN_SYSTEM:
                self.plans += 1
                return {"rewrite": True}
            if system == prompts.REWRITE_SYSTEM:
                return {"queries": [f"查询{self.plans}"]}
            self.grades += 1
            return {"sufficient": self.grades == 2, "can_retry": True, "missing": []}

        async def embed(self, texts):
            return [[1.0] for _ in texts]

        async def rerank(self, query, items):
            return items[:1]

        async def text(self, *args):
            pytest.fail("检索评测不得生成答案")

    class Vectors:
        def __init__(self):
            self.calls = 0

        async def search(self, query, vector, versions):
            self.calls += 1
            if self.calls == 1:
                return [evidence("A02 other.pdf")]
            return [{**evidence(), "score": 0.1}]

    models, vectors = Models(), Vectors()
    knowledge = SimpleNamespace(models=models, vectors=vectors, catalog_lock=asyncio.Lock(), parent_documents=parent_documents,
                                catalog=SimpleNamespace(versions=lambda: ["v1"]))
    record = await retrieval.evaluate_case(build_graph(knowledge), case(), "run-1")
    assert record["status"] == "success" and vectors.calls == 2 and models.grades == 2
    assert record["trace_id"] is None
    assert not {"candidates", "rerank_candidates", "seen", "sufficient", "can_retry", "missing", "attempt"} & record.keys()
    assert record["scores"] == {"candidate_document_recall": 1, "rerank_document_recall": 0,
                                "recall_at_5": 0, "ndcg_at_5": 0}
    assert record["evidence"][0]["filename"] == "A02 other.pdf"
    assert record["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_failed_retrieval_is_not_scored_or_exposed(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    class Graph:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("secret")

    record = await retrieval.evaluate_case(Graph(), case(), "run-1")
    assert record["status"] == "error" and record["error"] == "RuntimeError"
    assert all(value is None for value in record["scores"].values())
    assert "secret" not in json.dumps(record)


@pytest.mark.asyncio
async def test_retrieval_uploads_trace_and_full_candidates(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test-retrieval")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test-retrieval")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key="pk-lf-test-retrieval", secret_key="sk-lf-test-retrieval",
                      base_url="http://127.0.0.1:1", tracer_provider=TracerProvider(),
                      span_exporter=exporter, flush_at=1)
    monkeypatch.setattr(tracing, "configured_client", lambda: client)

    class Models:
        async def structured(self, system, prompt, default):
            with tracing.observation("llm", as_type="generation", input=prompt):
                if system == prompts.PLAN_SYSTEM:
                    return {"rewrite": True}
                if system == prompts.REWRITE_SYSTEM:
                    return {"queries": ["查询"]}
                return {"sufficient": True, "can_retry": False, "missing": []}

        async def embed(self, texts):
            with tracing.observation("embedding", as_type="embedding", input=texts):
                return [[1.0]]

        async def rerank(self, query, items):
            with tracing.observation("rerank", as_type="span") as span:
                tracing.update(span, output={"scores": [0.9]})
                return items

    class Vectors:
        async def search(self, query, vector, versions):
            with tracing.observation("milvus_hybrid_search", as_type="retriever") as span:
                hits = [{**evidence(), "search_text": "标题及原文"}]
                tracing.update(span, output=hits)
                return hits

    knowledge = SimpleNamespace(models=Models(), vectors=Vectors(), catalog_lock=asyncio.Lock(), parent_documents=parent_documents,
                                catalog=SimpleNamespace(versions=lambda: ["v1"]))
    try:
        record = await retrieval.evaluate_case(build_graph(knowledge), case(), "run-1")
        spans = exporter.get_finished_spans()
        trace_id = Langfuse.create_trace_id(seed="bank-rag-retrieval-run-1-BF-001")
        assert record["status"] == "success" and record["trace_id"] == trace_id
        assert not {"candidates", "rerank_candidates", "seen", "sufficient", "can_retry", "missing", "attempt"} & record.keys()
        assert all(span.context.trace_id == int(trace_id, 16) for span in spans)
        assert {"rag_search", "rag-langgraph", "plan", "rewrite", "retrieve", "rerank_stage", "grade",
                "llm", "embedding", "milvus_hybrid_search", "rerank"} <= {span.name for span in spans}
        assert not any(span.name == "answer" for span in spans)
        root = next(span for span in spans if span.name == "rag_search")
        output = json.loads(root.attributes["langfuse.observation.output"])
        assert output["candidates"][0]["search_text"] == "标题及原文"
        assert output["rerank_candidates"][0]["content"] == "原文依据"
        assert output["evidence"][0]["content"] == "原文依据"
        assert output["seen"] == ["查询"] and output["sufficient"] is True
    finally:
        client.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["setup", "update", "flush"])
async def test_trace_failure_is_reported_without_changing_results(monkeypatch, capsys, failure):
    class Span:
        def update(self, **kwargs):
            if failure == "update":
                raise RuntimeError("secret")

    class Client:
        def flush(self):
            if failure == "flush":
                raise RuntimeError("secret")

    class Graph:
        async def ainvoke(self, *args, **kwargs):
            return {"graded": True, "candidates": [evidence()], "rerank_candidates": [evidence()],
                    "evidence": [evidence()], "sufficient": True}

    @contextmanager
    def query(*args):
        yield None if failure == "setup" else Span()

    monkeypatch.setattr(tracing, "enabled", lambda: True)
    monkeypatch.setattr(tracing, "query", query)
    monkeypatch.setattr(tracing, "graph_config", lambda: None)
    monkeypatch.setattr(tracing, "configured_client", Client)
    record = await retrieval.evaluate_case(Graph(), case(), "run-1")
    assert record["status"] == "success" and record["scores"]["recall_at_5"] == 1
    assert record["trace_error"] == ("LangfuseUnavailable" if failure == "setup" else "RuntimeError")
    assert record["evidence"][0]["content"] == "原文依据"
    assert "candidates" not in record and "rerank_candidates" not in record
    assert "secret" not in json.dumps(record) + capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("offline", [False, True])
async def test_retrieval_checks_langfuse_before_running(tmp_path, monkeypatch, offline):
    dataset = tmp_path / "input-cases.jsonl"
    dataset.write_text(json.dumps(case(), ensure_ascii=False) + "\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "catalog.sqlite3").touch()
    config = SimpleNamespace(provider="openai", api_key="test", model="model", embedding_model="embed.gguf",
                             rerank_model="rerank.gguf", milvus_uri="test", milvus_token="test", data_dir=data_dir)

    class Client:
        def auth_check(self):
            if offline:
                raise RuntimeError("secret")
            return False

    monkeypatch.setattr(retrieval, "ROOT", tmp_path)
    monkeypatch.setattr(retrieval, "settings", lambda: config)
    monkeypatch.setattr(retrieval, "Knowledge", lambda config: SimpleNamespace(
        catalog=SimpleNamespace(versions=lambda: ["v1"])))
    monkeypatch.setattr(retrieval, "build_graph", lambda knowledge: pytest.fail("连接失败时不得开始检索"))
    monkeypatch.setattr(tracing, "configured_client", Client)
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(name, "test")
    with pytest.raises(SystemExit, match="Langfuse (连接|认证)失败") as exc:
        await retrieval.main(dataset, 1)
    assert "secret" not in str(exc.value)
    assert not (tmp_path / "evaluations" / "result").exists()


@pytest.mark.asyncio
async def test_fixed_evidence_is_used_without_reference_in_generation():
    item = {**case(), "evidence": [evidence()]}
    models, faithful, correct = GenerationModels(), Metric(0.8), Metric(0.7)
    record = await generation.evaluate_case(item, models, faithful, correct)
    assert record["status"] == "success"
    assert record["scores"] == {"faithfulness": 0.8, "answer_correctness": 0.7}
    assert all("原文依据" in prompt and "参考答案" not in prompt for prompt in models.prompts)
    assert faithful.inputs[0]["retrieved_contexts"] == ["原文依据"]
    assert faithful.inputs[0]["response"] == correct.inputs[0]["response"] == "根据原文回答。"
    assert correct.inputs[0]["reference"] == "参考答案"
    assert record["latency_ms"] >= 0 and record["scoring_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_generation_saves_and_scores_the_same_citation_free_answer():
    class Models(GenerationModels):
        async def text(self, system, prompt):
            return "事实【1】。\n\n参考来源：【1】A01 target.pdf"

    faithful, correct = Metric(0.8), Metric(0.7)
    record = await generation.evaluate_case({**case(), "evidence": [evidence()]}, Models(), faithful, correct)
    assert record["answer"] == "事实。"
    assert faithful.inputs[0]["response"] == correct.inputs[0]["response"] == record["answer"]
    assert faithful.inputs[0]["retrieved_contexts"] == ["原文依据"]


@pytest.mark.asyncio
async def test_empty_or_failed_evidence_is_not_misrepresented():
    models, faithful, correct = GenerationModels(), Metric(), Metric()
    item = {**case(), "evidence": []}
    record = await generation.evaluate_case(item, models, faithful, correct)
    assert record["status"] == "insufficient" and not models.prompts
    assert record["scores"]["faithfulness"] is None and not faithful.inputs
    assert record["scores"]["answer_correctness"] == 0.8
    failed = await generation.evaluate_case({**item, "status": "error"}, models, faithful, correct)
    assert failed["status"] == "error" and failed["error"] == "输入证据来自失败题"
    assert all(value is None for value in failed["scores"].values())


@pytest.mark.asyncio
async def test_metric_failure_preserves_other_score():
    class FailedMetric:
        async def ascore(self, **kwargs):
            raise ValueError("secret")

    record = await generation.evaluate_case({**case(), "evidence": [evidence()]},
                                            GenerationModels(), FailedMetric(), Metric(0.7))
    assert record["scores"] == {"faithfulness": None, "answer_correctness": 0.7}
    assert record["metric_errors"] == {"faithfulness": "ValueError"}


def test_manual_evidence_validation_and_summary():
    item = {"id": "Q1", "question": "问题", "reference": "参考答案",
            "evidence": [{"filename": "资料.pdf", "content": "原文"}]}
    data = json.dumps([item], ensure_ascii=False)
    assert generation.load_evidence(data, 1) == [item]
    assert generation.load_evidence(json.dumps({"cases": [item]}), 1) == [item]
    for invalid in (json.dumps([item, item]), "{}", "{", json.dumps([{**item, "evidence": [{}]}]),
                    json.dumps([{**item, "evidence": [{**item["evidence"][0], "score": float("nan")}]}])):
        with pytest.raises(ValueError):
            generation.load_evidence(invalid, 1)
    records = [{"status": "success", "scores": {"x": 0.5}, "latency_ms": 10},
               {"status": "insufficient", "scores": {"x": 1.0}, "latency_ms": 30},
               {"status": "error", "scores": {"x": None}, "latency_ms": 9000}]
    result = summarize(records, ("x",), 3)
    assert result["failed"] == 1 and result["scores"]["x"] == {"mean": 0.75, "scored": 2, "total": 3}
    assert result["latency_ms"] == {"mean": 20, "p50": 20, "p95": 29, "scored": 2, "total": 3}


def test_failed_json_update_preserves_completed_cases(tmp_path):
    path = tmp_path / "result.json"
    save_summary(path, {"cases": [{"id": "Q1"}]})
    with pytest.raises(ValueError):
        save_summary(path, {"cases": [{"id": "Q1"}, {"score": float("nan")}]})
    assert json.loads(path.read_text(encoding="utf-8")) == {"cases": [{"id": "Q1"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("trace_enabled", [False, True])
async def test_separate_scripts_save_and_replay_fixed_evidence(tmp_path, monkeypatch, request, trace_enabled):
    class Clock:
        @staticmethod
        def now():
            return datetime(2026, 11, 5, 18, 50)

    monkeypatch.setattr(retrieval, "datetime", Clock)
    monkeypatch.setattr(generation, "datetime", Clock)
    dataset = tmp_path / "input-cases.jsonl"
    item = case()
    item.pop("behavior")
    dataset.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    data_dir = tmp_path / "isolated"
    data_dir.mkdir()
    (data_dir / "catalog.sqlite3").touch()
    config = SimpleNamespace(provider="openai", api_key="test", model="model", embedding_model="embed.gguf",
                             rerank_model="rerank.gguf", milvus_uri="test", milvus_token="test",
                             collection="test", data_dir=data_dir, reasoning_effort="high", temperature=0.5)

    class Graph:
        async def ainvoke(self, state, *, config, interrupt_before):
            assert interrupt_before == ["answer"]
            return {"graded": True, "candidates": [evidence()], "rerank_candidates": [evidence()],
                    "evidence": [evidence()], "sufficient": True, "warning": "部分查询失败"}

    knowledge = SimpleNamespace(catalog=SimpleNamespace(versions=lambda: ["v1"]))
    monkeypatch.setattr(retrieval, "ROOT", tmp_path)
    monkeypatch.setattr(retrieval, "settings", lambda: config)
    monkeypatch.setattr(retrieval, "Knowledge", lambda config: knowledge)
    monkeypatch.setattr(retrieval, "build_graph", lambda knowledge: Graph())
    monkeypatch.setenv("LANGFUSE_ENABLED", str(trace_enabled).lower())
    if trace_enabled:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test-replay")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test-replay")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:1")
        client = Langfuse(public_key="pk-lf-test-replay", secret_key="sk-lf-test-replay",
                          base_url="http://127.0.0.1:1", tracer_provider=TracerProvider(),
                          span_exporter=InMemorySpanExporter(), flush_at=1)
        monkeypatch.setattr(client, "auth_check", lambda: True)
        monkeypatch.setattr(tracing, "configured_client", lambda: client)
        request.addfinalizer(client.shutdown)
    else:
        monkeypatch.setattr(tracing, "configured_client", lambda: pytest.fail("关闭时不得连接 Langfuse"))
    input_path = await retrieval.main(dataset, 1)
    snapshot = input_path.read_bytes()
    retrieval_summary = json.loads(snapshot)
    assert input_path == tmp_path / "evaluations" / "result" / "retrieval_11-05-18-50.json"
    assert retrieval_summary["run_id"] == "11-05-18-50"
    second_input = await retrieval.main(dataset, 1)
    assert second_input.name == "retrieval_11-05-18-50-02.json"
    assert input_path.read_bytes() == snapshot
    assert json.loads(second_input.read_bytes())["run_id"] == "11-05-18-50-02"
    assert retrieval_summary["dataset"] == dataset.name
    assert retrieval_summary["summary"]["scores"]["recall_at_5"]["mean"] == 1
    assert retrieval_summary["langfuse_enabled"] is trace_enabled
    assert retrieval_summary["temperature"] == 0.5
    assert retrieval_summary["rag_reasoning_effort"] == "high"
    assert "candidates" not in retrieval_summary["cases"][0]
    assert bool(retrieval_summary["cases"][0]["trace_id"]) is trace_enabled
    assert retrieval_summary["cases"][0]["warning"] == "部分查询失败"

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 3

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(generation, "ROOT", tmp_path)
    generation_config = SimpleNamespace(provider="openai", api_key="test", model="model", embedding_model="embed.gguf",
                                        reasoning_effort="high", temperature=None)
    monkeypatch.setattr(generation, "settings", lambda: generation_config)
    monkeypatch.setattr(generation, "Models", lambda config: GenerationModels())
    monkeypatch.setattr(generation, "AsyncOpenAI", Client)
    monkeypatch.setattr(generation, "create_judge", lambda model, client, options: object())
    monkeypatch.setattr(generation, "Faithfulness", Metric)
    monkeypatch.setattr(generation, "AnswerCorrectness", Metric)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("EVAL_MODEL_ID", "judge")
    monkeypatch.setenv("EVAL_REASONING_EFFORT", "medium")
    monkeypatch.setenv("EVAL_TEMPERATURE", "none")
    output_path = await generation.main(input_path, 1)
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path == tmp_path / "evaluations" / "result" / "generation_11-05-18-50.json"
    assert summary["run_id"] == "11-05-18-50"
    assert summary["answer_format"] == "citation_free_v1"
    assert summary["temperature"] is None and summary["judge_temperature"] is None
    assert summary["rag_reasoning_effort"] == "high" and summary["judge_reasoning_effort"] == "medium"
    assert summary["judge_request_retries"] == 3
    result = summary["cases"][0]
    assert input_path.read_bytes() == snapshot
    assert result["status"] == "success" and result["evidence"][0]["content"] == "原文依据"
    assert result["expected_behavior"] == ""
    assert "部分查询失败" in result["answer"]
    assert summary["summary"]["scores"]["faithfulness"]["scored"] == 1
    assert summary["evidence_sha256"] == hashlib.sha256(snapshot).hexdigest()
    manual_path = tmp_path / "manual.json"
    manual_path.write_text(json.dumps([{"id": "Q1", "question": "问题", "reference": "参考答案",
                                        "evidence": [{"filename": "资料.pdf", "content": "原文"}]}],
                                      ensure_ascii=False), encoding="utf-8")
    manual_output = await generation.main(manual_path, 1)
    assert manual_output.name == "generation_11-05-18-50-02.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary
    assert json.loads(manual_output.read_text(encoding="utf-8"))["run_id"] == "11-05-18-50-02"
    assert json.loads(manual_output.read_text(encoding="utf-8"))["cases"][0]["status"] == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_model", ["gpt-6.1-terra", "gpt-6-luna"])
@pytest.mark.parametrize("effort,temperature", [("medium", None), ("none", 0), ("none", 0.5)])
async def test_ragas_metrics_use_real_adapters_with_mocked_provider(judge_model, effort, temperature):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        prompt = json.dumps(payload["messages"], ensure_ascii=False)
        if "Your task is to judge the faithfulness" in prompt:
            output = {"statements": [{"statement": "Supported.", "reason": "Supported.", "verdict": 1},
                                     {"statement": "Unsupported.", "reason": "Missing.", "verdict": 0}]}
        elif "Given a ground truth and an answer statements" in prompt:
            output = {"TP": [{"statement": "Matched.", "reason": "Supported."}],
                      "FP": [{"statement": "Extra.", "reason": "Unsupported."}],
                      "FN": [{"statement": "Missing.", "reason": "Omitted."}]}
        else:
            output = {"statements": ["The rate is 3%."]}
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "judge", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(output)}}]})

    models = GenerationModels()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        async with AsyncOpenAI(api_key="test", base_url="http://test.invalid/v1", http_client=transport) as api:
            judge = generation.create_judge(judge_model, api, {"reasoning_effort": effort, "temperature": temperature})
            record = await generation.evaluate_case({**case(), "evidence": [evidence()]}, models,
                Faithfulness(llm=judge), AnswerCorrectness(llm=judge, embeddings=generation.ScoreEmbedding(models)))
    assert record["scores"] == {"faithfulness": 0.5, "answer_correctness": 0.625}
    assert not record["metric_errors"] and requests
    assert all(item["store"] is False and item["reasoning_effort"] == effort
               and item["max_completion_tokens"] == 4096
               and item["model"] == judge_model and "max_tokens" not in item and "top_p" not in item
               for item in requests)
    assert all(("temperature" not in item if temperature is None else item["temperature"] == temperature)
               for item in requests)
