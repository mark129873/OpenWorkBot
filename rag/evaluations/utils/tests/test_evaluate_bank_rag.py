import asyncio
import json
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langfuse import Langfuse
from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness

from evaluations import evaluate_generation, evaluate_retrieval
from evaluations.utils import evaluate_bank_rag as evaluation
from evaluations.utils import run_bank_rag as runner
from evaluations.utils import upload_bank_rag as upload
from evaluations.utils.benchmark import create_output


def test_evaluation_entry_roots_remain_the_rag_directory():
    root = Path(__file__).resolve().parents[3]
    assert all(module.ROOT == root for module in (evaluate_generation, evaluate_retrieval, evaluation, runner, upload))
    assert (root / "evaluations" / "benchmarks" / "bank-rag-v2-50.jsonl").is_file()


@pytest.mark.parametrize("script", ["evaluate_retrieval.py", "evaluate_generation.py",
                                    "utils/run_bank_rag.py", "utils/evaluate_bank_rag.py"])
def test_evaluation_scripts_start_directly_from_project_root(script):
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, str(root / "evaluations" / script), "--help"],
                            cwd=root.parent, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    if script in ("evaluate_retrieval.py", "utils/run_bank_rag.py"):
        assert b"--dataset" in result.stdout
    if script == "utils/evaluate_bank_rag.py":
        assert b"--run" in result.stdout


def benchmark_fixture(tmp_path):
    root = tmp_path / "benchmarks" / "bank-rag-benchmark"
    documents = root / "documents"
    documents.mkdir(parents=True)
    for number in range(1, 101):
        (documents / f"A{number:02}_file.pdf").touch()
    return root


def test_load_jsonl_selects_requested_number_of_cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    cases = [case(), {**case(), "id": "MH-001"}]
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in cases) + "\n", encoding="utf-8")
    assert evaluation.load_benchmark(path) == cases[:1]
    assert evaluation.load_benchmark(path, 50) == cases
    with pytest.raises(ValueError, match="大于零"):
        evaluation.load_benchmark(path, 0)


@pytest.mark.parametrize("problem", ["empty", "json", "duplicate", "target", "behavior", "id", "id_whitespace"])
def test_load_jsonl_rejects_invalid_records(tmp_path, problem):
    path = tmp_path / "cases.jsonl"
    item = case()
    if problem == "empty":
        content = "\n"
    elif problem == "json":
        content = "{invalid}\n"
    else:
        if problem == "target":
            item["targets"] = ["invalid.pdf"]
        if problem == "behavior":
            item["behavior"] = None
        if problem == "id":
            item["id"] = ""
        if problem == "id_whitespace":
            item["id"] = " BF-001 "
        content = json.dumps(item, ensure_ascii=False) + "\n"
        if problem == "duplicate":
            content *= 2
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        evaluation.load_benchmark(path, 50)


def test_load_jsonl_allows_missing_behavior(tmp_path):
    path = tmp_path / "cases.jsonl"
    item = case()
    item.pop("behavior")
    path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    assert evaluation.load_benchmark(path) == [{**item, "behavior": ""}]


def test_v2_dataset_contains_first_50_original_cases():
    path = evaluation.ROOT / "evaluations" / "benchmarks" / "bank-rag-v2-50.jsonl"
    cases = evaluation.load_benchmark(path, None)
    assert [item["id"] for item in cases] == [f"BF-{index:03}" for index in range(1, 51)]
    assert all(item["source"] == "Bank RAG Benchmark" and item["license"] == "CC BY 4.0"
               and item["behavior"] == "" for item in cases)
    assert [[name[:3] for name in cases[index]["targets"]] for index in (22, 23)] == [["A10", "A23"], ["A11", "A12"]]


@pytest.mark.asyncio
async def test_upload_entry_rebuilds_and_ingests_all_documents_without_workbook(tmp_path, monkeypatch):
    benchmark_fixture(tmp_path)
    documents = upload.load_documents(tmp_path)
    events = []

    class Session:
        def __init__(self):
            self.ingested = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            assert name == "rag_ingest"
            events.append("ingest")
            self.ingested.append(arguments["path"])
            return SimpleNamespace(isError=False, structuredContent={"status": "success"})

    class Models:
        def __init__(self, config):
            pass

        async def embed(self, texts):
            events.append("embedding")
            return [[1.0]]

    @asynccontextmanager
    async def stdio(server_params):
        assert server_params.args == ["-m", "app.server"]
        yield None, None

    session = Session()
    model = tmp_path / "embedding.gguf"
    model.touch()
    config = SimpleNamespace(workspace=tmp_path, embedding_model=str(model), milvus_uri="uri",
                             milvus_token="test", collection="test")
    monkeypatch.setattr(upload, "settings", lambda: config)
    monkeypatch.setattr(upload, "Models", Models)
    monkeypatch.setattr(upload, "reset_collection", lambda config: events.append("reset"))
    monkeypatch.setattr(upload, "stdio_client", stdio)
    monkeypatch.setattr(upload, "ClientSession", lambda *args: session)
    await upload.main()
    assert events[:2] == ["embedding", "reset"]
    assert session.ingested == [str(path) for path in documents]


@pytest.mark.asyncio
async def test_upload_does_not_reset_when_embedding_file_is_missing(tmp_path, monkeypatch):
    config = SimpleNamespace(workspace=tmp_path, embedding_model=str(tmp_path / "missing.gguf"),
                             milvus_uri="uri", milvus_token="test")
    monkeypatch.setattr(upload, "settings", lambda: config)
    monkeypatch.setattr(upload, "reset_collection", lambda config: pytest.fail("不能删除集合"))
    with pytest.raises(SystemExit) as error:
        await upload.main()
    assert error.value.code == 1


@pytest.mark.asyncio
async def test_upload_does_not_reset_when_embedding_is_unavailable(tmp_path, monkeypatch, capsys):
    model = tmp_path / "embedding.gguf"
    model.touch()
    config = SimpleNamespace(workspace=tmp_path, embedding_model=str(model), milvus_uri="uri", milvus_token="test")

    class Models:
        def __init__(self, config):
            pass

        async def embed(self, texts):
            raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(upload, "settings", lambda: config)
    monkeypatch.setattr(upload, "load_documents", lambda workspace: [])
    monkeypatch.setattr(upload, "Models", Models)
    monkeypatch.setattr(upload, "reset_collection", lambda config: pytest.fail("不能删除集合"))
    with pytest.raises(SystemExit) as error:
        await upload.main()
    assert error.value.code == 1 and "未删除集合" in capsys.readouterr().err


@pytest.mark.parametrize("exists", [True, False])
def test_reset_collection_clears_catalog_after_drop(tmp_path, monkeypatch, exists):
    config = SimpleNamespace(data_dir=tmp_path, milvus_uri="uri", milvus_token="token", collection="test")
    catalog = upload.Catalog(tmp_path)
    vectors = upload.VectorStore(config, catalog)
    catalog.set_setting("collection", vectors._identity())
    catalog.set_setting("embedding_model", "old")
    document_id, version = "1" * 32, "2" * 32
    catalog.activate(document_id, "old.pdf", "old.pdf", version, 1)
    structure_dir = tmp_path / "documents" / document_id
    structure_dir.mkdir(parents=True)
    (structure_dir / f"{version}.json").write_text("{}", encoding="utf-8")
    catalog.stage("staged", "other.pdf")
    dropped = []

    class Client:
        def has_collection(self, name):
            return exists

        def drop_collection(self, name):
            dropped.append(name)

        def close(self):
            pass

    monkeypatch.setattr(upload.VectorStore, "_connect", lambda self: Client())
    upload.reset_collection(config)
    assert dropped == (["test"] if exists else [])
    assert not catalog.list() and catalog.setting("collection") is None
    assert catalog.setting("embedding_model") is None
    assert not structure_dir.exists()
    with catalog.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM staged").fetchone()[0] == 0


def test_reset_rejects_unowned_collection_and_preserves_catalog(tmp_path, monkeypatch):
    config = SimpleNamespace(data_dir=tmp_path, milvus_uri="uri", milvus_token="token", collection="test")
    catalog = upload.Catalog(tmp_path)
    catalog.set_setting("collection", "other")
    catalog.activate("doc", "old.pdf", "old.pdf", "version", 1)

    class Client:
        def has_collection(self, name):
            return True

        def drop_collection(self, name):
            pytest.fail("不能删除不属于当前知识库的集合")

        def close(self):
            pass

    monkeypatch.setattr(upload.VectorStore, "_connect", lambda self: Client())
    with pytest.raises(SystemExit):
        upload.reset_collection(config)
    assert catalog.setting("collection") == "other" and len(catalog.list()) == 1


def test_failed_drop_preserves_catalog(tmp_path, monkeypatch):
    config = SimpleNamespace(data_dir=tmp_path, milvus_uri="uri", milvus_token="token", collection="test")
    catalog = upload.Catalog(tmp_path)
    catalog.set_setting("collection", upload.VectorStore(config, catalog)._identity())
    catalog.activate("doc", "old.pdf", "old.pdf", "version", 1)

    class Client:
        def has_collection(self, name):
            return True

        def drop_collection(self, name):
            raise RuntimeError("drop failed")

        def close(self):
            pass

    monkeypatch.setattr(upload.VectorStore, "_connect", lambda self: Client())
    with pytest.raises(RuntimeError, match="drop failed"):
        upload.reset_collection(config)
    assert len(catalog.list()) == 1 and catalog.setting("collection")


def test_failed_structure_cleanup_preserves_catalog(tmp_path, monkeypatch):
    config = SimpleNamespace(data_dir=tmp_path, milvus_uri="uri", milvus_token="token", collection="test")
    catalog = upload.Catalog(tmp_path)
    catalog.set_setting("collection", upload.VectorStore(config, catalog)._identity())
    catalog.activate("1" * 32, "old.pdf", "old.pdf", "2" * 32, 1)

    class Client:
        def has_collection(self, name):
            return True

        def drop_collection(self, name):
            pass

        def close(self):
            pass

    def fail(_data_dir):
        raise OSError("cleanup failed")

    monkeypatch.setattr(upload.VectorStore, "_connect", lambda self: Client())
    monkeypatch.setattr(upload, "remove_all_structures", fail)
    with pytest.raises(OSError, match="cleanup failed"):
        upload.reset_collection(config)
    assert len(catalog.list()) == 1 and catalog.setting("collection")


class FakeClient:
    def __init__(self, evidence):
        self.evidence = evidence
        self.api = SimpleNamespace(observations=SimpleNamespace(get_many=self.observations))

    def observations(self, **kwargs):
        assert kwargs["fields"] == "basic,io,usage"
        root = SimpleNamespace(name="rag_search", end_time=True, output=None, usage_details=None)
        graph = SimpleNamespace(name="rag-langgraph", end_time=True,
                                output=json.dumps({"evidence": self.evidence, "answer": "回答"}), usage_details=None)
        llm = SimpleNamespace(name="llm", end_time=True, output=None, usage_details={"input": 11, "output": 5})
        return SimpleNamespace(data=[root, graph, llm], meta=SimpleNamespace(cursor=None))


class FakeMetric:
    def __init__(self, value):
        self.value = value
        self.inputs = []

    async def ascore(self, **kwargs):
        self.inputs.append(kwargs)
        return SimpleNamespace(value=self.value)


def case():
    return {"id": "BF-001", "question": "问题", "reference": "标准答案", "behavior": "直接回答",
            "targets": ["A01 target.pdf"]}


def run_record(case_id="BF-001", planned=None):
    run_id = "12345678123456781234567812345678"
    return {**case(), "id": case_id, "run_id": run_id, "model": "gpt-6-luna",
            "parameters": {"dataset": "bank-rag-v1-50.jsonl", "cases": planned or [case_id]},
            "trace_id": Langfuse.create_trace_id(seed=f"bank-rag-{run_id}-{case_id}"),
            "status": "success", "answer": "回答", "latency_ms": 100.0, "error": None}


def test_output_ids_do_not_overwrite_existing_runs_or_reports(tmp_path):
    (tmp_path / "11-03-18-13.json").write_text("old", encoding="utf-8")
    first, first_path, file = create_output(tmp_path / "runs", "11-03-18-13", ".jsonl", conflicts=tmp_path)
    file.close()
    second, second_path, file = create_output(tmp_path / "runs", "11-03-18-13", ".jsonl", conflicts=tmp_path)
    file.close()
    assert (first, second) == ("11-03-18-13-02", "11-03-18-13-03")
    assert first_path.exists() and second_path.exists()
    assert (tmp_path / "11-03-18-13.json").read_text(encoding="utf-8") == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 50])
async def test_run_then_score_the_same_cases_without_uploads(tmp_path, monkeypatch, limit):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("EVAL_MODEL_ID", "gpt-6.1-terra")
    monkeypatch.setenv("EVAL_REASONING_EFFORT", "medium")
    monkeypatch.setenv("EVAL_TEMPERATURE", "none")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(name, "test")
    config = SimpleNamespace(provider="openai", model="gpt-6-luna", api_key="test",
                             embedding_model="embed", rerank_model="rerank", milvus_uri="uri",
                             milvus_token="test", workspace=tmp_path, base_url="http://test.invalid",
                             collection="collection", reasoning_effort="high", temperature=0.5)
    events = []
    judge_models = []
    selected = [{**case(), "id": f"BF-{index:03}"} for index in range(1, limit + 1)]

    class FakeLangfuse(FakeClient):
        create_trace_id = staticmethod(Langfuse.create_trace_id)

        def __init__(self):
            super().__init__([{"filename": "A01_target.pdf", "content": "实际证据"}])

        def auth_check(self):
            return True

        def shutdown(self):
            pass

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 3

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            events.append((name, arguments))
            if name == "rag_search":
                return SimpleNamespace(isError=False, structuredContent={"status": "success", "answer": "回答"})
            assert name == "rag_flush_tracing"
            return SimpleNamespace(isError=False, structuredContent={"status": "success"})

    @asynccontextmanager
    async def stdio(server_params):
        assert server_params.args == ["-m", "app.server"]
        assert server_params.env["OPENWORKBOT_BENCHMARK_FLUSH"] == "1"
        yield None, None

    def fake_judge(model, client, options):
        judge_models.append(model)
        assert options == {"reasoning_effort": "medium", "temperature": None}
        return object()

    dataset = tmp_path / "input-cases.jsonl"

    def load_selected(path, count):
        assert path == dataset
        return selected[:count]

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "settings", lambda: config)
    monkeypatch.setattr(runner, "load_benchmark", load_selected)
    monkeypatch.setattr(runner, "Langfuse", FakeLangfuse)
    monkeypatch.setattr(runner, "stdio_client", stdio)
    monkeypatch.setattr(runner, "ClientSession", Session)
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "settings", lambda: config)
    monkeypatch.setattr(evaluation, "Langfuse", FakeLangfuse)
    monkeypatch.setattr(evaluation, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(evaluation, "create_judge", fake_judge)
    monkeypatch.setattr(evaluation, "SharedFaithfulness", lambda **kwargs: FakeMetric(0.8))
    monkeypatch.setattr(evaluation, "SharedAnswerCorrectness", lambda **kwargs: FakeMetric(0.7))

    run_path = await runner.main(dataset, limit)
    records = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines()]
    assert run_path.parent == tmp_path / "evaluations" / "result" / "runs"
    assert re.fullmatch(r"\d{2}-\d{2}-\d{2}-\d{2}", run_path.stem)
    assert records[0]["run_id"] == run_path.stem
    assert records[0]["parameters"]["temperature"] == 0.5
    assert records[0]["parameters"]["rag_reasoning_effort"] == "high"
    assert [item["id"] for item in records] == [item["id"] for item in selected]
    assert [event[0] for event in events] == ["rag_search"] * limit + ["rag_flush_tracing"]
    assert all(event[1]["operation_id"].endswith(item["id"]) for event, item in zip(events, records))

    report_path = await evaluation.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path.parent == tmp_path / "evaluations" / "result"
    assert report_path.stem == run_path.stem == report["batch_id"] == report["evaluation_id"]
    assert report["model"] == "gpt-6-luna"
    assert report["parameters"]["judge_model"] == "gpt-6.1-terra"
    assert report["parameters"]["judge_temperature"] is None and report["parameters"]["temperature"] == 0.5
    assert report["parameters"]["reasoning_effort"] == "medium"
    assert report["parameters"]["judge_request_retries"] == 3
    assert report["parameters"]["answer_format"] == "citation_free_v1"
    assert judge_models == ["gpt-6.1-terra"]
    assert report["parameters"]["dataset"] == dataset.name
    assert report["parameters"]["cases"] == [item["id"] for item in selected]
    assert report["parameters"]["langfuse_enabled"] is True
    assert [item["id"] for item in report["cases"]] == [item["id"] for item in selected]
    assert [item["trace_id"] for item in report["cases"]] == [item["trace_id"] for item in records]
    assert report["run_id"] == records[0]["run_id"]
    assert report["summary"]["recall_at_5"]["total"] == limit
    assert report["summary"]["mcp_latency_ms"]["scored"] == limit
    assert report["summary"]["mcp_latency_ms"]["total"] == limit
    assert all(item["scores"]["faithfulness"] == 0.8 and not item["errors"]
               for item in report["cases"])
    assert len(events) == limit + 1
    if limit == 1:
        second_report = await evaluation.main(run_path)
        assert second_report.stem == f"{run_path.stem}-02"
        assert len(events) == limit + 1
        legacy = tmp_path / "01-01-00-00.jsonl"
        legacy.write_text(json.dumps(run_record(), ensure_ascii=False) + "\n", encoding="utf-8")
        legacy_path = await evaluation.main(legacy)
        legacy_report = json.loads(legacy_path.read_text(encoding="utf-8"))
        assert legacy_path.stem == legacy_report["batch_id"] == "01-01-00-00"
        assert legacy_report["run_id"] == run_record()["run_id"]
    else:
        partial = run_path.parent / "partial.jsonl"
        partial.write_text(json.dumps(records[0], ensure_ascii=False) + "\n", encoding="utf-8")
        partial_report = json.loads((await evaluation.main(partial)).read_text(encoding="utf-8"))
        assert partial_report["missing_cases"] == [item["id"] for item in selected[1:]]
        assert partial_report["summary"]["recall_at_5"] == {"mean": 1, "scored": 1, "total": 50}
        assert partial_report["summary"]["mcp_latency_ms"]["scored"] == 1
        assert partial_report["summary"]["mcp_latency_ms"]["total"] == 50
        with pytest.raises(SystemExit, match="批次未完成"):
            await evaluation.main()


@pytest.mark.asyncio
async def test_both_stages_stop_when_langfuse_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setattr(runner, "settings", lambda: pytest.fail("不应启动 RAG"))
    monkeypatch.setattr(evaluation, "settings", lambda: pytest.fail("不应开始评分"))
    with pytest.raises(SystemExit, match="LANGFUSE_ENABLED"):
        await runner.main(tmp_path / "input-cases.jsonl")
    with pytest.raises(SystemExit, match="LANGFUSE_ENABLED"):
        await evaluation.main(tmp_path / "run.jsonl")


@pytest.mark.asyncio
async def test_scoring_requires_explicit_evaluation_model(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(name, "test")
    monkeypatch.delenv("EVAL_MODEL_ID", raising=False)
    config = SimpleNamespace(provider="openai", model="gpt-6-luna", api_key="test", embedding_model="embed")
    monkeypatch.setattr(evaluation, "settings", lambda: config)
    monkeypatch.setattr(evaluation, "Langfuse", lambda: pytest.fail("不应连接 Langfuse"))
    with pytest.raises(SystemExit, match="EVAL_MODEL_ID"):
        await evaluation.main(tmp_path / "run.jsonl")


@pytest.mark.parametrize("problem", ["mixed_run", "wrong_trace", "unknown_case", "bad_latency", "missing_answer"])
def test_load_run_rejects_mismatched_or_incomplete_cases(tmp_path, problem):
    planned = ["BF-001", "BF-002"]
    items = [run_record("BF-001", planned), run_record("BF-002", planned)]
    if problem == "mixed_run":
        items[1]["run_id"] = "a" * 32
    elif problem == "wrong_trace":
        items[1]["trace_id"] = items[0]["trace_id"]
    elif problem == "unknown_case":
        items[1]["id"] = "BF-003"
    elif problem == "bad_latency":
        items[1]["latency_ms"] = -1
    else:
        items[1]["answer"] = None
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="RAG 批次题目"):
        evaluation.load_run(path)


@pytest.mark.asyncio
async def test_runner_keeps_completed_case_if_trace_flush_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
        monkeypatch.setenv(name, "test")
    config = SimpleNamespace(provider="openai", model="gpt-6-luna", api_key="test", embedding_model="embed",
                             rerank_model="rerank", milvus_uri="uri", milvus_token="test", collection="test",
                             reasoning_effort="none", temperature=0)

    class Client:
        create_trace_id = staticmethod(Langfuse.create_trace_id)

        def auth_check(self):
            return True

        def shutdown(self):
            pass

    class Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            if name == "rag_search":
                return SimpleNamespace(isError=False, structuredContent={"status": "success", "answer": "回答"})
            assert name == "rag_flush_tracing"
            return SimpleNamespace(isError=True)

    @asynccontextmanager
    async def stdio(server_params):
        yield None, None

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "settings", lambda: config)
    monkeypatch.setattr(runner, "load_benchmark", lambda path, limit: [case()])
    monkeypatch.setattr(runner, "Langfuse", Client)
    monkeypatch.setattr(runner, "stdio_client", stdio)
    monkeypatch.setattr(runner, "ClientSession", Session)
    with pytest.raises(RuntimeError):
        await runner.main(tmp_path / "input-cases.jsonl")
    run_path = next((tmp_path / "evaluations" / "result" / "runs").glob("*.jsonl"))
    assert evaluation.load_run(run_path)[0]["id"] == "BF-001"


@pytest.mark.asyncio
async def test_case_scores_trace_evidence_without_posting_scores():
    evidence = [{"filename": "A01_target.pdf", "content": "实际证据"}]
    client = FakeClient(evidence)
    faithful, correct = FakeMetric(0.8), FakeMetric(0.7)
    record = await evaluation.evaluate_case(client, run_record(), faithful, correct, "eval-1")
    assert record["trace_id"] == run_record()["trace_id"]
    assert record["scores"] == {"recall_at_5": 1, "ndcg_at_5": 1, "faithfulness": 0.8, "answer_correctness": 0.7}
    assert record["tokens"] == {"input": 11, "output": 5}
    assert record["scoring_latency_ms"] >= 0
    assert set(record["metric_latency_ms"]) == {"faithfulness", "answer_correctness"}
    assert faithful.inputs[0]["retrieved_contexts"] == ["实际证据"]
    assert correct.inputs[0]["reference"] == "标准答案"
    assert "score_write" not in record
    assert not record["errors"]


@pytest.mark.asyncio
async def test_historical_trace_saves_and_scores_one_citation_free_answer(monkeypatch):
    raw = "单笔100万元【1】。\n\n参考来源：【1】A01_target.pdf\n\n提示：部分查询失败，结果可能不完整。"
    evidence = [{"filename": "A01_target.pdf", "content": "实际证据"}]

    async def trace(*args):
        return {"answer": raw, "evidence": evidence}, {"input": 1, "output": 1}

    monkeypatch.setattr(evaluation, "trace_state", trace)
    faithful, correct = FakeMetric(0.8), FakeMetric(0.7)
    record = await evaluation.evaluate_case(None, {**run_record(), "answer": raw}, faithful, correct, "eval-1")
    assert record["answer"] == "单笔100万元。\n\n提示：部分查询失败，结果可能不完整。"
    assert faithful.inputs[0]["response"] == correct.inputs[0]["response"] == record["answer"]
    assert faithful.inputs[0]["retrieved_contexts"] == ["实际证据"] and not record["errors"]


@pytest.mark.asyncio
async def test_trace_mismatch_is_checked_before_citation_cleanup(monkeypatch):
    raw = "事实【1】。\n\n参考来源：【1】A01_target.pdf"

    async def trace(*args):
        return {"answer": raw.replace("【1】", "【2】"), "evidence": []}, None

    monkeypatch.setattr(evaluation, "trace_state", trace)
    faithful, correct = FakeMetric(0.8), FakeMetric(0.7)
    record = await evaluation.evaluate_case(None, {**run_record(), "answer": raw}, faithful, correct, "eval-1")
    assert record["answer"] == "事实。" and record["errors"]["trace_or_retrieval"] == "ValueError"
    assert not faithful.inputs and not correct.inputs


@pytest.mark.asyncio
async def test_citation_only_trace_is_not_scored(monkeypatch):
    raw = "【1】\n\n参考来源：【1】A01_target.pdf"

    async def trace(*args):
        return {"answer": raw, "evidence": [{"filename": "A01_target.pdf", "content": "依据"}]}, None

    monkeypatch.setattr(evaluation, "trace_state", trace)
    faithful, correct = FakeMetric(0.8), FakeMetric(0.7)
    record = await evaluation.evaluate_case(None, {**run_record(), "answer": raw}, faithful, correct, "eval-1")
    assert record["status"] == "error" and record["answer"] == ""
    assert record["errors"]["answer"] == "清除引用后回答为空"
    assert not faithful.inputs and not correct.inputs


@pytest.mark.asyncio
async def test_failed_rag_is_reported_without_scoring():
    client = FakeClient([])
    client.api.observations.get_many = lambda **kwargs: pytest.fail("失败的 RAG 不应读取追踪")
    faithful, correct = FakeMetric(0.8), FakeMetric(0.7)
    item = {**run_record(), "status": "error", "answer": None, "error": "RuntimeError"}
    record = await evaluation.evaluate_case(client, item, faithful, correct, "eval-1")
    assert all(score is None for score in record["scores"].values())
    assert record["errors"]["rag_search"] == "RuntimeError"
    assert not faithful.inputs and not correct.inputs


@pytest.mark.asyncio
async def test_empty_evidence_is_a_retrieval_miss_but_faithfulness_is_unscored():
    client, faithful, correct = FakeClient([]), FakeMetric(0.8), FakeMetric(0.7)
    record = await evaluation.evaluate_case(client, run_record(), faithful, correct, "eval-2")
    assert record["scores"] == {"recall_at_5": 0, "ndcg_at_5": 0, "faithfulness": None, "answer_correctness": 0.7}
    assert not faithful.inputs and record["errors"]["faithfulness"]
    assert set(record["metric_latency_ms"]) == {"answer_correctness"}


@pytest.mark.asyncio
async def test_trace_timeout_leaves_case_unscored(monkeypatch):
    async def timeout(*args):
        raise TimeoutError()

    monkeypatch.setattr(evaluation, "trace_state", timeout)
    client, faithful, correct = FakeClient([]), FakeMetric(0.8), FakeMetric(0.7)
    record = await evaluation.evaluate_case(client, run_record(), faithful, correct, "eval-3")
    assert record["scores"] == {"recall_at_5": None, "ndcg_at_5": None,
                                "faithfulness": None, "answer_correctness": None}
    assert "30 秒" in record["errors"]["trace_or_retrieval"]
    assert not faithful.inputs and not correct.inputs


@pytest.mark.asyncio
async def test_trace_timeout_reports_safe_error_type(monkeypatch):
    moments = iter([0, 0, 0, 0, 31, 31])
    monkeypatch.setattr(evaluation, "time", SimpleNamespace(monotonic=lambda: next(moments, 31)))

    def fail(**kwargs):
        raise RuntimeError("secret must not appear")

    client = SimpleNamespace(api=SimpleNamespace(observations=SimpleNamespace(get_many=fail)))
    with pytest.raises(TimeoutError) as caught:
        await evaluation.trace_state(client, "trace-id")
    assert "RuntimeError" in str(caught.value)
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_trace_state_retries_until_observations_are_available():
    client = FakeClient([{"filename": "A01_target.pdf", "content": "依据"}])
    original, calls = client.api.observations.get_many, []

    def delayed(**kwargs):
        calls.append(1)
        return SimpleNamespace(data=[], meta=SimpleNamespace(cursor=None)) if len(calls) == 1 else original(**kwargs)

    client.api.observations.get_many = delayed
    state, tokens = await evaluation.trace_state(client, "trace-1")
    assert len(calls) == 2 and state["evidence"][0]["content"] == "依据"
    assert tokens == {"input": 11, "output": 5}


@pytest.mark.asyncio
async def test_mismatched_trace_is_not_used_for_retrieval_or_tokens():
    client = FakeClient([{"filename": "A01_target.pdf", "content": "依据"}])
    original = client.api.observations.get_many

    def mismatch(**kwargs):
        response = original(**kwargs)
        graph = next(item for item in response.data if item.name == "rag-langgraph")
        graph.output = json.dumps({"answer": "另一个回答", "evidence": client.evidence})
        return response

    client.api.observations.get_many = mismatch
    record = await evaluation.evaluate_case(client, run_record(), FakeMetric(0.8), FakeMetric(0.7), "eval-mismatch")
    assert record["scores"] == {"recall_at_5": None, "ndcg_at_5": None,
                                "faithfulness": None, "answer_correctness": None}
    assert record["tokens"] is None and record["errors"]["trace_or_retrieval"] == "ValueError"


@pytest.mark.asyncio
async def test_langfuse_sdk_reads_trace_without_posting_scores():
    calls = []

    def respond(request):
        calls.append(request)
        assert request.method == "GET"
        common = {"traceId": run_record()["trace_id"], "projectId": "project-1", "startTime": "2026-01-01T00:00:00Z",
                  "endTime": "2026-01-01T00:00:01Z", "type": "SPAN"}
        data = [{**common, "id": "root", "name": "rag_search"},
                {**common, "id": "graph", "name": "rag-langgraph",
                 "output": json.dumps({"answer": "回答", "evidence": [{"filename": "A01_target.pdf", "content": "依据"}]})},
                {**common, "id": "llm", "name": "llm", "usageDetails": {"input": 11, "output": 5}}]
        return httpx.Response(200, json={"data": data, "meta": {"cursor": None}})

    transport = httpx.Client(transport=httpx.MockTransport(respond))
    client = Langfuse(public_key="pk-lf-test-sdk", secret_key="sk-lf-test-sdk",
                      base_url="http://test.invalid", httpx_client=transport, tracing_enabled=False)
    try:
        record = await evaluation.evaluate_case(client, run_record(), FakeMetric(0.8), FakeMetric(0.7), "eval-sdk")
        assert record["scores"] == {"recall_at_5": 1, "ndcg_at_5": 1, "faithfulness": 0.8, "answer_correctness": 0.7}
        assert record["tokens"] == {"input": 11, "output": 5} and not record["errors"]
        assert len(calls) == 1
        assert calls[0].url.path == "/api/public/v2/observations"
        assert calls[0].url.params["fields"] == "basic,io,usage"
    finally:
        client.shutdown()
        transport.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_model", ["gpt-6.1-terra", "gpt-6-luna"])
@pytest.mark.parametrize("effort,temperature", [("medium", None), ("none", 0), ("none", 0.5)])
async def test_real_ragas_adapter_uses_approved_chat_parameters_and_local_embeddings(judge_model, effort, temperature):
    requests = []
    split_ready, judge_ready = asyncio.Event(), asyncio.Event()
    started = {"split": 0, "judge": 0}

    async def respond(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        prompt = json.dumps(payload["messages"], ensure_ascii=False)
        if "Your task is to judge the faithfulness" in prompt:
            started["judge"] += 1
            if started["judge"] == 2:
                judge_ready.set()
            await asyncio.wait_for(judge_ready.wait(), 2)
            output = {"statements": [{"statement": "Supported.", "reason": "Supported.", "verdict": 1},
                                     {"statement": "Unsupported.", "reason": "Missing.", "verdict": 0}]}
        elif "Given a ground truth and an answer statements" in prompt:
            started["judge"] += 1
            if started["judge"] == 2:
                judge_ready.set()
            await asyncio.wait_for(judge_ready.wait(), 2)
            output = {"TP": [{"statement": "Matched.", "reason": "Supported."}],
                      "FP": [{"statement": "Extra.", "reason": "Unsupported."}],
                      "FN": [{"statement": "Missing.", "reason": "Omitted."}]}
        else:
            started["split"] += 1
            if started["split"] == 2:
                split_ready.set()
            await asyncio.wait_for(split_ready.wait(), 2)
            output = {"statements": ["The rate is 3%."]}
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "gpt-6.1-terra", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(output)}}]})

    class Models:
        def __init__(self):
            self.texts = []

        async def embed(self, texts):
            self.texts.extend(texts)
            return [[1.0, 0.0] for _ in texts]

    models = Models()
    evidence = [{"filename": "A01_target.pdf", "content": "The rate is 3%."}]

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        async with AsyncOpenAI(api_key="test-key", base_url="http://test.invalid/v1", http_client=transport) as api:
            llm = evaluation.create_judge(judge_model, api, {"reasoning_effort": effort, "temperature": temperature})
            faithfulness = evaluation.SharedFaithfulness(llm=llm)
            correctness = evaluation.SharedAnswerCorrectness(
                llm=llm, embeddings=evaluation.LocalEmbedding(models), faithfulness=faithfulness)
            client = FakeClient(evidence)
            record = await evaluation.evaluate_case(client, run_record(), faithfulness, correctness, "eval-shared")
    assert record["scores"]["faithfulness"] == 0.5
    assert record["scores"]["answer_correctness"] == 0.625
    assert len(requests) == 4 and started == {"split": 2, "judge": 2}
    assert len(models.texts) == 2
    assert all(path == "/v1/chat/completions" and payload["reasoning_effort"] == effort and payload["store"] is False
               and payload["max_completion_tokens"] == 4096 and payload["model"] == judge_model
               and payload["response_format"]["type"] == "json_object"
               and "tools" not in payload and "max_tokens" not in payload and "top_p" not in payload
               for path, payload in requests)
    assert all(("temperature" not in payload if temperature is None else payload["temperature"] == temperature)
               for _, payload in requests)


def test_ragas_statement_prompts_and_output_schemas_match():
    from ragas.metrics.collections.faithfulness.util import (
        StatementGeneratorInput as FaithInput, StatementGeneratorOutput as FaithOutput,
        StatementGeneratorPrompt as FaithPrompt,
    )
    from ragas.metrics.collections.answer_correctness.util import (
        StatementGeneratorInput as CorrectInput, StatementGeneratorOutput as CorrectOutput,
        StatementGeneratorPrompt as CorrectPrompt,
    )

    assert FaithPrompt().to_string(FaithInput(question="问题", answer="回答")) == (
        CorrectPrompt().to_string(CorrectInput(question="问题", answer="回答")))
    assert FaithOutput.model_json_schema() == CorrectOutput.model_json_schema()


@pytest.mark.asyncio
async def test_shared_split_failure_does_not_prevent_correctness(monkeypatch):
    async def fail(self, question, response):
        raise RuntimeError("split failed")

    def respond(request):
        payload = json.loads(request.content)
        prompt = json.dumps(payload["messages"], ensure_ascii=False)
        if "Given a ground truth and an answer statements" in prompt:
            output = {"TP": [{"statement": "匹配", "reason": "支持"}], "FP": [], "FN": []}
        else:
            output = {"statements": ["匹配"]}
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "gpt-6-luna", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(output)}}]})

    class Models:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(Faithfulness, "_create_statements", fail)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        async with AsyncOpenAI(api_key="test-key", base_url="http://test.invalid/v1", http_client=transport) as api:
            llm = evaluation.create_judge("gpt-6-luna", api, {"reasoning_effort": "none", "temperature": 0})
            faithfulness = evaluation.SharedFaithfulness(llm=llm)
            correctness = evaluation.SharedAnswerCorrectness(
                llm=llm, embeddings=evaluation.LocalEmbedding(Models()), faithfulness=faithfulness)
            client = FakeClient([{"filename": "A01_target.pdf", "content": "依据"}])
            record = await evaluation.evaluate_case(client, run_record(), faithfulness, correctness, "eval-fallback")
    assert record["scores"]["faithfulness"] is None
    assert record["errors"]["faithfulness"] == "RuntimeError"
    assert record["scores"]["answer_correctness"] == 1


@pytest.mark.asyncio
async def test_failed_correctness_does_not_discard_faithfulness():
    class FailedMetric:
        async def ascore(self, **kwargs):
            raise ValueError("metric failed")

    client = FakeClient([{"filename": "A01_target.pdf", "content": "依据"}])
    record = await evaluation.evaluate_case(client, run_record(), FakeMetric(0.8), FailedMetric(), "eval-partial")
    assert record["scores"]["faithfulness"] == 0.8
    assert record["scores"]["answer_correctness"] is None
    assert record["errors"]["answer_correctness"] == "ValueError"


def test_report_summary_ignores_unscored_values_and_is_saved_as_json(tmp_path):
    records = [{"scores": {name: 0.5 for name in evaluation.SCORE_NAMES}}]
    summary = evaluation.summarize(records, total=1)
    assert summary["recall_at_5"] == {"mean": 0.5, "scored": 1, "total": 1}
    unscored = [{"scores": {name: None for name in evaluation.SCORE_NAMES}}]
    assert evaluation.summarize(unscored, total=1)["recall_at_5"] == {"mean": None, "scored": 0, "total": 1}
    report = {"cases": records, "summary": summary}
    target = tmp_path / "result.json"
    evaluation.save_report(target, report)
    assert json.loads(target.read_text(encoding="utf-8")) == report


def test_report_latency_summary_uses_answered_cases_and_handles_missing_data():
    scores = {name: None for name in evaluation.SCORE_NAMES}
    records = [
        {"scores": scores, "status": "success", "answer": "回答", "latency_ms": 0},
        {"scores": scores, "status": "insufficient", "answer": "部分回答", "latency_ms": 100},
        {"scores": scores, "status": "success", "answer": "回答", "latency_ms": 200},
        {"scores": scores, "status": "error", "answer": None, "latency_ms": 9999},
    ]
    latency = evaluation.summarize(records, total=5)["mcp_latency_ms"]
    assert latency == {"mean": 100, "p50": 100, "p95": pytest.approx(190), "scored": 3, "total": 5}
    assert evaluation.summarize(records[:1], total=1)["mcp_latency_ms"] == {
        "mean": 0, "p50": 0, "p95": 0, "scored": 1, "total": 1}
    assert evaluation.summarize(records[-1:], total=1)["mcp_latency_ms"] == {
        "mean": None, "p50": None, "p95": None, "scored": 0, "total": 1}
