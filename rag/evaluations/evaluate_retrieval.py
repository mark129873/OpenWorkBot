import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langfuse import Langfuse

from app import prompts, rag_parameters as params, tracing
from app.config import settings
from app.graph import build_graph
from app.knowledge import Knowledge
from evaluations.utils.benchmark import create_output, document_number, error_name, load_benchmark, require_langfuse, retrieval_scores
from evaluations.utils.evaluation_utils import save_summary, summarize

SCORE_NAMES = ("candidate_document_recall", "rerank_document_recall", "recall_at_5", "ndcg_at_5")


def document_recall(chunks: list[dict], expected: list[str]) -> float:
    targets = {document_number(name) for name in expected}
    if not targets or None in targets:
        raise ValueError("目标文档编号无效")
    found = set()
    for item in chunks:
        filename = item.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("检索结果缺少文档名")
        found.add(document_number(filename))
    return len(found & targets) / len(targets)


async def evaluate_case(graph, case: dict, run_id: str) -> dict:
    record = {**case, "status": "error", "evidence": [], "candidates": [], "rerank_candidates": [],
              "scores": dict.fromkeys(SCORE_NAMES), "latency_ms": None, "error": None,
              "trace_id": None, "trace_error": None}
    started = time.perf_counter()
    span = None
    try:
        operation_id = f"bank-rag-retrieval-{run_id}-{case['id']}"
        with tracing.query(case["question"], operation_id) as span:
            if span is not None:
                record["trace_id"] = Langfuse.create_trace_id(seed=operation_id)
            elif tracing.enabled():
                record["trace_error"] = "LangfuseUnavailable"
            config = tracing.graph_config() or {}
            config["recursion_limit"] = max(25, (params.SUPPLEMENTAL_ROUNDS + 1) * 5 + 5)
            state = await graph.ainvoke(
                {"query": case["question"], "attempt": 0, "seen": [], "candidates": []},
                config=config, interrupt_before=["answer"],
            )
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            if state.get("graded") is not True or any(not isinstance(state.get(name), list)
                                                      for name in ("candidates", "rerank_candidates", "evidence")):
                raise ValueError("检索图未返回完整证据")
            recall, ndcg = retrieval_scores(state["evidence"], case["targets"])
            record.update({key: state.get(key) for key in ("candidates", "rerank_candidates", "evidence",
                          "seen", "sufficient", "can_retry", "missing", "attempt", "warning")})
            record["scores"] = {"candidate_document_recall": document_recall(state["candidates"], case["targets"]),
                                "rerank_document_recall": document_recall(state["rerank_candidates"], case["targets"]),
                                "recall_at_5": recall, "ndcg_at_5": ndcg}
            record["status"] = "success"
            if span is not None:
                try:
                    span.update(output={key: record[key] for key in ("candidates", "rerank_candidates", "evidence",
                                                                     "seen", "sufficient", "can_retry", "missing",
                                                                     "attempt", "warning")})
                except Exception as exc:
                    record["trace_error"] = error_name(exc)
    except Exception as exc:
        record["error"] = error_name(exc)
    finally:
        if record["latency_ms"] is None:
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if span is not None:
            try:
                # shortcut: SDK 发送不确认入库，需要可靠回执时回读追踪。
                await asyncio.to_thread(tracing.configured_client().flush)
            except Exception as exc:
                record["trace_error"] = error_name(exc)
        if record["trace_error"]:
            print(f"{case['id']} Langfuse 追踪未完成：{record['trace_error']}", file=sys.stderr, flush=True)
    for key in ("candidates", "rerank_candidates", "seen", "sufficient", "can_retry", "missing", "attempt"):
        record.pop(key, None)
    return record


async def main(dataset: Path, limit: int = 1) -> Path:
    cases = load_benchmark(dataset, limit)
    config = settings()
    if (config.provider not in {"openai", "anthropic"} or not config.api_key or not config.model
            or not config.embedding_model or not config.rerank_model or not config.milvus_uri or not config.milvus_token):
        raise SystemExit("请在 rag/.env 中配置模型、Milvus、Embedding 和重排")
    if not (config.data_dir / "catalog.sqlite3").is_file():
        raise SystemExit("找不到知识库目录，请检查 RAG_DATA_DIR")
    knowledge = Knowledge(config)
    versions = sorted(knowledge.catalog.versions())
    if not versions:
        raise SystemExit("知识库没有已就绪文档")
    if tracing.enabled():
        require_langfuse()
        try:
            valid = await asyncio.to_thread(tracing.configured_client().auth_check)
        except Exception as exc:
            raise SystemExit(f"Langfuse 连接失败：{error_name(exc)}") from None
        if not valid:
            raise SystemExit("Langfuse 认证失败")
    graph = build_graph(knowledge)
    directory = ROOT / "data" / "evaluations" / "bank-rag" / "retrieval"
    run_id, path, output = create_output(directory, datetime.now().strftime("%m-%d-%H-%M"), ".json")
    metadata = {"run_id": run_id, "langfuse_enabled": tracing.enabled(), "dataset": dataset.name,
                "temperature": config.temperature,
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "case_ids": [case["id"] for case in cases],
                "index_versions_sha256": hashlib.sha256("\n".join(versions).encode("utf-8")).hexdigest(),
                "model": config.model, "provider": config.provider, "collection": config.collection,
                "rag_reasoning_effort": config.reasoning_effort if config.provider == "openai" else None,
                "embedding_model": Path(config.embedding_model).name, "rerank_model": Path(config.rerank_model).name,
                "prompt_file_sha256": hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest(),
                "parameters": {name.lower(): getattr(params, name) for name in (
                    "REWRITE_QUERY_LIMIT", "QUERIES_PER_ROUND", "DENSE_TOP_K", "BM25_TOP_K", "FUSED_TOP_K",
                    "RRF_K", "RERANK_CANDIDATES", "EVIDENCE_TOP_K", "SUPPLEMENTAL_ROUNDS",
                    "SEARCH_RETRIES", "JSON_RETRIES", "MODEL_REQUEST_RETRIES", "MAX_OUTPUT_TOKENS")},
                "latency_scope": "检索规划至最终证据，不含答案生成和评分",
                "score_scope": "目标文件编号；最终指标按去重后的前5份文档"}
    records = []
    with output:
        json.dump({**metadata, "summary": summarize(records, SCORE_NAMES, len(cases)), "cases": records},
                  output, ensure_ascii=False, indent=2, allow_nan=False)
    for case in cases:
        record = await evaluate_case(graph, case, run_id)
        records.append(record)
        save_summary(path, {**metadata, "summary": summarize(records, SCORE_NAMES, len(cases)), "cases": records})
        print(f"{case['id']} 检索：{record['status']}；异常：{record['error']}", flush=True)
    print(f"检索结果：{path}", flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立评测银行 RAG 检索并保存固定证据")
    parser.add_argument("--dataset", type=Path, required=True, help="题集 JSONL 路径")
    parser.add_argument("--limit", type=int, default=1, help="题目数量，默认 1")
    arguments = parser.parse_args()
    if arguments.limit < 1:
        parser.error("--limit 必须大于零")
    try:
        asyncio.run(main(arguments.dataset, arguments.limit))
    except Exception as exc:
        print(f"检索评测未完成：{error_name(exc)}", file=sys.stderr)
        raise SystemExit(1)
