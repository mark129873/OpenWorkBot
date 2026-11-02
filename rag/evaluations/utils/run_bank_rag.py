import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langfuse import Langfuse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app import rag_parameters as params, tracing
from app.config import settings
from evaluations.utils.benchmark import call_tool, create_output, error_name, load_benchmark, require_langfuse

async def run_case(session: ClientSession, case: dict, run_id: str) -> dict:
    operation_id = f"bank-rag-{run_id}-{case['id']}"
    record = {**case, "trace_id": Langfuse.create_trace_id(seed=operation_id),
              "status": "error", "answer": None, "latency_ms": None, "error": None}
    start = time.perf_counter()
    try:
        response = await call_tool(session, "rag_search", {"query": case["question"], "operation_id": operation_id})
        record["status"] = response.get("status", "unknown")
        record["answer"] = response.get("answer")
        if (not isinstance(record["status"], str) or not record["status"]
                or not isinstance(record["answer"], str) or not record["answer"].strip()):
            raise ValueError("MCP 未返回有效回答")
    except Exception as exc:
        record["status"] = "error"
        record["error"] = error_name(exc)
    record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return record


async def main(dataset: Path, limit: int = 1) -> Path:
    require_langfuse()
    config = settings()
    if (config.provider != "openai" or config.model != "gpt-6-luna" or not config.api_key
            or not config.embedding_model or not config.rerank_model or not config.milvus_uri or not config.milvus_token):
        raise SystemExit("请在 rag/.env 中配置 OpenAI、Milvus、Embedding 和重排模型")
    cases = load_benchmark(dataset, limit)
    output_dir = ROOT / "data" / "evaluations" / "bank-rag"
    parameters = {"dataset": dataset.name, "cases": [case["id"] for case in cases], "temperature": config.temperature,
                  "rag_reasoning_effort": config.reasoning_effort,
                  "embedding_model": Path(config.embedding_model).name,
                  "rerank_model": Path(config.rerank_model).name, "collection": config.collection,
                  "prompt_version": tracing.prompt_version(),
                  "rag": {name.lower(): getattr(params, name) for name in (
                      "REWRITE_QUERY_LIMIT", "QUERIES_PER_ROUND", "DENSE_TOP_K", "BM25_TOP_K",
                      "FUSED_TOP_K", "RRF_K", "RERANK_CANDIDATES", "EVIDENCE_TOP_K",
                      "SUPPLEMENTAL_ROUNDS", "SEARCH_RETRIES", "JSON_RETRIES",
                      "MODEL_REQUEST_RETRIES", "MAX_OUTPUT_TOKENS")}}
    langfuse = Langfuse()
    try:
        if not await asyncio.to_thread(langfuse.auth_check):
            raise SystemExit("Langfuse 配置无效")
        server_params = StdioServerParameters(command=sys.executable, args=["-m", "app.server"], cwd=ROOT,
                                              env={**os.environ, "OPENWORKBOT_BENCHMARK_FLUSH": "1"})
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                run_id, run_path, file = create_output(output_dir / "runs", datetime.now().strftime("%m-%d-%H-%M"),
                                                       ".jsonl", conflicts=output_dir)
                with file:
                    print(f"RAG 批次：{run_path}", flush=True)
                    try:
                        for case in cases:
                            print(f"正在运行 {case['id']}", flush=True)
                            record = await run_case(session, case, run_id)
                            record.update(run_id=run_id, model=config.model, parameters=parameters)
                            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                            file.flush()
                            print(f"{case['id']} 状态：{record['status']}；异常：{record['error']}", flush=True)
                    finally:
                        try:
                            result = await call_tool(session, "rag_flush_tracing", {})
                            if result.get("status") != "success":
                                raise RuntimeError("追踪发送未完成")
                        except Exception:
                            print(f"Langfuse 追踪发送未完成，已保存批次：{run_path}", file=sys.stderr, flush=True)
                            raise
        print(f"RAG 批次：{run_path}", flush=True)
        return run_path
    finally:
        langfuse.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行银行 RAG 数据集")
    parser.add_argument("--dataset", type=Path, required=True, help="题集 JSONL 路径")
    parser.add_argument("--limit", type=int, default=1, help="运行题目数量，默认 1")
    arguments = parser.parse_args()
    if arguments.limit < 1:
        parser.error("--limit 必须大于零")
    try:
        asyncio.run(main(arguments.dataset, arguments.limit))
    except Exception as exc:
        print(f"RAG 运行未完成：{error_name(exc)}", file=sys.stderr)
        raise SystemExit(1)
