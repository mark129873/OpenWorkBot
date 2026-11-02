import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from langfuse import Langfuse
from openai import AsyncOpenAI
from ragas.metrics.collections import AnswerCorrectness, Faithfulness

from app import rag_parameters as params
from app.config import settings
from app.graph import answer_body
from app.models import Models
from evaluations.utils.benchmark import create_output, error_name, load_benchmark, require_langfuse, retrieval_scores
from evaluations.utils.evaluation_utils import create_judge, judge_settings

SCORE_NAMES = ("recall_at_5", "ndcg_at_5", "faithfulness", "answer_correctness")
TIME_ID = re.compile(r"[0-9]{2}-[0-9]{2}-[0-9]{2}-[0-9]{2}(?:-[0-9]{2,})?")
LEGACY_ID = re.compile(r"[0-9a-f]{32}")


def load_run(path: Path) -> list[dict]:
    cases = load_benchmark(path, None)
    run_id, model, parameters = (cases[0].get(key) for key in ("run_id", "model", "parameters"))
    valid_id = isinstance(run_id, str) and (TIME_ID.fullmatch(run_id) or LEGACY_ID.fullmatch(run_id))
    if (not valid_id or not isinstance(model, str) or not model or not isinstance(parameters, dict)
            or not isinstance(parameters.get("dataset"), str)
            or not isinstance(parameters.get("cases"), list) or not parameters["cases"]
            or any(not isinstance(item, str) or not item for item in parameters["cases"])
            or len(set(parameters["cases"])) != len(parameters["cases"])):
        raise ValueError("RAG 批次信息无效")
    for case in cases:
        latency = case.get("latency_ms")
        if (case.get("run_id") != run_id or case.get("model") != model or case.get("parameters") != parameters
                or case["id"] not in parameters["cases"]
                or case.get("trace_id") != Langfuse.create_trace_id(seed=f"bank-rag-{run_id}-{case['id']}")
                or not isinstance(case.get("status"), str) or not case["status"]
                or not isinstance(latency, (int, float)) or isinstance(latency, bool)
                or not math.isfinite(latency) or latency < 0
                or (case["status"] != "error" and (not isinstance(case.get("answer"), str) or not case["answer"].strip()))):
            raise ValueError(f"RAG 批次题目 {case['id']} 无效")
    return cases


async def trace_state(client: Langfuse, trace_id: str) -> tuple[dict, dict | None]:
    deadline = time.monotonic() + 30
    last_error = None
    while time.monotonic() < deadline:
        try:
            observations, cursor, seen, complete = [], None, set(), False
            while time.monotonic() < deadline:
                remaining = max(1, int(deadline - time.monotonic()))
                response = await asyncio.to_thread(
                    client.api.observations.get_many, trace_id=trace_id, fields="basic,io,usage", limit=1000,
                    cursor=cursor, request_options={"timeout_in_seconds": min(5, remaining), "max_retries": 0})
                observations.extend(response.data)
                cursor = response.meta.cursor
                if not cursor or not response.data:
                    complete = True
                    break
                if cursor in seen:
                    raise ValueError("trace 分页游标重复")
                seen.add(cursor)
            if not complete:
                continue
            root = next((item for item in observations if item.name == "rag_search" and item.end_time), None)
            graph = next((item for item in observations if item.name == "rag-langgraph" and item.end_time), None)
            if root and graph and graph.output is not None:
                state = json.loads(graph.output) if isinstance(graph.output, str) else graph.output
                if (not isinstance(state, dict) or not isinstance(state.get("evidence"), list)
                        or not all(isinstance(item, dict) for item in state["evidence"])):
                    raise ValueError("trace 缺少最终证据")
                calls = [item for item in observations if item.name == "llm"]
                usage = None
                if calls and all(isinstance(item.usage_details, dict)
                                 and isinstance(item.usage_details.get("input"), int)
                                 and isinstance(item.usage_details.get("output"), int) for item in calls):
                    usage = {key: sum(item.usage_details[key] for item in calls) for key in ("input", "output")}
                return state, usage
        except ValueError:
            raise
        except Exception as exc:
            last_error = error_name(exc)
        await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
    detail = f"；最近错误：{last_error}" if last_error else ""
    raise TimeoutError(f"30 秒内未获取完整 trace{detail}")


class LocalEmbedding:
    def __init__(self, models: Models):
        self.models = models

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return (await self.models.embed([text]))[0]


# shortcut: 依赖 Ragas 0.4.3 的私有拆分方法，升级时需重新验证。
class SharedFaithfulness(Faithfulness):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._statement_task = None
        self._statement_input = None

    async def _create_statements(self, question: str, response: str) -> list[str]:
        key = (question, response)
        if self._statement_task is None:
            self._statement_input = key
            self._statement_task = asyncio.create_task(super()._create_statements(question, response))
        if self._statement_input == key:
            return await self._statement_task
        return await super()._create_statements(question, response)


class SharedAnswerCorrectness(AnswerCorrectness):
    def __init__(self, *, faithfulness: SharedFaithfulness, **kwargs):
        super().__init__(**kwargs)
        self._faithfulness = faithfulness
        self._response = None
        self._reference = None
        self._reference_task = None

    async def ascore(self, user_input: str, response: str, reference: str):
        self._response, self._reference = response, reference
        if response != reference:
            self._reference_task = asyncio.create_task(super()._generate_statements(user_input, reference))
        try:
            return await super().ascore(user_input, response, reference)
        finally:
            if self._reference_task is not None:
                if not self._reference_task.done():
                    self._reference_task.cancel()
                await asyncio.gather(self._reference_task, return_exceptions=True)

    async def _generate_statements(self, question: str, text: str) -> list[str]:
        if text == self._response:
            try:
                return await self._faithfulness._create_statements(question, text)
            except Exception:
                return await super()._generate_statements(question, text)
        if text == self._reference and self._reference_task is not None:
            return await self._reference_task
        return await super()._generate_statements(question, text)


async def evaluate_case(client: Langfuse, case: dict, faithfulness, correctness, evaluation_id: str,
                        batch_id: str | None = None) -> dict:
    record = {"id": case["id"], "question": case["question"], "reference": case["reference"],
              "expected_behavior": case["behavior"], "expected_documents": case["targets"],
              "run_id": case["run_id"], "batch_id": batch_id or case["run_id"],
              "evaluation_id": evaluation_id, "trace_id": case["trace_id"],
              "answer": answer_body(case["answer"], [], case["question"]) if isinstance(case["answer"], str) else None,
              "retrieved_documents": None,
              "status": case["status"], "latency_ms": case["latency_ms"], "tokens": None,
              "scoring_latency_ms": None, "metric_latency_ms": {},
              "scores": {name: None for name in SCORE_NAMES},
              "errors": {}}
    if record["status"] == "error":
        record["errors"]["rag_search"] = case.get("error") or "RAG 运行失败"
        return record

    try:
        state, tokens = await trace_state(client, record["trace_id"])
        if state.get("answer") != case["answer"]:
            raise ValueError("trace 与 MCP 回答不一致")
        record["tokens"] = tokens
        evidence = state["evidence"]
        record["answer"] = answer_body(state["answer"], evidence, case["question"])
        if not record["answer"]:
            record["status"] = "error"
            record["errors"]["answer"] = "清除引用后回答为空"
            return record
    except Exception as exc:
        record["errors"]["trace_or_retrieval"] = (str(exc) or "30 秒内未获取完整 trace") if isinstance(exc, TimeoutError) else error_name(exc)
        return record

    try:
        record["retrieved_documents"] = [item.get("filename") for item in evidence]
        recall, ndcg = retrieval_scores(evidence, case["targets"])
        record["scores"].update(recall_at_5=recall, ndcg_at_5=ndcg)
        if record["tokens"] is None:
            record["errors"]["tokens"] = "未取得全部模型 token 用量"
    except Exception as exc:
        record["errors"]["trace_or_retrieval"] = error_name(exc)

    metrics = [("answer_correctness", correctness, {"user_input": case["question"], "response": record["answer"],
                                                    "reference": case["reference"]})]
    if evidence and all(isinstance(item.get("content"), str) and item["content"].strip() for item in evidence):
        metrics.insert(0, ("faithfulness", faithfulness, {"user_input": case["question"], "response": record["answer"],
                                                         "retrieved_contexts": [item["content"] for item in evidence]}))
    else:
        record["errors"]["faithfulness"] = "缺少有效证据"
    async def score(name, metric, inputs):
        started = time.perf_counter()
        try:
            value = float((await metric.ascore(**inputs)).value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("评分结果无效")
            record["scores"][name] = value
        except Exception as exc:
            record["errors"][name] = error_name(exc)
        finally:
            record["metric_latency_ms"][name] = round((time.perf_counter() - started) * 1000, 2)
    started = time.perf_counter()
    await asyncio.gather(*(score(name, metric, inputs) for name, metric, inputs in metrics))
    record["scoring_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return record


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def summarize(records: list[dict], total: int) -> dict:
    summary = {}
    for name in SCORE_NAMES:
        values = [record["scores"][name] for record in records if record["scores"][name] is not None]
        summary[name] = {"mean": sum(values) / len(values) if values else None,
                         "scored": len(values), "total": total}
    latencies = sorted(record["latency_ms"] for record in records
                       if record.get("status") != "error" and isinstance(record.get("answer"), str)
                       and record["answer"].strip() and record.get("latency_ms") is not None)
    p95 = None
    if latencies:
        rank = (len(latencies) - 1) * 0.95
        lower = int(rank)
        p95 = latencies[lower] + (latencies[math.ceil(rank)] - latencies[lower]) * (rank - lower)
    summary["mcp_latency_ms"] = {"mean": sum(latencies) / len(latencies) if latencies else None,
                                 "p50": statistics.median(latencies) if latencies else None, "p95": p95,
                                 "scored": len(latencies), "total": total}
    return summary


async def main(run_path: Path | None = None) -> Path:
    require_langfuse()
    config = settings()
    judge_model = os.getenv("EVAL_MODEL_ID", "").strip()
    judge_options = judge_settings()
    if config.provider != "openai" or not config.api_key or not config.embedding_model or not judge_model:
        raise SystemExit("请在 rag/.env 中配置 OpenAI、Embedding 和 EVAL_MODEL_ID")
    latest = run_path is None
    if latest:
        paths = list((ROOT / "data" / "evaluations" / "bank-rag" / "runs").glob("*.jsonl"))
        if not paths:
            raise SystemExit("未找到 RAG 批次，请先运行 rag/evaluations/utils/run_bank_rag.py")
        run_path = max(paths, key=lambda path: path.stat().st_mtime_ns)
    cases = load_run(run_path)
    run_id, parameters = cases[0]["run_id"], cases[0]["parameters"]
    if latest and len(cases) != len(parameters["cases"]):
        raise SystemExit("最新 RAG 批次未完成；如需评测已有题目，请使用 --run 指定批次文件")
    batch_id = run_path.stem if LEGACY_ID.fullmatch(run_id) and TIME_ID.fullmatch(run_path.stem) else run_id
    output_dir = ROOT / "data" / "evaluations" / "bank-rag"
    completed = {case["id"] for case in cases}
    report = {"run_id": run_id, "batch_id": batch_id, "model": cases[0]["model"],
              "parameters": {**parameters, "top_documents": 5, "trace_wait_seconds": 30, "langfuse_enabled": True,
                             "judge_model": judge_model, "judge_temperature": judge_options["temperature"],
                             "reasoning_effort": judge_options["reasoning_effort"],
                             "max_completion_tokens": 4096, "judge_request_retries": params.MODEL_REQUEST_RETRIES, "store": False,
                             "answer_correctness_weights": [0.75, 0.25], "token_scope": "rag_llm_only",
                             "answer_format": "citation_free_v1"},
              "missing_cases": [case_id for case_id in parameters["cases"] if case_id not in completed], "cases": []}
    langfuse = Langfuse()
    try:
        if not await asyncio.to_thread(langfuse.auth_check):
            raise SystemExit("Langfuse 配置无效")
        async with AsyncOpenAI(api_key=config.api_key, base_url=config.base_url or "https://api.openai.com/v1",
                               timeout=120.0, max_retries=params.MODEL_REQUEST_RETRIES) as api:
            judge = create_judge(judge_model, api, judge_options)
            embeddings = LocalEmbedding(Models(config))
            evaluation_id, report_path, report_file = create_output(output_dir, batch_id, ".json")
            report["evaluation_id"] = evaluation_id
            with report_file:
                json.dump(report, report_file, ensure_ascii=False, indent=2, allow_nan=False)
            for case in cases:
                print(f"正在评测 {case['id']}", flush=True)
                faithfulness = SharedFaithfulness(llm=judge)
                correctness = SharedAnswerCorrectness(llm=judge, embeddings=embeddings, faithfulness=faithfulness)
                record = await evaluate_case(langfuse, case, faithfulness, correctness, evaluation_id, batch_id)
                report["cases"].append(record)
                report["summary"] = summarize(report["cases"], len(parameters["cases"]))
                save_report(report_path, report)
                print(f"{case['id']} 评分：{record['scores']}；异常：{record['errors']}", flush=True)
        print(f"评测结果：{report_path}", flush=True)
        return report_path
    finally:
        langfuse.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Langfuse 读取银行 RAG 批次并评分")
    parser.add_argument("--run", type=Path, help="RAG 批次 JSONL 文件，默认使用最新批次")
    arguments = parser.parse_args()
    try:
        asyncio.run(main(arguments.run))
    except Exception as exc:
        print(f"评测未完成：{error_name(exc)}", file=sys.stderr)
        raise SystemExit(1)
