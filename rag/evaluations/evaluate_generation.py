import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from openai import AsyncOpenAI
from ragas.metrics.collections import AnswerCorrectness, Faithfulness

from app import prompts, rag_parameters as params
from app.config import settings
from app.graph import answer_evidence, grade_evidence
from app.models import Models
from evaluations.utils.benchmark import create_output, error_name
from evaluations.utils.evaluation_utils import create_judge, judge_settings, save_summary, summarize

SCORE_NAMES = ("faithfulness", "answer_correctness")


def load_evidence(content: str, limit: int) -> list[dict]:
    if limit < 1:
        raise ValueError("评测题目数量必须大于零")
    try:
        data = json.loads(content)
        json.dumps(data, allow_nan=False)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("固定证据 JSON 无效") from exc
    data = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(data, list) or not data:
        raise ValueError("固定证据题目为空或格式无效")
    seen = set()
    for number, case in enumerate(data, 1):
        if (not isinstance(case, dict)
                or any(not isinstance(case.get(key), str) or not case[key].strip()
                       for key in ("id", "question", "reference"))
                or case["id"] != case["id"].strip()
                or not isinstance(case.get("evidence"), list)
                or any(not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key].strip()
                                                        for key in ("filename", "content"))
                       for item in case["evidence"])
                or not isinstance(case.get("warning", ""), str)):
            raise ValueError(f"固定证据第 {number} 题内容无效")
        if case["id"] in seen:
            raise ValueError(f"固定证据第 {number} 题编号重复")
        seen.add(case["id"])
    return data[:limit]


class ScoreEmbedding:
    def __init__(self, models: Models):
        self.models = models

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return (await self.models.embed([text]))[0]


async def evaluate_case(case: dict, models, faithfulness, correctness) -> dict:
    record = {key: case[key] for key in ("id", "question", "reference", "evidence")}
    record.update(expected_behavior=case.get("behavior"), status="error", answer=None, sufficient=None,
                  scores=dict.fromkeys(SCORE_NAMES), latency_ms=None, scoring_latency_ms=None,
                  error=None, metric_errors={})
    if case.get("status") == "error":
        record["error"] = "输入证据来自失败题"
        return record
    started = time.perf_counter()
    try:
        state = {"query": case["question"], "evidence": case["evidence"], "warning": case.get("warning", "")}
        state.update(await grade_evidence(models, state))
        answer = await answer_evidence(models, state)
        if not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
            raise ValueError("生成回答为空")
        record.update(status=answer["status"], answer=answer["answer"], sufficient=state["sufficient"])
    except Exception as exc:
        record["error"] = error_name(exc)
    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    if record["status"] == "error":
        return record

    metrics = [("answer_correctness", correctness, {"user_input": case["question"],
                                                   "response": record["answer"], "reference": case["reference"]})]
    if case["evidence"]:
        metrics.append(("faithfulness", faithfulness, {"user_input": case["question"], "response": record["answer"],
                        "retrieved_contexts": [item["content"] for item in case["evidence"]]}))
    else:
        record["metric_errors"]["faithfulness"] = "缺少有效证据"

    async def score(name, metric, inputs):
        try:
            value = float((await metric.ascore(**inputs)).value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("评分结果无效")
            record["scores"][name] = value
        except Exception as exc:
            record["metric_errors"][name] = error_name(exc)

    started = time.perf_counter()
    await asyncio.gather(*(score(name, metric, inputs) for name, metric, inputs in metrics))
    record["scoring_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return record


async def main(evidence_path: Path, limit: int = 1) -> Path:
    raw = evidence_path.read_bytes()
    cases = load_evidence(raw.decode("utf-8"), limit)
    config = settings()
    judge_model = os.getenv("EVAL_MODEL_ID", "").strip()
    judge_key = os.getenv("OPENAI_API_KEY", "").strip()
    judge_options = judge_settings()
    if (config.provider not in {"openai", "anthropic"} or not config.api_key or not config.model
            or not config.embedding_model or not judge_model or not judge_key):
        raise SystemExit("请在 rag/.env 中配置生成模型、评分 Embedding、OPENAI_API_KEY 和 EVAL_MODEL_ID")
    models = Models(config)
    directory = ROOT / "evaluations" / "result"
    run_id, path, output = create_output(directory, datetime.now().strftime("generation_%m-%d-%H-%M"), ".json")
    run_id = run_id.removeprefix("generation_")
    metadata = {"run_id": run_id, "evidence_file": evidence_path.name,
                "evidence_sha256": hashlib.sha256(raw).hexdigest(), "model": config.model,
                "case_ids": [case["id"] for case in cases], "ragas_version": version("ragas"),
                "provider": config.provider, "temperature": config.temperature,
                "rag_reasoning_effort": config.reasoning_effort if config.provider == "openai" else None,
                "prompt_file_sha256": hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest(),
                "judge_model": judge_model, "judge_temperature": judge_options["temperature"],
                "judge_reasoning_effort": judge_options["reasoning_effort"],
                "judge_max_completion_tokens": 4096, "judge_request_retries": params.MODEL_REQUEST_RETRIES, "store": False,
                "score_embedding_model": Path(config.embedding_model).name,
                "answer_correctness_weights": [0.75, 0.25],
                "answer_format": "citation_free_v1",
                "max_output_tokens": params.MAX_OUTPUT_TOKENS, "json_retries": params.JSON_RETRIES,
                "model_request_retries": params.MODEL_REQUEST_RETRIES,
                "latency_scope": "固定证据的判断和回答；评分耗时单独统计"}
    records = []
    with output:
        json.dump({**metadata, "summary": summarize(records, SCORE_NAMES, len(cases)), "cases": records},
                  output, ensure_ascii=False, indent=2, allow_nan=False)
    async with AsyncOpenAI(api_key=judge_key, base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
                           timeout=120.0, max_retries=params.MODEL_REQUEST_RETRIES) as api:
        judge = create_judge(judge_model, api, judge_options)
        faithfulness = Faithfulness(llm=judge)
        correctness = AnswerCorrectness(llm=judge, embeddings=ScoreEmbedding(models))
        for case in cases:
            record = await evaluate_case(case, models, faithfulness, correctness)
            records.append(record)
            save_summary(path, {**metadata, "summary": summarize(records, SCORE_NAMES, len(cases)), "cases": records})
            print(f"{case['id']} 生成：{record['status']}；异常：{record['error']}；评分异常：{record['metric_errors']}",
                  flush=True)
    print(f"生成结果：{path}", flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立评测固定证据下的生成")
    parser.add_argument("--evidence", type=Path, required=True, help="固定证据 JSON 路径")
    parser.add_argument("--limit", type=int, default=1, help="题目数量，默认 1")
    arguments = parser.parse_args()
    if arguments.limit < 1:
        parser.error("--limit 必须大于零")
    try:
        asyncio.run(main(arguments.evidence, arguments.limit))
    except Exception as exc:
        print(f"生成评测未完成：{error_name(exc)}", file=sys.stderr)
        raise SystemExit(1)
