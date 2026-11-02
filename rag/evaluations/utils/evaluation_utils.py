import json
import math
import os
import statistics
from pathlib import Path


def judge_settings() -> dict:
    value = os.getenv("EVAL_TEMPERATURE", "none").strip().lower()
    try:
        temperature = None if value in {"", "none"} else float(value)
    except ValueError:
        raise ValueError("EVAL_TEMPERATURE 必须为数字或 none") from None
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("EVAL_TEMPERATURE 必须为 0 到 2 之间的数字或 none")
    return {"temperature": temperature,
            "reasoning_effort": (os.getenv("EVAL_REASONING_EFFORT") or "medium").strip().lower() or "medium"}


def create_judge(model: str, client, settings: dict):
    from ragas.llms import llm_factory

    judge = llm_factory(model, client=client, max_tokens=4096,
                        reasoning_effort=settings["reasoning_effort"], store=False)
    judge.model_args["max_completion_tokens"] = judge.model_args.pop("max_tokens")
    judge.model_args.pop("top_p", None)
    if settings["temperature"] is None:
        judge.model_args.pop("temperature", None)
    else:
        judge.model_args["temperature"] = settings["temperature"]
    # shortcut: RAGAS 0.4.3 会改写温度，升级后复核参数映射。
    judge._map_openai_params = lambda: judge.model_args.copy()
    return judge


def summarize(records: list[dict], score_names: tuple[str, ...], total: int) -> dict:
    summary = {"total": total, "completed": len(records),
               "failed": sum(item.get("status") == "error" for item in records)}
    summary["scores"] = {}
    for name in score_names:
        values = [item["scores"][name] for item in records if item["scores"].get(name) is not None]
        summary["scores"][name] = {"mean": sum(values) / len(values) if values else None,
                                   "scored": len(values), "total": total}
    for field in ("latency_ms", "scoring_latency_ms"):
        if not any(field in item for item in records):
            continue
        values = sorted(item[field] for item in records
                        if item.get("status") != "error" and item.get(field) is not None)
        p95 = None
        if values:
            rank = (len(values) - 1) * 0.95
            lower = math.floor(rank)
            p95 = values[lower] + (values[math.ceil(rank)] - values[lower]) * (rank - lower)
        summary[field] = {"mean": sum(values) / len(values) if values else None,
                          "p50": statistics.median(values) if values else None, "p95": p95,
                          "scored": len(values), "total": total}
    return summary


def save_summary(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
