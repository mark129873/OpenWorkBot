import json
import logging
import threading
import time
from pathlib import Path

from app.config import settings
from app.user_context import require_user

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()


def record_llm_call(
    session_id: str,
    turn_id: str,
    model: str,
    system: str,
    messages: list,
    response_text: str,
    tool_calls: list[dict],
    usage: dict,
    duration_ms: float,
    tool_names: list[str] | None = None,
    subtask_id: str | None = None,
    error: str | None = None,
    stop_reason: str | None = None,
) -> None:
    try:
        record = {
            "time": time.time() * 1000.0,
            "session_id": session_id,
            "turn_id": turn_id,
            "subtask_id": subtask_id,
            "model": model,
            "duration_ms": round(duration_ms, 1),
            "error": error,
            "request": {
                "system": system,
                "messages": messages,
                "tool_names": tool_names or [],
            },
            "response": {
                "text": response_text,
                "tool_calls": tool_calls,
                "stop_reason": stop_reason,
            },
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_tokens", 0),
            },
        }
        target = Path(settings.modelio_dir) / require_user()
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"model-io-{session_id}.jsonl"
        max_bytes = max(1, int(settings.modelio_max_bytes))
        with _write_lock:
            if path.exists() and path.stat().st_size > max_bytes:
                path.rename(target / f"model-io-{session_id}-{time.time_ns()}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # 观测失败不得阻断对话。
        logger.warning("model-io 写入失败（忽略）：%s", exc)


def clear_user_logs() -> None:
    root = Path(settings.modelio_dir).resolve()
    target = root / require_user()
    if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
        raise ValueError("日志目录不能是链接")
    if target.resolve().parent != root:
        raise ValueError("日志目录无效")
    with _write_lock:
        if target.exists():
            for path in target.glob("model-io-*.jsonl"):
                if path.is_symlink() or path.parent.resolve() != target.resolve():
                    raise ValueError("日志文件无效")
                path.unlink()
