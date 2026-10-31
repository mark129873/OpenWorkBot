import asyncio
import logging
from copy import deepcopy
from typing import AsyncIterator

from sqlalchemy import func, select

from app.models import Message, Part
from app.sessions import store

logger = logging.getLogger(__name__)

_MAX_COMPACT_FAILURES = 3
_MICRO_PLACEHOLDER = "[旧工具结果已清除以节省上下文]"

_failure_counts: dict[str, int] = {}


def context_budget(settings) -> int:
    return settings.context_window - settings.compact_output_reserve - settings.compact_buffer


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_history_tokens(history: list) -> int:
    total = 0
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content) + 4
        elif isinstance(content, list):
            for block in content:
                total += estimate_tokens(json_dumps_safe(block)) + 4
    return total


def json_dumps_safe(obj) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def needs_compact(settings, history: list, last_input_tokens: int | None = None,
                  measured_history_tokens: int | None = None) -> bool:
    """真实用量加新增历史估算，避免旧用量掩盖大输出。"""
    estimate = estimate_history_tokens(history)
    if last_input_tokens is not None and last_input_tokens > 0:
        growth = max(0, estimate - measured_history_tokens) if measured_history_tokens is not None else 0
        estimate = max(estimate, last_input_tokens + growth)
    return estimate > context_budget(settings)


def is_context_limit_error(exc: Exception) -> bool:
    """仅输入上下文超限可重试，其他错误不重试。"""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") or error.get("type") if isinstance(error, dict) else None
    if code in {"context_length_exceeded", "prompt_too_long"}:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in (
        "prompt is too long", "prompt too long", "prompt_too_long",
        "context_length_exceeded", "maximum context length", "input is too long",
    ))


def microcompact_parts(db, session_id: str, keep: int, *, before_sequence: int | None = None) -> int:
    """替换旧工具输出，保留顺序与最近完整结果。"""
    query = select(Part).where(
        Part.session_id == session_id, func.json_extract(Part.data, "$.type") == "tool")
    if before_sequence is not None:
        query = query.join(Message, Part.message_id == Message.id).where(Message.sequence < before_sequence)
    parts = list(db.execute(query.order_by(Part.time_updated.asc())).scalars())
    candidates = [p for p in parts if _has_output(p)]
    to_clear = candidates[:-keep] if keep > 0 else candidates
    changed = 0
    for part in to_clear:
        data = store._load(part.data)
        key = "error" if data["state"]["status"] == "error" else "output"
        data["state"][key] = _MICRO_PLACEHOLDER
        part.data = store._dump(data)
        changed += 1
    if changed:
        db.commit()
    return changed


def _has_output(part: Part) -> bool:
    data = store.part_data(part)
    return bool(data.get("output")) and data.get("output") != _MICRO_PLACEHOLDER


def build_transcript(history: list) -> str:
    lines: list[str] = []
    for msg in history:
        role = "用户" if msg.get("role") == "user" else "助手"
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
        elif isinstance(content, list):
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    lines.append(f"{role}: {block.get('text', '')}")
                elif btype == "tool_use":
                    lines.append(f"助手调用工具 {block.get('name')}，参数: {json_dumps_safe(block.get('input'))}")
                elif btype == "tool_result":
                    lines.append(f"工具结果: {block.get('content', '')}")
    return "\n".join(lines)


COMPACT_PROMPT = """你是会话压缩器。把下面这段"助手与用户的对话稿"压缩为结构化摘要，供助手后续继续工作使用。要求：
1. 【当前任务】逐字保留最后一条用户请求原文；
2. 【已完成】关键步骤与结果；
3. 【关键决定】重要方案决定；
4. 【涉及文件】创建/修改过的文件路径；
5. 【待办】未完成事项；
6. 【失败与教训】报错与规避方式；
7. 【用户约束】用户提出的硬性要求，逐字保留。
直接输出摘要正文，不要输出任何解释。"""


async def summarize_history(client, history: list) -> str:
    transcript = build_transcript(history)
    chunks: list[str] = []
    async for event in client.stream(
        system=COMPACT_PROMPT,
        messages=[{"role": "user", "content": f"<对话稿>\n{transcript}\n</对话稿>"}],
        tools=[],
    ):
        if event.get("type") == "text_delta":
            chunks.append(event.get("text", ""))
    return "".join(chunks).strip()


async def maybe_compact(deps, *, force: bool = False) -> AsyncIterator[dict]:
    """仅压缩旧轮前缀，保留活跃轮及工具配对。"""
    settings, history = deps.settings, deps.history
    active_start = getattr(deps, "active_history_start", None)
    if active_start is None:
        active_start = max(0, len(history) - 1)
    if not force and not needs_compact(settings, history, deps.last_input_tokens,
                                      getattr(deps, "measured_history_tokens", None)):
        return
    if active_start <= 0:
        return  # 当前轮过大且无安全前缀时失败，不硬截断。
    session_id = deps.recorder.session_id
    cutoff = deps.recorder.first_user_sequence - 1

    # 仅按 callID 替换旧前缀，保留未落库活跃历史。
    cleared = microcompact_parts(deps.recorder.db, session_id, settings.microcompact_keep,
                                 before_sequence=cutoff + 1)
    if cleared:
        cleared_ids = set()
        parts = deps.recorder.db.scalars(select(Part).where(Part.session_id == session_id)).all()
        for part in parts:
            data = store.part_data(part)
            if data.get("output") == _MICRO_PLACEHOLDER:
                cleared_ids.add(data.get("tool_call_id"))
        for message in history[:active_start]:
            if isinstance(message.get("content"), list):
                for block in message["content"]:
                    if block.get("type") == "tool_result" and block.get("tool_use_id") in cleared_ids:
                        block["content"] = _MICRO_PLACEHOLDER
        deps.last_input_tokens = None
        deps.measured_history_tokens = None
    if not force and not needs_compact(settings, history, deps.last_input_tokens,
                                      getattr(deps, "measured_history_tokens", None)):
        return
    if _failure_counts.get(session_id, 0) >= _MAX_COMPACT_FAILURES:
        logger.warning("compact 熔断中，跳过摘要 session=%s", session_id)
        return

    # 摘要失败时保持原历史与边界。
    prior = deepcopy(history[:active_start])
    tokens_before = estimate_history_tokens(history)
    try:
        summary_text = await asyncio.wait_for(summarize_history(deps.client, prior), timeout=120)
        if not isinstance(summary_text, str) or not summary_text.strip():
            raise ValueError("摘要为空，保留原上下文")
        summary_text = summary_text.strip()
        summary = {"role": "user", "content": f"<此前对话摘要>\n{summary_text}\n</此前对话摘要>"}
        if estimate_history_tokens([summary]) >= estimate_history_tokens(prior):
            raise ValueError("摘要未缩短上下文，保留原历史")
    except Exception as exc:
        _failure_counts[session_id] = _failure_counts.get(session_id, 0) + 1
        logger.warning("compact 摘要失败(%s/%s): %s", _failure_counts[session_id], _MAX_COMPACT_FAILURES, exc)
        return

    store.save_compaction(deps.recorder.db, session_id, deps.recorder.turn_id,
                          cutoff, summary_text, tokens_before, estimate_tokens(summary_text))
    _failure_counts[session_id] = 0
    deps.history[:] = [summary, *history[active_start:]]
    deps.active_history_start = 1
    deps.last_input_tokens = None
    deps.measured_history_tokens = None
    yield {"type": "compacted", "tokens_before": tokens_before,
           "tokens_after": estimate_history_tokens(deps.history)}


def reset_failure_count(session_id: str) -> None:
    _failure_counts.pop(session_id, None)
