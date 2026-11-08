import asyncio
from copy import deepcopy

_MICRO_PLACEHOLDER = "[旧工具结果已清除以节省上下文]"


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
        elif event.get("type") == "stop_reason" and event.get("reason") == "max_tokens":
            raise ValueError("摘要输出达到上限")
    return "".join(chunks).strip()


async def prepare_history(client, history: list, config, *, on_summary=None) -> list:
    """只压缩发送给模型的副本，不修改保存的原始记录。"""
    projected = deepcopy(history)
    if not needs_compact(config, projected):
        return projected
    results = [block for message in projected if isinstance(message.get("content"), list)
               for block in message["content"] if block.get("type") == "tool_result"]
    keep = max(0, config.microcompact_keep)
    for block in results[:-keep] if keep else results:
        block["content"] = _MICRO_PLACEHOLDER
    if not needs_compact(config, projected):
        return projected
    boundaries = [i for i, message in enumerate(projected) if i > 0 and (
        (message.get("role") == "user" and isinstance(message.get("content"), str))
        or (message.get("role") == "assistant" and projected[i - 1].get("role") == "user"))]
    if not boundaries:
        raise ValueError("上下文过长且没有可压缩的完整前缀")
    cutoff = boundaries[-1]
    summary = await asyncio.wait_for(summarize_history(client, projected[:cutoff]), timeout=120)
    if not summary or estimate_history_tokens([{"role": "user", "content": summary}]) >= estimate_history_tokens(projected[:cutoff]):
        raise ValueError("上下文摘要无效，保留原始记录")
    projected = [{"role": "user", "content": f"<此前对话摘要>\n{summary}\n</此前对话摘要>"}, *projected[cutoff:]]
    if needs_compact(config, projected):
        raise ValueError("压缩后上下文仍然过长")
    if on_summary:
        await on_summary(cutoff, summary)
    return projected


async def context_history(runtime, context_id: str) -> list:
    from app import db
    from app.config import settings
    from app.sessions import chat_store

    with db.new_session() as session:
        rows = chat_store.records(session, "agent_context", context_id=context_id)
        saved = chat_store.get(session, f"summary:{context_id}")
    start = 0
    if saved:
        boundary = next((i for i, row in enumerate(rows) if row["id"] == saved.get("boundary_id")), None)
        if boundary is not None:
            start = boundary + 1
    messages = [{"role": row["role"], "content": row["content"]} for row in rows[start:]]
    ids = [row["id"] for row in rows[start:]]
    if saved and start:
        messages.insert(0, {"role": "user", "content": f"<此前对话摘要>\n{saved['summary']}\n</此前对话摘要>"})
        ids.insert(0, saved["boundary_id"])

    async def save_summary(cutoff: int, summary: str) -> None:
        boundary_id = ids[cutoff - 1]
        await runtime.change(lambda session: chat_store.put(session, f"summary:{context_id}", {
            "type": "agent_summary", "context_id": context_id, "boundary_id": boundary_id, "summary": summary}))

    return await prepare_history(runtime.client, messages, settings, on_summary=save_summary)
