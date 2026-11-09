import asyncio
import json
import logging
import math
from copy import deepcopy
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


class ContextCompactionError(ValueError):
    pass


def context_budget(settings) -> int:
    return max(1, settings.context_window - settings.compact_output_reserve - settings.compact_buffer)


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning("token 词表不可用，使用字符估算")
        return None


def estimate_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding:
        return len(encoding.encode(text, disallowed_special=()))
    return max(len(text) // 3, len(text.encode("utf-8")) // 2, 1)


def json_dumps_safe(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(obj)


def estimate_history_tokens(history: list) -> int:
    return sum(estimate_tokens(json_dumps_safe(message)) + 4 for message in history)


def estimate_request_tokens(system: str, messages: list, tools: list) -> int:
    return estimate_tokens(system) + estimate_history_tokens(messages) + estimate_tokens(json_dumps_safe(tools)) + 8


def record_usage(context: dict, system: str, messages: list, tools: list, usage: dict, model: str,
                 *, separate_cache: bool = False) -> None:
    actual = usage.get("input_tokens", 0)
    if separate_cache and isinstance(actual, int):
        actual += (usage.get("cache_read_tokens") or 0) + (usage.get("cache_creation_tokens") or 0)
    if not isinstance(actual, int) or actual <= 0:
        context.pop("compact_usage", None)
        return
    context["compact_usage"] = {"system": system, "messages": deepcopy(messages), "tools": deepcopy(tools),
                                "model": model, "estimated": estimate_request_tokens(system, messages, tools),
                                "actual": actual}


def request_tokens(context: dict, system: str, messages: list, tools: list, model: str) -> int:
    estimated = estimate_request_tokens(system, messages, tools)
    anchor = context.get("compact_usage") or {}
    previous = anchor.get("messages")
    same_prefix = (isinstance(previous, list) and (messages[:len(previous)] == previous or
                   (bool(previous) and len(messages) >= len(previous) and messages[-1] == previous[-1]
                    and messages[:len(previous) - 1] == previous[:-1])))
    if (anchor.get("model") == model and anchor.get("system") == system and anchor.get("tools") == tools
            and same_prefix):
        estimated = max(estimated, anchor["actual"] + max(0, estimated - anchor["estimated"]))
    return estimated


def needs_compact(settings, message_count: int, tokens: int) -> bool:
    fraction = settings.compact_trigger_fraction
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("上下文压缩比例配置无效")
    return (message_count >= max(1, settings.compact_trigger_messages)
            or tokens >= max(1, settings.compact_trigger_tokens)
            or tokens >= max(1, int(settings.context_window * fraction))
            or tokens >= context_budget(settings))


def is_context_limit_error(exc: Exception) -> bool:
    if getattr(exc, "code", None) in {"context_length_exceeded", "prompt_too_long"}:
        return True
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
    lines = []
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


COMPACT_PROMPT = """你是会话压缩器。将已有摘要和较早对话一次性整理为简洁、忠实的结构化摘要，供助手继续工作。
记录任务目标、用户约束、重要决定、已完成工作、关键工具结果和文件路径、错误与未完成事项。
保留相关的工具结果文件路径，不臆造或忽略不确定性。较新的用户要求优先于旧决定。
直接输出摘要正文，不要输出解释。"""


async def summarize_history(client, history: list, previous_summary: str = "") -> str:
    transcript = build_transcript(history)
    text = f"<已有摘要>\n{previous_summary}\n</已有摘要>\n<对话稿>\n{transcript}\n</对话稿>"
    chunks = []
    completed = False
    async for event in client.stream(system=COMPACT_PROMPT, messages=[{"role": "user", "content": text}], tools=[]):
        if event.get("type") == "text_delta":
            chunks.append(event.get("text", ""))
        elif event.get("type") == "stop_reason":
            if event.get("reason") in {"max_tokens", "length"}:
                raise ContextCompactionError("摘要输出达到上限，已保留原始上下文。")
            if event.get("reason") not in {"end_turn", "stop"}:
                raise ContextCompactionError("上下文摘要输出不完整，请稍后重试。")
            completed = True
        elif event.get("type") == "tool_use":
            raise ContextCompactionError("上下文摘要无效，请稍后重试。")
    if not completed:
        raise ContextCompactionError("上下文摘要输出不完整，请稍后重试。")
    return "".join(chunks).strip()


def _safe_cutoff(messages: list, keep: int) -> int:
    cutoff = max(0, len(messages) - keep)
    if cutoff < len(messages) and isinstance(messages[cutoff], ToolMessage):
        call_id = messages[cutoff].tool_call_id
        for index in range(cutoff - 1, -1, -1):
            message = messages[index]
            if isinstance(message, AIMessage) and any(call.get("id") == call_id for call in message.tool_calls):
                return index
        while cutoff < len(messages) and isinstance(messages[cutoff], ToolMessage):
            cutoff += 1
    return cutoff


def select_messages(messages: list, keep: int) -> tuple[list, list]:
    cutoff = _safe_cutoff(messages, max(0, keep))
    latest_user = next((message for message in reversed(messages) if isinstance(message, HumanMessage)
                        and str(message.id or "").startswith("input:")), None)
    latest_update = next((message for message in reversed(messages) if isinstance(message, HumanMessage)
                          and str(message.id or "").startswith("bus:")), None)
    latest_human = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
    protected = {id(message) for message in (latest_user, latest_update, latest_human) if message is not None}
    old, rescued = [], []
    for message in messages[:cutoff]:
        identity = str(message.id or "")
        metadata = message.additional_kwargs or {}
        if (id(message) in protected or isinstance(message, SystemMessage)
                or identity.startswith(("goal:", "help:"))
                or metadata.get("dynamic_context_reminder")):
            rescued.append(message)
        else:
            old.append(message)
    return old, [*rescued, *messages[cutoff:]]


def history_for_state(state: dict) -> list:
    from app.agent.context import projected_history

    messages, identities = projected_history(state.get("messages", []))
    for message, ids in zip(messages, identities):
        if (message["role"] == "user" and isinstance(message["content"], str)
                and any(identity.startswith("bus:") for identity in ids)):
            message["content"] = "内部消息（非用户直接输入）：\n" + message["content"]
    if summary := state.get("summary_text"):
        messages.insert(0, {"role": "user", "content": f"<此前对话摘要>\n{summary}\n</此前对话摘要>"})
    return messages


def state_instructions(state: dict) -> str:
    return "\n".join(str(message.content) for message in state.get("messages", [])
                     if isinstance(message, SystemMessage))


async def _expand_archived(runtime, history: list, previous: str, budget: int) -> list:
    from app.agent.tool_output import read_reference

    result = deepcopy(history)
    body = f"<已有摘要>\n{previous}\n</已有摘要>\n<对话稿>\n{build_transcript(result)}\n</对话稿>"
    remaining = budget - await runtime.run_tool(estimate_request_tokens, COMPACT_PROMPT,
                                                [{"role": "user", "content": body}], [])
    for message in result:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            content = block.get("content")
            if (block.get("type") != "tool_result" or not isinstance(content, str)
                    or "[完整工具结果：" not in content or remaining <= 0):
                continue
            original = await runtime.run_tool(read_reference, content, max_chars=max(1, remaining * 4))
            if original is None:
                continue
            footer = content[content.rfind("[完整工具结果："):]
            replacement = f"{original}\n{footer}"
            delta = await runtime.run_tool(lambda: estimate_tokens(replacement) - estimate_tokens(content))
            if delta < remaining:
                block["content"] = replacement
                remaining -= max(0, delta)
    return result


async def compact_state(runtime, state: dict, context: dict, assemble, *, force: bool = False) -> dict | None:
    from app.agent.context import projected_history
    from app.config import settings

    system, full_messages, tools = assemble(state, context)
    tokens = await runtime.run_tool(request_tokens, context, system, full_messages, tools,
                                    getattr(runtime.client, "model", ""))
    messages = state.get("messages", [])
    count = len(messages) + bool(state.get("summary_text"))
    if not force and not needs_compact(settings, count, tokens):
        return None

    keep = max(0, settings.compact_keep_messages)
    old, preserved = select_messages(messages, keep)
    while keep:
        next_state = {**state, "messages": preserved, "summary_text": ""}
        next_system, next_messages, next_tools = assemble(next_state, context)
        summary_reserve = min(max(1, settings.max_tokens), max(1, context_budget(settings) // 4))
        next_tokens = await runtime.run_tool(estimate_request_tokens, next_system, next_messages, next_tools)
        if next_tokens + summary_reserve < context_budget(settings):
            break
        keep -= 1
        old, preserved = select_messages(messages, keep)
    if not old and (force or tokens >= context_budget(settings)):
        old, preserved = select_messages(messages, 0)
    if not old:
        if force or tokens >= context_budget(settings):
            raise ContextCompactionError("上下文过长且没有可压缩的历史，请缩短输入。")
        return None
    if any(not message.id for message in old):
        raise ContextCompactionError("上下文消息无法安全压缩，请稍后重试。")

    history, _ = projected_history(old)
    previous = state.get("summary_text") or ""
    history = await _expand_archived(runtime, history, previous, context_budget(settings))
    summary_input = f"<已有摘要>\n{previous}\n</已有摘要>\n<对话稿>\n{build_transcript(history)}\n</对话稿>"
    summary_tokens = await runtime.run_tool(estimate_request_tokens, COMPACT_PROMPT,
                                            [{"role": "user", "content": summary_input}], [])
    if summary_tokens >= context_budget(settings):
        raise ContextCompactionError("待压缩上下文过长，无法一次性摘要，请缩短输入。")
    try:
        summary = await asyncio.wait_for(summarize_history(runtime.client, history, previous), timeout=120)
    except ContextCompactionError:
        raise
    except Exception as exc:
        raise ContextCompactionError("上下文摘要失败，请稍后重试。") from exc
    source_tokens = await runtime.run_tool(estimate_tokens, previous + build_transcript(history))
    if not summary or estimate_tokens(summary) >= source_tokens:
        raise ContextCompactionError("上下文摘要无效，已保留原始记录。")
    new_state = {**state, "messages": preserved, "summary_text": summary}
    next_system, next_messages, next_tools = assemble(new_state, context)
    next_tokens = await runtime.run_tool(estimate_request_tokens, next_system, next_messages, next_tools)
    if next_tokens >= context_budget(settings):
        raise ContextCompactionError("压缩后上下文仍然过长，请缩短输入。")
    if flush := context.get("flush_memory"):
        try:
            await flush(old)
        except Exception:
            logger.exception("压缩前长期记忆提取提交失败")
    context.pop("compact_usage", None)
    return {"messages": [RemoveMessage(id=message.id) for message in old], "summary_text": summary}
