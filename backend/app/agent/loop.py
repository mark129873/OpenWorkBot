import asyncio
import logging
import time
import uuid
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from app.agent import hooks
from app.agent.model_call import record_model_step
from app.agent.tools import TOOLS, execute_tool
from app.sessions.approvals import authorize_tool
from app.sessions.recorder import TurnRecorder

logger = logging.getLogger(__name__)

MAX_STEPS_DEFAULT = 40

# SSE 仅传预览，全文从回放读取。
_PREVIEW_LEN = 400


async def _auto_approver(tool_name: str, tool_input: dict, reason: str, **_kwargs) -> dict:
    """默认自动批准；HTTP 入口须注入交互审批。"""
    return {"approved": True, "denial_reason": None}


@dataclass
class TurnDeps:
    """settings 为 None 时禁用压缩。"""

    client: object  # 协议：async stream(system, messages, tools)。
    recorder: TurnRecorder
    system_prompt: str
    history: list = field(default_factory=list)
    tools: list = field(default_factory=lambda: list(TOOLS))
    approver: Callable = _auto_approver  # 异步返回 approved、denial_reason。
    max_steps: int = MAX_STEPS_DEFAULT
    stop_flag: Callable[[], bool] = field(default_factory=lambda: (lambda: False))
    settings: object = None
    last_input_tokens: int | None = None
    measured_history_tokens: int | None = None  # 调用前历史估算，用于计算新增用量。
    active_history_start: int | None = None  # 摘要不得跨越当前用户消息。
    queue: object | None = None
    persist_user_message: bool = True  # 重跑已有用户消息，不重复保存。
    user_sequence: int | None = None


def _preview(text: str) -> str:
    return text if len(text) <= _PREVIEW_LEN else text[:_PREVIEW_LEN] + "…"


def _is_error_output(output: str) -> bool:
    return isinstance(output, str) and output.startswith("Error:")


def _new_tool_part_id() -> str:
    """同一工具调用的状态更新复用此 ID。"""
    return f"tl{uuid.uuid4().hex[:30]}"


def _last_user_text(history: list) -> str:
    for msg in reversed(history):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""


async def run_turn(deps: TurnDeps) -> AsyncIterator[dict]:
    """最后发出 turn_completed 事件。"""
    turn_id = deps.recorder.begin_turn(
        user_text=_last_user_text(deps.history),
        persist_user=deps.persist_user_message,
        user_sequence=deps.user_sequence,
    )
    logger.info("turn 开始 session=%s turn=%s", deps.recorder.session_id, turn_id)
    yield {"type": "turn_started", "turn_id": turn_id, "started_at": deps.recorder.started_at}

    deps.active_history_start = max(0, len(deps.history) - 1)
    state = "success"
    stop_requested = False

    try:
        from app.agent.compact import (estimate_history_tokens, is_context_limit_error,
                                       maybe_compact, needs_compact)

        steps = 0
        context_retries = 0  # 每轮仅一次超限恢复，成功模型步不重置。
        while True:
            # 仅需续轮时检查步数，末步正常结束仍成功。
            if steps >= deps.max_steps:
                error_message = f"达到单轮步数上限 {deps.max_steps}"
                # 前置失败也落盘，刷新后可重试。
                deps.recorder.write_error_part(deps.recorder.step_message(), error_message)
                yield {"type": "error", "message": error_message}
                state = "failed"
                break
            if deps.stop_flag():
                stop_requested = True
                break

            if deps.settings is not None:
                async for compact_event in maybe_compact(deps):
                    yield compact_event
                if deps.stop_flag():
                    stop_requested = True
                    break
                if needs_compact(deps.settings, deps.history, deps.last_input_tokens,
                                 deps.measured_history_tokens):
                    error_message = "上下文仍超过预算，无法安全压缩当前轮；请缩短请求或工具输出后重试。"
                    deps.recorder.write_error_part(deps.recorder.step_message(), error_message)
                    yield {"type": "error", "message": error_message}
                    state = "failed"
                    break

            steps += 1
            input_estimate = estimate_history_tokens(deps.history)
            message_id = deps.recorder.step_message()
            reply = None  # 调用前失败可能尚无 reply。
            try:
                with record_model_step(deps, deps.system_prompt, deps.history, deps.tools) as reply:
                    async with aclosing(deps.client.stream(
                        system=deps.system_prompt, messages=deps.history, tools=deps.tools
                    )) as stream:
                        async for event in stream:
                            if deps.stop_flag():
                                stop_requested = True
                                break
                            reply.accept(event)
                            kind = event.get("type")
                            if kind == "text_delta":
                                yield {"type": "delta", "text": event.get("text", ""), "message_id": message_id}
                            elif kind == "usage":
                                deps.recorder.add_usage(**reply.usage)
                                deps.last_input_tokens = reply.usage["input_tokens"]
                                deps.measured_history_tokens = input_estimate
                                yield {"type": "token_count", "input_tokens": reply.usage["input_tokens"],
                                       "output_tokens": reply.usage["output_tokens"]}
            except Exception as exc:
                # 流已有内容时不重试，避免重复输出或工具计划。
                if (deps.settings is not None and is_context_limit_error(exc)
                        and context_retries < 1 and reply is not None
                        and not reply.text and not reply.tool_uses):
                    context_retries += 1
                    before = estimate_history_tokens(deps.history)
                    async for compact_event in maybe_compact(deps, force=True):
                        yield compact_event
                    if estimate_history_tokens(deps.history) < before:
                        deps.recorder.finish_message(message_id, "context_recovered")
                        continue
                if is_context_limit_error(exc):
                    exc = RuntimeError(f"上下文超限且无法继续安全恢复；请缩短请求或工具输出后重试。原错误：{exc}")
                logger.exception("LLM 调用失败")
                # 先保存已展示正文，避免异常后丢失部分答案。
                if reply is not None and reply.text:
                    deps.recorder.write_text_part(message_id, reply.text)
                deps.recorder.write_error_part(message_id, str(exc))
                yield {"type": "error", "message": str(exc)}
                state = "failed"
                break
            text_acc, tool_uses = reply.text, reply.tool_uses
            if text_acc:
                deps.recorder.write_text_part(message_id, text_acc)
            if not stop_requested and reply.stop_reason == "max_tokens":
                error_message = "模型输出达到上限，结果未完成；请重试或缩小任务范围。"
                deps.recorder.write_error_part(message_id, error_message, finish="max_tokens")
                yield {"type": "error", "message": error_message}
                state = "failed"
                break
            deps.recorder.finish_message(message_id, "interrupted" if stop_requested else
                                         (reply.stop_reason or ("tool_calls" if tool_uses else "stop")))
            part_ids: dict[str, str] = {}
            for tool_use in tool_uses:
                part_id = _new_tool_part_id()
                part_ids[tool_use["id"]] = part_id
                deps.recorder.upsert_tool_part(
                    message_id,
                    part_id,
                    {
                        "tool_call_id": tool_use["id"],
                        "name": tool_use["name"],
                        "input": tool_use.get("input") or {},
                        "status": "pending",
                    },
                )

            if stop_requested:
                break

            if not tool_uses:
                deps.history.append({"role": "assistant", "content": text_acc})
                injection = hooks.trigger("Stop", {"turn_id": turn_id, "steps": steps})
                if injection:
                    deps.history.append({"role": "user", "content": injection})
                    continue
                break

            deps.history.append(
                {"role": "assistant", "content": reply.assistant_blocks()}
            )
            tool_results: list[dict] = []
            for tool_use in tool_uses:
                if deps.stop_flag():
                    stop_requested = True
                    break
                result_box: dict = {}
                async for event in _execute_one(deps, message_id, tool_use, part_ids[tool_use["id"]], result_box):
                    yield event
                tool_results.append(result_box["tool_result"])
                # 子助手停止后立即结束父轮，避免误报预算或步数超限。
                if deps.stop_flag():
                    stop_requested = True
                    break

            if stop_requested:
                break
            if tool_results:
                deps.history.append({"role": "user", "content": tool_results})
    except Exception as exc:
        logger.exception("turn 循环异常")
        yield {"type": "error", "message": str(exc)}
        state = "failed"

    if stop_requested:
        state = "stopped"
    fact = deps.recorder.end_turn(state)
    logger.info(
        "turn 收口 session=%s turn=%s state=%s 工时=%.0fms tokens=%s",
        deps.recorder.session_id, turn_id, state, fact["active_ms"], fact["tokens_used"],
    )
    yield {
        "type": "turn_completed",
        "turn_id": turn_id,
        "ended_at": fact["ended_at"],
        "active_ms": fact["active_ms"],
        "state": state,
    }


async def _execute_one(
    deps: TurnDeps, message_id: str, tool_use: dict, part_id: str, result_box: dict
) -> AsyncIterator[dict]:
    """结果写入 result_box；拒绝时不执行、不发 tool_started。"""
    name = tool_use["name"]
    tool_input = tool_use.get("input") or {}
    call_id = tool_use["id"]
    started = time.time()

    hooks.trigger("PreToolUse", {"name": name, "tool_call_id": call_id})

    output = await authorize_tool(
        deps, name, tool_input, data_dir=deps.settings.data_dir if deps.settings else None
    )
    # 审批返回后重查停止，防止迟到批准执行。
    if deps.stop_flag():
        output = "Error: 操作已停止"
    status = "denied" if output is not None else "completed"
    if output is None:
        deps.recorder.upsert_tool_part(
            message_id, part_id,
            {"tool_call_id": call_id, "name": name, "input": tool_input, "status": "running"},
        )
        if name == "subtask":
            from app.agent.subtask import run_subtask_events

            async for event in run_subtask_events(deps, message_id, tool_input):
                if "_subtask_output" in event:
                    output = event["_subtask_output"]
                    status = event.get("_subtask_status", status)
                else:
                    yield event
        elif name == "todo_write":
            from app.agent.todo import handle_todo_write

            output, extra_events = await handle_todo_write(deps, message_id, tool_input)
            for event in extra_events:
                yield event
        else:
            output = await asyncio.to_thread(execute_tool, name, tool_input)
        if _is_error_output(output) and status != "stopped":
            status = "failed"

    duration_ms = (time.time() - started) * 1000
    deps.recorder.upsert_tool_part(
        message_id, part_id,
        {
            "tool_call_id": call_id, "name": name, "input": tool_input,
            "status": status, "output": output, "duration_ms": duration_ms,
        },
    )
    hooks.trigger(
        "PostToolUse",
        {"name": name, "tool_call_id": call_id, "status": status, "duration_ms": duration_ms},
    )

    if status in ("completed", "failed", "stopped"):
        yield {"type": "tool_started", "tool_call_id": call_id, "name": name, "input": tool_input}
    yield {
        "type": "tool_completed",
        "tool_call_id": call_id,
        "name": name,
        "status": status,
        "output_preview": _preview(output or ""),
    }
    result_box["tool_result"] = {"type": "tool_result", "tool_use_id": call_id, "content": output or "",
                                 **({"is_error": True} if status != "completed" else {})}
