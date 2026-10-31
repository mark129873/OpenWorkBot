import asyncio
import logging
from contextlib import aclosing

from app.agent.model_call import record_model_step
from app.agent.tools import TOOL_HANDLERS, TOOLS, execute_tool
from app.sessions.approvals import authorize_tool

logger = logging.getLogger(__name__)

SUBTASK_MAX_TURNS = 30

# 排除依赖父循环上下文的工具，禁止递归子任务。
SUBTASK_TOOLS = [t for t in TOOLS if t["name"] in TOOL_HANDLERS]

SUBTASK_SYSTEM_PROMPT = (
    "你是子助手，负责完成主助手委派的一个目标。规则：\n"
    "1. 直接执行，不要向用户提问（你面对的不是用户，是主助手）；\n"
    "2. 所有路径都在工作区沙箱内；\n"
    "3. 工具报错时阅读 Error 修正重试；\n"
    "4. 完成后用中文简要汇报结果。"
)


async def run_subtask_events(deps, message_id: str, tool_input: dict):
    """末尾用 _subtask_output 哨兵向父循环返回结果。"""
    from app.models import new_id

    goal = str(tool_input.get("goal", "")).strip()
    if not goal:
        yield {"_subtask_output": "Error: subtask 缺少 goal 参数"}
        return

    subtask_id = new_id()
    await deps.queue.put({"type": "subtask_started", "subtask_id": subtask_id, "goal": goal})

    child_history: list[dict] = [{"role": "user", "content": goal}]
    output_acc: list[str] = []
    status = "completed"

    try:
        for _turn in range(SUBTASK_MAX_TURNS):
            if deps.stop_flag():
                status = "stopped"
                break
            with record_model_step(
                deps, SUBTASK_SYSTEM_PROMPT, child_history, SUBTASK_TOOLS, subtask_id=subtask_id
            ) as reply:
                # 逐事件检查停止，退出时关闭 SDK 流。
                async with aclosing(deps.client.stream(
                    system=SUBTASK_SYSTEM_PROMPT, messages=child_history, tools=SUBTASK_TOOLS
                )) as stream:
                    async for event in stream:
                        if deps.stop_flag():
                            status = "stopped"
                            break
                        reply.accept(event)
                        if event.get("type") == "text_delta":
                            # 收到即保存，断线或停止不丢已展示内容。
                            text = event.get("text", "")
                            output_acc.append(text)
                            await deps.queue.put(
                                {"type": "subtask_delta", "subtask_id": subtask_id, "text": text}
                            )
                        elif event.get("type") == "usage":
                            # 子模型计入父轮，不覆盖主模型上下文预算。
                            deps.recorder.add_usage(**reply.usage, subtask=True)
                            yield {"type": "token_count", **reply.usage}
            if deps.stop_flag():
                status = "stopped"
            if status == "stopped":
                break
            tool_uses = reply.tool_uses
            if not tool_uses:
                break

            child_history.append(
                {
                    "role": "assistant",
                    "content": reply.assistant_blocks(),
                }
            )
            tool_results = []
            for t in tool_uses:
                if deps.stop_flag():
                    status = "stopped"
                    break
                # 子工具沿用父会话权限与审批。
                out = await authorize_tool(
                    deps, t["name"], t.get("input") or {}
                )
                # 等待时停止优先，迟到批准不执行。
                if deps.stop_flag():
                    status = "stopped"
                    break
                if out is None:
                    out = await asyncio.to_thread(
                        execute_tool, t["name"], t.get("input") or {}
                    )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": t["id"], "content": out}
                )
            if status == "stopped":
                break
            child_history.append({"role": "user", "content": tool_results})
        else:
            output_acc.append(f"\n[达到子助手轮数上限 {SUBTASK_MAX_TURNS}]")
    except Exception as exc:
        logger.exception("subtask 执行异常")
        status = "failed"
        output_acc.append(f"\nError: {exc}")

    output = "".join(output_acc).strip() or "(子助手无输出)"
    if status == "failed" and not output.startswith("Error:"):
        output = "Error: " + output

    await deps.queue.put({"type": "subtask_completed", "subtask_id": subtask_id, "status": status})
    yield {"_subtask_output": output, "_subtask_status": status}
