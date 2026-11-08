import asyncio
import time
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from app import db
from app.agent import agent_bus, compact, hooks, skills, todo
from app.agent.model_call import record_model_step
from app.agent.task_tools import DIRECT_TOOLS, MAIN_TOOLS, SHARED_TOOLS
from app.config import settings
from app.models import new_id
from app.sessions import chat_store as store


async def dialog(runtime, note, *, source_seq=None, required_review=None, review_report_id=None, reply_id=None,
                 allow_direct=False, direct_task_id=None, direct_report_id=None):
    from app.agent.runtime import _PROMPT_PATH, _TERMINAL, _task_context

    generation = runtime.generation
    context = {"user_id": runtime.user_id, "owner_id": f"main:{new_id()}", "note": note,
               "step": 0, "unavailable": 0, "message_id": None, "text": ""}

    async def before_model(state, context):
        if generation != runtime.generation:
            raise asyncio.CancelledError
        if context.get("pending_calls"):
            outputs = [context["outputs"][call["id"]] for call in context.pop("pending_calls")]
            def save(session):
                group = store.get(session, f"group:{context['group_id']}")
                group["status"] = "completed"
                store.put(session, f"group:{context['group_id']}", group)
                store.append_history(session, "main", "user", outputs, key=f"result:{context['group_id']}")
            await runtime.change(save)
            context["note"] = "继续根据工具结果处理；不要重复创建已受理的任务。"
        if context["step"] >= 40:
            raise ValueError("主助手达到 40 步上限")
        with db.new_session() as session:
            tasks, omitted = _task_context(session, required_review)
            direct_task = store.get(session, f"task:{direct_task_id}") if direct_task_id else None
            context["decision_seq"] = store.get(session, "meta")["user_seq"]
            decisions = [event for event in store.records(session, "user_event")
                         if source_seq is not None and event["user_seq"] > source_seq]
            approvals = [item for item in store.records(session, "approval")
                         if item["status"] in {"requested", "resolved"}]
        messages = await compact.context_history(runtime, "main")
        task_note = ("部分任务已省略，请用 task_search 查找；" if omitted else "") + "任务为摘要，必要时用 task_inspect 查看详情。"
        if direct_task and direct_task.get("steps", 0) >= 40:
            task_note += "直接执行已达到 40 步上限，请停止并说明；后续工作可分配后台助手。"
        messages.append({"role": "user", "content": f"{context['note']}\n{task_note}\n"
                         f"较新的用户操作：{store.dump(decisions)}\n审批：{store.dump(approvals)}\n当前任务：{store.dump(tasks)}"})
        dialog_tools = [tool for tool in MAIN_TOOLS if required_review or tool["name"] != "task_review"] + SHARED_TOOLS
        if allow_direct and (not direct_task_id or (direct_task and direct_task["status"] not in _TERMINAL
                            and not direct_task.get("stop_requested") and direct_task.get("steps", 0) < 40)):
            dialog_tools += DIRECT_TOOLS
        context.update(messages=messages, tools=dialog_tools, message_id=None, text="", saved_length=0,
                       last_update=0.0, turn_id=new_id())

    async def model_call(state, context):
        step = context["step"]
        context["step"] += 1
        deps = SimpleNamespace(client=runtime.client,
                               recorder=SimpleNamespace(session_id="main", turn_id=context["turn_id"]))
        system = _PROMPT_PATH.read_text(encoding="utf-8") + skills.skill_prompt() + "\n" + todo.PROMPT
        messages = todo.remind(runtime.memory_messages(runtime.main_memory, context["messages"]), state)
        choice = {"tool_choice": {"type": "tool", "name": "task_review"}} if required_review and step == 0 else {}
        with record_model_step(deps, system, messages, context["tools"]) as reply:
            async for event in runtime.client.stream(system=system, messages=messages,
                                                     tools=context["tools"], **choice):
                if generation != runtime.generation:
                    raise asyncio.CancelledError
                reply.accept(event)
                if event.get("type") == "text_delta" and event.get("text"):
                    context["text"] += event["text"]
                    if context["message_id"] is None:
                        context["message_id"] = await runtime.change(lambda session: store.add_visible(
                            session, "assistant", context["text"], status="streaming"))
                        context["saved_length"], context["last_update"] = len(context["text"]), time.monotonic()
                    elif (len(context["text"]) - context["saved_length"] >= 64
                          or time.monotonic() - context["last_update"] >= 0.12):
                        await runtime.change(lambda session: store.update_visible(
                            session, context["message_id"], text=context["text"]))
                        context["saved_length"], context["last_update"] = len(context["text"]), time.monotonic()
        if generation != runtime.generation:
            raise asyncio.CancelledError
        context["reply"] = reply
        return AIMessage(content=reply.text, tool_calls=[{"id": call["id"], "name": call["name"],
                         "args": call.get("input") or {}, "type": "tool_call"} for call in reply.tool_uses])

    async def after_model(state, context):
        reply = context["reply"]
        message_id, text = context["message_id"], context["text"]
        if reply.stop_reason == "max_tokens":
            raise ValueError("主助手输出达到上限")
        if required_review and context["step"] == 1:
            if (len(reply.tool_uses) != 1 or reply.tool_uses[0].get("name") != "task_review"
                    or (reply.tool_uses[0].get("input") or {}).get("task_id") != required_review):
                raise ValueError("模型没有对指定任务作出有效判断")
        if any(call.get("name") not in {tool["name"] for tool in context["tools"]} for call in reply.tool_uses):
            context["unavailable"] += 1
            blocks = reply.assistant_blocks()
            results = [{"type": "tool_result", "tool_use_id": call["id"],
                        "content": "Error: 当前工具不可用，本步未执行。", "is_error": True} for call in reply.tool_uses]
            def unavailable(session):
                if message_id:
                    store.update_visible(session, message_id, text=text, status="interrupted")
                store.append_history(session, "main", "assistant", blocks)
                store.append_history(session, "main", "user", results)
            await runtime.change(unavailable)
            if context["unavailable"] >= 2:
                def acknowledge(session):
                    text = "这条回复暂时无法完成，请重试。"
                    store.add_visible(session, "assistant", text, status="failed")
                    store.append_history(session, "main", "assistant", text, key=f"failed:{new_id()}")
                    if reply_id:
                        row = store.get(session, f"reply:{reply_id}")
                        if row and row["status"] == "pending":
                            row["status"] = "failed"
                            store.put(session, f"reply:{reply_id}", row)
                        event = store.get(session, f"input:{reply_id}")
                        if event and event["status"] == "pending":
                            event["status"] = "failed"
                            store.put(session, f"input:{reply_id}", event)
                    if direct_task_id:
                        task = store.get(session, f"task:{direct_task_id}")
                        if task and task["status"] not in _TERMINAL:
                            task.update(status="failed", direct_phase="finished", result_sent=True)
                            store.put(session, f"task:{direct_task_id}", task)
                            runtime._release_dependents(session, task)
                        if direct_report_id:
                            agent_bus.processed(session, store.get(session, f"bus:{direct_report_id}"))
                await runtime.change(acknowledge)
                return {"jump_to": "end"}
            context["note"] = "刚才的工具调用没有执行，请直接回复用户；任务结果由系统送达后再判断。"
            return {"jump_to": "model"}

        def update_usage(session):
            meta = store.get(session, "meta")
            meta["context_usage"] = {"used": reply.usage.get("input_tokens", 0), "window": settings.context_window}
            store.put(session, "meta", meta)
        await runtime.change(update_usage, "usage")
        blocks = reply.assistant_blocks()
        if not blocks:
            raise ValueError("主助手没有返回内容")
        if not reply.tool_uses:
            injection = hooks.trigger("Stop", {"turn_id": context["turn_id"], "steps": context["step"]})
            def finish_reply(session):
                row = store.get(session, f"reply:{reply_id}") if reply_id else None
                if reply_id and (not row or row["status"] != "pending"):
                    if message_id:
                        store.update_visible(session, message_id, text=text, status="interrupted")
                    return False
                if message_id:
                    store.update_visible(session, message_id, text=text, status="completed")
                store.append_history(session, "main", "assistant", blocks)
                if reply_id and not (isinstance(injection, str) and injection):
                    row["status"] = "completed"
                    store.put(session, f"reply:{reply_id}", row)
                    event = store.get(session, f"input:{reply_id}")
                    if event and event["status"] == "pending":
                        event["status"] = "processed"
                        store.put(session, f"input:{reply_id}", event)
                if direct_task_id and not (isinstance(injection, str) and injection):
                    task = store.get(session, f"task:{direct_task_id}")
                    if task and task["status"] not in _TERMINAL:
                        task["status"] = ("stopped" if task.get("stop_requested") else
                                          "failed" if task.get("direct_failed") else "completed")
                        task["direct_phase"] = "finished"
                        task["result_sent"] = True
                        store.put(session, f"task:{direct_task_id}", task)
                        runtime._release_dependents(session, task)
                    if direct_report_id:
                        agent_bus.processed(session, store.get(session, f"bus:{direct_report_id}"))
                return True
            if not await runtime.change(finish_reply):
                return {"jump_to": "end"}
            if isinstance(injection, str) and injection:
                await runtime.change(lambda session: store.append_history(session, "main", "user", injection))
                return {"jump_to": "model"}
            return {"jump_to": "end"}
        if message_id:
            await runtime.change(lambda session: store.update_visible(session, message_id, text=text, status="completed"))
        if any(call.get("name") in {tool["name"] for tool in DIRECT_TOOLS} for call in reply.tool_uses):
            await runtime.change(lambda session: runtime._queue_direct(
                session, blocks, reply.tool_uses, source_seq, reply_id, direct_task_id, direct_report_id,
                context["decision_seq"]))
            return {"jump_to": "end"}
        ids = [call.get("id") for call in reply.tool_uses]
        if any(not isinstance(identity, str) or not identity for identity in ids) or len(ids) != len(set(ids)):
            raise ValueError("模型返回了无效或重复的工具调用 ID")
        context.update(pending_calls=reply.tool_uses, outputs={}, group_id=f"main:{context['turn_id']}")
        def register(session):
            row = store.get(session, f"reply:{reply_id}") if reply_id else None
            if reply_id and (not row or row["status"] != "pending"):
                return False
            if reply_id:
                event = store.get(session, f"input:{reply_id}")
                if event and event["status"] == "pending":
                    event["status"] = "processed"
                    store.put(session, f"input:{reply_id}", event)
            store.put(session, f"group:{context['group_id']}", {"type": "main_tool_group", "context_id": "main",
                      "id": context["group_id"], "calls": reply.tool_uses, "outputs": {}, "status": "pending"})
            store.append_history(session, "main", "assistant", blocks, key=f"step:{context['group_id']}")
            return True
        if not await runtime.change(register):
            context.pop("pending_calls")
            return {"jump_to": "end"}

    async def tool_call(call, context):
        if generation != runtime.generation:
            raise asyncio.CancelledError
        original = next((item for item in context["pending_calls"] if item["id"] == call["id"]), None)
        if not original or original["name"] != call["name"] or (original.get("input") or {}) != call["args"]:
            raise ValueError("工具调用与模型消息不一致")
        started_at = time.monotonic()
        hook_context = {"name": call["name"], "tool_call_id": call["id"]}
        hooks.trigger("PreToolUse", hook_context)
        def execute(session):
            seq = context["decision_seq"] if call["name"] in {"task_update", "task_stop", "write_todos"} else source_seq
            output = runtime._main_tool(session, {**original, "id": f"{context['group_id']}:{call['id']}"},
                                        seq, review_report_id)
            group = store.get(session, f"group:{context['group_id']}")
            group["outputs"][call["id"]] = output
            store.put(session, f"group:{context['group_id']}", group)
            return output
        try:
            output = await runtime.change(execute)
        except Exception:
            hooks.trigger("PostToolUse", {**hook_context, "status": "error", "duration_ms": (time.monotonic() - started_at) * 1000})
            raise
        hooks.trigger("PostToolUse", {**hook_context, "status": "error" if output.startswith("Error:") else "completed",
                                      "duration_ms": (time.monotonic() - started_at) * 1000})
        context["outputs"][call["id"]] = {"type": "tool_result", "tool_use_id": call["id"], "content": output,
                                          **({"is_error": True} if output.startswith("Error:") else {})}
        return output

    async def skip_tool_call(call, context, output):
        def save(session):
            group = store.get(session, f"group:{context['group_id']}")
            group["outputs"][call["id"]] = output
            store.put(session, f"group:{context['group_id']}", group)
        await runtime.change(save)
        context["outputs"][call["id"]] = {"type": "tool_result", "tool_use_id": call["id"],
                                          "content": output, "is_error": True}

    async def close_group():
        if not context.get("pending_calls"):
            return
        def save(session):
            group = store.get(session, f"group:{context['group_id']}")
            if not group or group["status"] == "completed":
                return
            outputs = []
            for call in group["calls"]:
                output = group["outputs"].get(call["id"], "Error: 回复已中断，工具未执行。")
                outputs.append({"type": "tool_result", "tool_use_id": call["id"], "content": output,
                                **({"is_error": True} if output.startswith("Error:") else {})})
            store.append_history(session, "main", "user", outputs, key=f"result:{context['group_id']}")
            group["status"] = "completed"
            store.put(session, f"group:{context['group_id']}", group)
        await runtime.change(save)

    async with runtime.graph.write_lock:
        thread_id = runtime.graph.thread_id("main:loop", parent_context="main")
    context.update(thread_id=thread_id, before_model=before_model, model_call=model_call,
                   after_model=after_model, tool_call=tool_call, skip_tool_call=skip_tool_call)
    try:
        await runtime.graph.invoke_loop(runtime.agent_loop, thread_id, context)
    except asyncio.CancelledError:
        await asyncio.shield(close_group())
        if context["message_id"]:
            def interrupt(session):
                store.update_visible(session, context["message_id"], text=context["text"], status="interrupted")
                store.append_history(session, "main", "assistant", [{"type": "text", "text": context["text"]}],
                                     key=f"interrupted:{context['message_id']}")
            await asyncio.shield(runtime.change(interrupt))
        raise
    except Exception:
        await close_group()
        if context["message_id"]:
            await runtime.change(lambda session: store.update_visible(session, context["message_id"],
                                                                      text=context["text"], status="failed"))
        raise
    finally:
        if direct_task_id:
            with db.new_session() as session:
                task = store.get(session, f"task:{direct_task_id}")
            if task and task["status"] in _TERMINAL:
                try:
                    await runtime.release_direct_sandbox(direct_task_id)
                except Exception:
                    pass
