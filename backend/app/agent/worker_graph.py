import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from app import db
from app.agent import compact, context as agent_context, hooks, shared_state, skills, todo
from app.agent.model_call import record_model_step
from app.agent.task_tools import WORKER_TOOLS
from app.models import new_id
from app.sessions import chat_store as store


def _execution(call, context):
    with db.new_session() as session:
        group = store.get(session, f"group:{context['group_id']}")
        executions = [store.get(session, f"execution:{identity}") for identity in group["calls"]]
    execution = next((item for item in executions if item["call_id"] == call["id"]), None)
    if not execution or execution["name"] != call["name"] or execution["input"] != call["args"]:
        raise ValueError("工具调用与执行记录不一致")
    return execution


async def _tool_call(worker, call, context):
    execution = _execution(call, context)
    worker.sandbox = context.get("sandbox")
    worker.sandbox_scope = context["owner_id"]
    return await worker.process_execution(execution["id"])


async def _invoke(worker, before_model, model_call, after_model=None, *, group_id=None, completed_calls=None, assemble=None):
    runtime = worker.runtime
    async with runtime.graph.write_lock:
        thread_id = runtime.graph.thread_id(f"{worker.context_id}:loop", parent_context=worker.context_id)

    async def tool_call(call, context):
        from app.agent.worker import AgentWaiting

        try:
            return await _tool_call(worker, call, context)
        except AgentWaiting:
            if context.get("retain_sandbox"):
                context["retain_on_error"] = True
            raise

    async def skip_tool_call(call, context, output):
        execution = _execution(call, context)
        def skip(session):
            current = store.get(session, f"execution:{execution['id']}")
            current.update(status="completed", output=output)
            store.put(session, f"execution:{execution['id']}", current)
        await runtime.change(skip)

    direct = worker.task().get("direct")
    if direct:
        context = runtime.direct_sandboxes.setdefault(worker.task_id, {
            "user_id": runtime.user_id, "thread_id": f"{runtime.user_id}:{worker.context_id}:{worker.run_id}",
            "owner_id": f"direct:{worker.task_id}:{worker.run_id}", "retain_sandbox": True})

        async def apply_todos(items, call):
            execution = _execution(call, context)
            async with runtime.graph.write_lock:
                main_thread = runtime.graph.thread_id("main:loop", parent_context="main")
            async with runtime.graph.locks.setdefault(main_thread, asyncio.Lock()):
                async with runtime.lock:
                    with db.new_session() as session:
                        seq = execution.get("decision_seq")
                        stale = seq is not None and any(
                            item.get("event_type") == "message" and item["user_seq"] > seq
                            for item in store.records(session, "user_event"))
                    if not stale:
                        await runtime.agent_loop.graph.aupdate_state({"configurable": {"thread_id": main_thread}},
                                                                     {"todos": items})
                if stale:
                    error = "Error: 已有更新的用户消息，请重新判断 TODO。"
                    def fail(session):
                        current = store.get(session, f"execution:{execution['id']}")
                        current["output"] = error
                        store.put(session, f"execution:{execution['id']}", current)
                    await runtime.change(fail)
                    return error
        context["apply_todos"] = apply_todos
    elif worker.keep_sandbox:
        key = f"{worker.context_id}:{worker.run_id}"
        context = runtime.worker_sandboxes.setdefault(key, {
            "user_id": runtime.user_id, "thread_id": f"{runtime.user_id}:{key}", "owner_id": f"worker:{key}",
            "retain_sandbox": True, "task_id": worker.task_id, "context_id": worker.context_id,
            "run_id": worker.run_id})
    else:
        context = {"user_id": runtime.user_id, "thread_id": thread_id, "owner_id": f"{worker.run_id}:{new_id()}"}
    context.pop("injected", None)
    context.pop("retain_on_error", None)
    context.pop("_ordered_tool_groups", None)
    context.pop("compacted_context", None)

    async def acknowledge(identities):
        await agent_context.acknowledge(runtime, worker.context_id, identities)

    async def sync_todos(items, call):
        actor = "main" if direct else worker.context_id
        execution_id = None if direct else _execution(call, context)["id"]
        await runtime.change(lambda session: shared_state.replace(
            session, actor, worker.task_id, worker.run_id, items,
            thread_id=None if direct else thread_id, execution_id=execution_id))

    async def confirm_todos(items):
        if direct or (group_id is not None and not context.get("injected")):
            return
        with db.new_session() as session:
            marker = store.get(session, f"todo-sync:{worker.context_id}")
        if marker and marker.get("run_id") == worker.run_id:
            await runtime.change(lambda session: shared_state.confirm(session, worker.context_id, worker.run_id, items))

    async def compact_before_model(state, context):
        from app.agent.worker import AgentWaiting

        async def ensure_running():
            if worker.stop.is_set() or worker.task().get("stop_requested"):
                await worker.finish("stopped", "任务已停止；已开始的操作可能已产生副作用。")
                raise AgentWaiting

        await ensure_running()
        result = await compact.compact_state(runtime, state, context, assemble, force=context.pop("force_compact", False))
        await ensure_running()
        if result:
            if not await worker.deliver_messages():
                context["retain_on_error"] = True
                raise AgentWaiting
            await ensure_running()
            if not worker.helper_id:
                def pause(session):
                    current = store.get(session, f"task:{worker.task_id}")
                    if not current or not current.get("dependency"):
                        return False
                    current["status"] = "waiting_dependency"
                    store.put(session, f"task:{worker.task_id}", current)
                    return True
                if await runtime.change(pause):
                    context["retain_on_error"] = True
                    raise AgentWaiting
            records = agent_context.pending(worker.context_id)
            if records:
                context["pending_context_records"] = records
                result["messages"] += agent_context.messages_from_records(records)
        return result

    context.update(before_model=before_model, model_call=model_call, tool_call=tool_call, skip_tool_call=skip_tool_call,
                   after_model=after_model, group_id=group_id,
                   completed_calls=completed_calls or {},
                   collect_pending=lambda: agent_context.pending(worker.context_id), acknowledge_pending=acknowledge,
                   compact_before_model=compact_before_model if assemble else None,
                   sync_todos=sync_todos, confirm_todos=confirm_todos)
    await runtime.graph.invoke_loop(runtime.agent_loop, thread_id, context)


async def process_group(worker, group_id):
    with db.new_session() as session:
        group = store.get(session, f"group:{group_id}")
        executions = [store.get(session, f"execution:{identity}") for identity in group["calls"]]
    pending = [item for item in executions if item["status"] != "completed"]
    completed_todo = any(item["name"] == "write_todos" and not item.get("result_unknown")
                         and not item.get("output", "").startswith("Error:") for item in executions)
    if not pending and not completed_todo:
        await worker.finish_group(group_id)
        return

    async def before_model(state, context):
        if context.get("injected"):
            await worker.finish_group(group_id)
            return {"jump_to": "end"}
        context["injected"] = True
        calls = [{"id": item["call_id"], "name": item["name"], "args": item["input"], "type": "tool_call"}
                 for item in executions]
        return {"messages": [AIMessage(content="", id=f"step:{group_id}", tool_calls=calls)], "jump_to": "tools"}

    async def model_call(state, context):
        raise RuntimeError("已保存的工具调用不能再次请求模型")

    completed = {item["call_id"]: item["output"] for item in executions if item["status"] == "completed"}
    await _invoke(worker, before_model, model_call, group_id=group_id, completed_calls=completed)


async def run(worker):
    from app.agent.worker import _PROMPT

    runtime = worker.runtime
    task = worker.task()
    if worker.helper_id:
        with db.new_session() as session:
            state = store.get(session, f"helper:{worker.helper_id}")
        worker.context_id, worker.run_id = state["context_id"], state["run_id"]
    else:
        worker.context_id, worker.run_id = task["context_id"], task["run_id"]
    if not await runtime.reconcile_incoming(worker.context_id):
        return
    worker.keep_sandbox = True

    async def before_model(state, context):
        if group_id := context.pop("group_id", None):
            await worker.finish_group(group_id)
        task = worker.task()
        if task.get("stop_requested") or worker.stop.is_set():
            await worker.finish("stopped", "任务已停止；已开始的操作可能已产生副作用。")
            return {"jump_to": "end"}
        if task.get("dependency") and not worker.helper_id:
            def pause(session):
                current = store.get(session, f"task:{worker.task_id}")
                current["status"] = "waiting_dependency"
                store.put(session, f"task:{worker.task_id}", current)
            await runtime.change(pause)
            return {"jump_to": "end"}
        if worker.helper_id:
            with db.new_session() as session:
                current = store.get(session, f"helper:{worker.helper_id}")
        else:
            current = task
        if current["steps"] >= 30:
            await worker.finish("limit", "本轮达到 30 步上限，请判断是否继续或停止。")
            return {"jump_to": "end"}
        if not await worker.deliver_messages():
            return {"jump_to": "end"}

        def step(session):
            key = f"helper:{worker.helper_id}" if worker.helper_id else f"task:{worker.task_id}"
            current = store.get(session, key)
            current["steps"] += 1
            store.put(session, key, current)
            return current["steps"]
        if not context.pop("reactive_retry", False):
            context["step_number"] = await runtime.change(step)

    def assemble(state, context):
        history = todo.remind(compact.history_for_state(state), state)
        if board := shared_state.prompt():
            history.append(board)
        system = _PROMPT + skills.skill_prompt() + "\n" + todo.PROMPT + "\n" + compact.state_instructions(state)
        return system, history, WORKER_TOOLS

    async def model_call(state, context):
        system, history, _ = assemble(state, context)
        deps = SimpleNamespace(client=runtime.client, recorder=SimpleNamespace(session_id="main", turn_id=worker.run_id))
        with record_model_step(deps, system, history, WORKER_TOOLS, subtask_id=worker.task_id) as reply:
            async for event in runtime.client.stream(system=system, messages=history, tools=WORKER_TOOLS):
                reply.accept(event)
        await runtime.run_tool(compact.record_usage, context, system, history, WORKER_TOOLS, reply.usage,
                               getattr(runtime.client, "model", ""),
                               separate_cache=type(runtime.client).__name__ == "AnthropicStreamClient")
        if reply.stop_reason == "max_tokens":
            await worker.finish("failed", "模型输出达到上限，本轮未完成。")
            return AIMessage(content="模型输出达到上限")
        blocks = reply.assistant_blocks()
        if not blocks:
            await worker.finish("failed", "模型没有返回内容，本轮未完成。")
            return AIMessage(content="模型没有返回内容")
        if not reply.tool_uses:
            step_id = f"step:{worker.task_id}:{worker.run_id}:{context['step_number']}"
            await runtime.change(lambda session: store.stage_context(session, worker.context_id, "assistant", blocks,
                                                                       key=step_id))
            with db.new_session() as session:
                incoming = store.records(session, "agent_message", context_id=worker.context_id, delivery="pending")
            if incoming or worker.stop.is_set() or worker.task().get("dependency"):
                context["retry"] = True
            else:
                injection = hooks.trigger("Stop", {"turn_id": worker.run_id, "task_id": worker.task_id,
                                                   "steps": context["step_number"]})
                if isinstance(injection, str) and injection:
                    await runtime.change(lambda session: store.stage_context(session, worker.context_id, "user", injection))
                    context["retry"] = True
                else:
                    await worker.finish("result", reply.text or "助手未提供结果。")
            return AIMessage(content=reply.text, id=step_id)

        group_id = f"{worker.task_id}:{worker.run_id}:{context['step_number']}"
        ids = [call.get("id") for call in reply.tool_uses]
        if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
            await worker.finish("failed", "模型返回了无效或重复的工具调用 ID。")
            return AIMessage(content="模型返回了无效的工具调用")

        def save_group(session):
            store.stage_context(session, worker.context_id, "assistant", blocks, key=f"step:{group_id}")
            calls = []
            for call in reply.tool_uses:
                identity = f"{worker.task_id}:{worker.run_id}:{call['id']}"
                existing = store.get(session, f"execution:{identity}")
                if existing and (existing["name"] != call["name"] or existing["input"] != (call.get("input") or {})):
                    raise ValueError("重复工具调用 ID 的参数不一致")
                if not existing:
                    task = store.get(session, f"task:{worker.task_id}")
                    store.put(session, f"execution:{identity}", {
                        "type": "tool_execution", "id": identity, "task_id": worker.task_id,
                        "context_id": worker.context_id, "run_id": worker.run_id, "call_id": call["id"],
                        "name": call["name"], "input": call.get("input") or {}, "status": "pending",
                        "user_seq": task.get("latest_user_seq", task["order"])})
                calls.append(identity)
            store.put(session, f"group:{group_id}", {"type": "tool_group", "id": group_id,
                      "context_id": worker.context_id, "calls": calls, "status": "pending"})
        await runtime.change(save_group)
        context["group_id"] = group_id
        return AIMessage(content=reply.text, id=f"step:{group_id}", tool_calls=[{"id": call["id"], "name": call["name"],
                         "args": call.get("input") or {}, "type": "tool_call"} for call in reply.tool_uses])

    async def after_model(state, context):
        if context.pop("retry", False):
            return {"jump_to": "model"}

    from app.agent.worker import AgentWaiting

    try:
        await worker.resume_groups()
        await _invoke(worker, before_model, model_call, after_model, assemble=assemble)
    except AgentWaiting:
        raise
    except BaseException:
        await runtime.release_worker_sandbox(worker.context_id, worker.run_id)
        raise
    with db.new_session() as session:
        state = (store.get(session, f"helper:{worker.helper_id}") if worker.helper_id
                 else store.get(session, f"task:{worker.task_id}"))
    if (not state or state.get("run_id") != worker.run_id
            or state["status"] not in {"waiting_agent", "waiting_dependency", "queued"}):
        try:
            await runtime.release_worker_sandbox(worker.context_id, worker.run_id)
        except Exception:
            pass
