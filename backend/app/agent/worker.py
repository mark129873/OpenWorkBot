"""独立任务循环。每次模型步骤和工具状态在执行前持久化。"""

import asyncio
import hashlib
import json
import os
import time
from types import SimpleNamespace

from app import db
from app.agent import agent_bus, compact, hooks, tools
from app.agent.model_call import record_model_step
from app.agent.runtime import read_output
from app.agent.task_tools import WORKER_TOOLS
from app.config import settings
from app.models import new_id
from app.sessions import approvals, chat_store as store, session_permissions

_PROMPT = """你是后台工作助手，专注完成给定任务。独立保存工作上下文；先读后改，工具执行失败不能当作成功。
必须遵守用户明确限定的操作和顺序；禁止的操作即使只读、用于定位或检查也不能执行。用户指定具体命令且禁止其他命令时，不得追加或替换命令；不确定时询问主助手。
关键进展或需要用户确认的问题使用 task_report 告诉主助手。完成后简短汇报结果、验证及未解决的问题。
收到用户修改、停止或同伴消息时按最新要求调整；不要冒充主助手或其他助手。"""


class AgentWaiting(Exception):
    pass


async def run_worker(runtime, task_id: str, helper_id: str | None = None) -> None:
    worker = Worker(runtime, task_id, helper_id)
    try:
        await worker.run()
    except AgentWaiting:
        return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await worker.finish("failed", f"执行异常：{exc}")


class Worker:
    def __init__(self, runtime, task_id: str, helper_id: str | None = None) -> None:
        self.runtime = runtime
        self.task_id = task_id
        self.helper_id = helper_id
        self.stop = runtime.stops.setdefault(task_id, asyncio.Event())
        self.context_id = ""
        self.run_id = ""

    def task(self) -> dict:
        with db.new_session() as session:
            task = store.get(session, f"task:{self.task_id}")
        if not task:
            raise ValueError("任务不存在")
        return task

    async def run(self) -> None:
        task = self.task()
        if self.helper_id:
            with db.new_session() as session:
                state = store.get(session, f"helper:{self.helper_id}")
            self.context_id, self.run_id = state["context_id"], state["run_id"]
        else:
            self.context_id, self.run_id = task["context_id"], task["run_id"]
        await self.resume_groups()
        while True:
            task = self.task()
            if task.get("stop_requested") or self.stop.is_set():
                await self.finish("stopped", "任务已停止；已开始的操作可能已产生副作用。")
                return
            if task.get("dependency") and not self.helper_id:
                def pause(session):
                    current = store.get(session, f"task:{self.task_id}")
                    current["status"] = "waiting_dependency"
                    store.put(session, f"task:{self.task_id}", current)
                await self.runtime.change(pause)
                return
            if self.helper_id:
                with db.new_session() as session:
                    state = store.get(session, f"helper:{self.helper_id}")
            else:
                state = task
            if state["steps"] >= 30:
                await self.finish("limit", "本轮达到 30 步上限，请判断是否继续或停止。")
                return
            await self.deliver_messages()
            def step(session):
                key = f"helper:{self.helper_id}" if self.helper_id else f"task:{self.task_id}"
                current = store.get(session, key)
                current["steps"] += 1
                store.put(session, key, current)
                return current["steps"]
            step_number = await self.runtime.change(step)
            history = await compact.context_history(self.runtime, self.context_id)
            deps = SimpleNamespace(client=self.runtime.client,
                                   recorder=SimpleNamespace(session_id="main", turn_id=self.run_id))
            with record_model_step(deps, _PROMPT, history, WORKER_TOOLS, subtask_id=self.task_id) as reply:
                async for event in self.runtime.client.stream(system=_PROMPT, messages=history, tools=WORKER_TOOLS):
                    reply.accept(event)
            if reply.stop_reason == "max_tokens":
                await self.finish("failed", "模型输出达到上限，本轮未完成。")
                return
            blocks = reply.assistant_blocks()
            if not blocks:
                await self.finish("failed", "模型没有返回内容，本轮未完成。")
                return
            if not reply.tool_uses:
                await self.runtime.change(lambda session: store.append_history(session, self.context_id, "assistant", blocks))
                with db.new_session() as session:
                    incoming = store.records(session, "agent_message", context_id=self.context_id, delivery="pending")
                if incoming or self.stop.is_set() or self.task().get("dependency"):
                    continue
                injection = hooks.trigger("Stop", {"turn_id": self.run_id, "task_id": self.task_id, "steps": step_number})
                if isinstance(injection, str) and injection:
                    await self.runtime.change(lambda session: store.append_history(session, self.context_id, "user", injection))
                    continue
                await self.finish("result", reply.text or "助手未提供结果。")
                return
            group_id = f"{self.task_id}:{self.run_id}:{step_number}"
            ids = [call.get("id") for call in reply.tool_uses]
            if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
                await self.finish("failed", "模型返回了无效或重复的工具调用 ID。")
                return
            def save_group(session):
                store.append_history(session, self.context_id, "assistant", blocks, key=f"step:{group_id}")
                calls = []
                for call in reply.tool_uses:
                    identity = f"{self.task_id}:{self.run_id}:{call['id']}"
                    existing = store.get(session, f"execution:{identity}")
                    if existing and (existing["name"] != call["name"] or existing["input"] != (call.get("input") or {})):
                        raise ValueError("重复工具调用 ID 的参数不一致")
                    if not existing:
                        store.put(session, f"execution:{identity}", {
                            "type": "tool_execution", "id": identity, "task_id": self.task_id,
                            "context_id": self.context_id, "run_id": self.run_id, "call_id": call["id"],
                            "name": call["name"], "input": call.get("input") or {}, "status": "pending"})
                    calls.append(identity)
                store.put(session, f"group:{group_id}", {"type": "tool_group", "id": group_id,
                          "context_id": self.context_id, "calls": calls, "status": "pending"})
            await self.runtime.change(save_group)
            await self.process_group(group_id)

    async def deliver_messages(self) -> None:
        def deliver(session):
            for message in store.records(session, "agent_message", context_id=self.context_id):
                if message["delivery"] == "pending":
                    agent_bus.deliver(session, message)
                    agent_bus.processed(session, message)
        await self.runtime.change(deliver)

    async def resume_groups(self) -> None:
        with db.new_session() as session:
            groups = [g["id"] for g in store.records(session, "tool_group", context_id=self.context_id)
                      if g["status"] == "pending"]
        for group_id in groups:
            await self.process_group(group_id)

    async def process_group(self, group_id: str) -> None:
        with db.new_session() as session:
            group = store.get(session, f"group:{group_id}")
        for identity in group["calls"]:
            with db.new_session() as session:
                execution = store.get(session, f"execution:{identity}")
            if execution["status"] != "completed":
                started_at = time.monotonic()
                context = {"name": execution["name"], "tool_call_id": execution["call_id"], "task_id": self.task_id}
                hooks.trigger("PreToolUse", context)
                try:
                    output = await self.execute(execution)
                except AgentWaiting:
                    raise
                except Exception:
                    hooks.trigger("PostToolUse", {**context, "status": "error", "duration_ms": (time.monotonic() - started_at) * 1000})
                    raise
                hooks.trigger("PostToolUse", {**context, "status": "error" if output.startswith("Error:") else "completed",
                                               "duration_ms": (time.monotonic() - started_at) * 1000})
                def save_result(session):
                    current = store.get(session, f"execution:{identity}")
                    current.update(status="completed", output=output)
                    store.put(session, f"execution:{identity}", current)
                await self.runtime.change(save_result)
        def finish_group(session):
            current = store.get(session, f"group:{group_id}")
            outputs = []
            for identity in current["calls"]:
                execution = store.get(session, f"execution:{identity}")
                output = execution["output"]
                outputs.append({"type": "tool_result", "tool_use_id": execution["call_id"], "content": output,
                                **({"is_error": True} if output.startswith("Error:") else {})})
            store.append_history(session, self.context_id, "user", outputs, key=f"result:{group_id}")
            current["status"] = "completed"
            store.put(session, f"group:{group_id}", current)
        await self.runtime.change(finish_group)

    async def execute(self, execution: dict) -> str:
        if self.stop.is_set() or self.task().get("stop_requested"):
            return "Error: 用户已停止任务，工具未执行。"
        name, args = execution["name"], execution["input"]
        if not isinstance(args, dict):
            return "Error: 工具参数必须是对象"
        if name in {"task_report", "agent_list", "agent_send", "agent_wait", "agent_help", "read_tool_output"}:
            return await self.communicate(name, args, execution["id"])
        if name not in tools.TOOL_HANDLERS and name != "todo_write":
            return f"Error: 未知工具 {name}"
        if name in {"read_file", "write_file", "edit_file", "delete_file"}:
            return await self.file_tool(execution)
        error = await self.authorize(execution)
        if error:
            return error
        await self.started(execution)
        if name == "todo_write":
            from app.agent.todo import parse_items, render_panel
            from app.sessions.store import replace_todos
            items, error = parse_items(args.get("items"))
            if error:
                return error
            await self.runtime.change(lambda session: replace_todos(session, "main", items, commit=False))
            return render_panel(items)
        if name == "bash":
            try:
                output, failed = await self.runtime.run_tool(tools.capture_bash, **args)
            except TypeError as exc:
                return f"Error: 参数不匹配：{exc}"
            output_id = new_id()
            await self.runtime.change(lambda session: store.put(session, f"output:{output_id}", {
                "type": "tool_output", "id": output_id, "task_id": self.task_id, "content": output}))
            return f"{'Error: ' if failed else ''}[输出 ID: {output_id}]\n{tools.output_preview(output, limit=29900)}"
        return await self.runtime.run_tool(tools.execute_tool, name, args)

    async def authorize(self, execution: dict) -> str | None:
        name, args = execution["name"], execution["input"]
        deps = SimpleNamespace(recorder=SimpleNamespace(session_id="main"),
                               stop_flag=lambda: self.stop.is_set(),
                               approver=lambda tool, inputs, reason: self.approve(execution, reason))
        return await approvals.authorize_tool(deps, name, args, data_dir=settings.data_dir)

    async def started(self, execution: dict) -> None:
        def started(session):
            current = store.get(session, f"execution:{execution['id']}")
            current["status"] = "started"
            store.put(session, f"execution:{execution['id']}", current)
        await self.runtime.change(started)

    async def file_tool(self, execution: dict) -> str:
        name, args = execution["name"], execution["input"]
        try:
            path = os.path.normcase(str(tools.safe_path(args["path"])))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            return f"Error: 文件路径无效：{exc}"
        writing = name != "read_file"
        if writing:
            waited = await self.wait_for_file(path, execution["id"])
            if waited:
                return waited
        lock = self.runtime.file_locks.setdefault(path, asyncio.Lock())
        async with lock:
            if self.stop.is_set():
                return "Error: 用户已停止任务，工具未执行。"
            initial_version = await self.runtime.run_tool(file_version, path)
            if writing:
                current_version = initial_version
                version_key = f"version:{self.context_id}:{hashlib.sha256(path.encode()).hexdigest()}"
                with db.new_session() as session:
                    read_version = store.get(session, version_key)
                    owners = [task for task in store.records(session, "task_state")
                              if path in task.get("files", []) and task["id"] != self.task_id
                              and task["status"] not in {"completed", "failed", "stopped"}]
                if owners:
                    return "Error: 文件已被其他任务占用，请等待并重新读取。"
                if read_version and read_version["version"] != current_version:
                    return "Error: 文件读取后已被修改，请重新读取并重新发起操作。"
            error = await self.authorize(execution)
            if error:
                return error
            try:
                if os.path.normcase(str(tools.safe_path(args["path"]))) != path:
                    return "Error: 文件路径已变化，请重新发起操作。"
            except (ValueError, OSError):
                return "Error: 文件路径已变化，请重新发起操作。"
            if writing:
                def claim(session):
                    task = store.get(session, f"task:{self.task_id}")
                    if path not in task["files"]:
                        task["files"].append(path)
                        store.put(session, f"task:{self.task_id}", task)
                await self.runtime.change(claim)
                if read_version and read_version["version"] != await self.runtime.run_tool(file_version, path):
                    return "Error: 审批期间文件已被修改，请重新读取并重新发起操作。"
            await self.started(execution)
            result = await self.runtime.run_tool(tools.execute_tool, name, args)
            if not result.startswith("Error:"):
                version = await self.runtime.run_tool(file_version, path)
                if not writing and version != initial_version:
                    return "Error: 读取期间文件发生变化，请重新读取。"
                key = f"version:{self.context_id}:{hashlib.sha256(path.encode()).hexdigest()}"
                await self.runtime.change(lambda session: store.put(session, key, {
                    "type": "file_version", "context_id": self.context_id, "path": path, "version": version}))
            return result

    async def wait_for_file(self, path: str, identity: str) -> str | None:
        with db.new_session() as session:
            prior = store.get(session, f"execution:{identity}")
        waited = bool(prior and prior.get("waited_for_file"))
        registered = False
        wait_key = f"file-wait:{identity}"
        while True:
            task = self.task()
            if self.stop.is_set() or task.get("stop_requested"):
                if waited:
                    await self.clear_file_wait(wait_key, path)
                return "Error: 用户已停止任务，工具未执行。"
            if task.get("dependency"):
                if waited:
                    await self.clear_file_wait(wait_key, path)
                return "Error: 任务正在等待依赖，工具未执行。"
            with db.new_session() as session:
                owners = [t for t in store.records(session, "task_state") if path in t.get("files", [])
                          and t["id"] != self.task_id and t["status"] not in {"completed", "failed", "stopped"}]
                incoming = [m for m in store.records(session, "agent_message", context_id=self.context_id)
                            if m["delivery"] == "pending" and m["source"] == "main"]
            if incoming:
                if waited:
                    await self.clear_file_wait(wait_key, path)
                return "Error: 收到新要求，请先处理再重新发起文件操作。"
            if not owners:
                if waited:
                    await self.clear_file_wait(wait_key, path)
                    return "Error: 文件曾被其他任务占用，请重新读取并重新发起操作。"
                return None
            if not registered:
                owner = owners[0]
                def register(session):
                    execution = store.get(session, f"execution:{identity}")
                    if execution:
                        execution["waited_for_file"] = True
                        store.put(session, f"execution:{identity}", execution)
                    store.put(session, wait_key, {"type": "file_wait", "key": wait_key, "task_id": self.task_id,
                              "context_id": self.context_id, "path": path, "owner": owner["id"], "status": "waiting"})
                    cycles = agent_bus.find_cycles(agent_bus.wait_graph(session))
                    if cycles:
                        agent_bus.send(session, source="runtime", target="main", task_id=self.task_id,
                                       content=f"检测到任务与文件等待环，请判断：{cycles}",
                                       kind="cycle", message_id=f"cycle:{identity}")
                await self.runtime.change(register)
                waited = registered = True
            await asyncio.sleep(0.25)

    async def clear_file_wait(self, key: str, path: str) -> None:
        await self.runtime.change(lambda session: store.put(session, key, {
            "type": "file_wait", "task_id": self.task_id, "path": path, "status": "completed"}))

    async def approve(self, execution: dict, reason: str) -> dict:
        request_id = new_id()
        scope = session_permissions.scope_for(execution["name"], execution["input"])
        event = asyncio.Event()
        self.runtime.approval_events[request_id] = event
        def request(session):
            task = store.get(session, f"task:{self.task_id}")
            state_key = f"helper:{self.helper_id}" if self.helper_id else f"task:{self.task_id}"
            state = store.get(session, state_key)
            state["status"] = "waiting_user"
            store.put(session, state_key, state)
            store.put(session, f"approval:{request_id}", {"type": "approval", "request_id": request_id,
                      "task_id": self.task_id, "task_title": task["title"], "run_id": self.run_id,
                      "tool_call_id": execution["call_id"], "tool": execution["name"], "input": execution["input"],
                      "scope": scope, "reason": reason, "options": approvals.build_options(execution["name"], execution["input"]),
                      "status": "requested", "order": store.get(session, "meta")["revision"] + 1})
        try:
            await self.runtime.change(request, "approval")
        except BaseException:
            self.runtime.approval_events.pop(request_id, None)
            raise
        decision_wait = asyncio.create_task(event.wait())
        stop_wait = asyncio.create_task(self.stop.wait())
        try:
            await asyncio.wait([decision_wait, stop_wait], return_when=asyncio.FIRST_COMPLETED)
        finally:
            decision_wait.cancel()
            stop_wait.cancel()
            await asyncio.gather(decision_wait, stop_wait, return_exceptions=True)
            self.runtime.approval_events.pop(request_id, None)
        with db.new_session() as session:
            record = store.get(session, f"approval:{request_id}")
        allowed = bool(record and record["status"] == "resolved" and record["option_id"] != "deny" and not self.stop.is_set())
        if allowed:
            from app.agent.permission_service import evaluate
            allowed = (evaluate(settings.data_dir, execution["name"], execution["input"])["decision"] != "deny"
                       and scope == session_permissions.scope_for(execution["name"], execution["input"]))
        if allowed and record["option_id"] == "allowSession" and scope is not None:
            session_permissions.grant("main", scope)
        def resolved(session):
            state_key = f"helper:{self.helper_id}" if self.helper_id else f"task:{self.task_id}"
            state = store.get(session, state_key)
            if state["status"] == "waiting_user":
                state["status"] = "running"
                store.put(session, state_key, state)
            current = store.get(session, f"approval:{request_id}")
            if current and current["status"] == "requested":
                current["status"] = "expired"
                store.put(session, f"approval:{request_id}", current)
        await self.runtime.change(resolved, "approval")
        return {"approved": allowed, "denial_reason": (record or {}).get("feedback") or reason}

    async def communicate(self, name: str, args: dict, identity: str) -> str:
        if name == "read_tool_output":
            with db.new_session() as session:
                return read_output(session, args, task_id=self.task_id)
        if name == "task_report":
            content = args.get("content")
            if not isinstance(content, str) or not content.strip():
                return "Error: 报告内容不能为空"
            def report(session):
                task = store.get(session, f"task:{self.task_id}")
                agent_bus.send(session, source=self.context_id, target="main", task_id=self.task_id,
                               content=f"任务【{task['title']}】进展：{content}", kind="progress", message_id=identity,
                               run_id=self.run_id)
            await self.runtime.change(report)
            return "已向主助手报告"
        if name == "agent_list":
            with db.new_session() as session:
                agents = [{"context_id": t["context_id"], "task_id": t["id"], "title": t["title"],
                           "role": "负责人", "status": t["status"]} for t in store.records(session, "task_state")
                          if t["status"] not in {"completed", "failed", "stopped"}]
                agents += [{"context_id": h["context_id"], "task_id": h["task_id"], "role": "协助者", "status": h["status"]}
                           for h in store.records(session, "agent_run") if h["status"] not in {"completed", "failed", "stopped"}]
            return json.dumps(agents, ensure_ascii=False)
        if name == "agent_send":
            target, content = args.get("context_id"), args.get("content")
            with db.new_session() as session:
                agents = store.records(session, "agent_run")
                tasks = store.records(session, "task_state")
            valid = (any(t["context_id"] == target and t["status"] not in {"completed", "failed", "stopped"} for t in tasks)
                     or any(a["context_id"] == target and a["status"] not in {"completed", "failed", "stopped"} for a in agents))
            valid = valid and target != self.context_id
            if not valid or not isinstance(content, str) or not content.strip():
                return "Error: 目标助手或消息无效"
            await self.runtime.change(lambda session: agent_bus.send(session, source=self.context_id, target=target,
                                      task_id=self.task_id,
                                      content=f"[来自任务 {self.task_id} 的助手 {self.context_id}] {content}", message_id=identity,
                                      run_id=self.run_id))
            return "消息已受理，将在对方安全边界处理"
        if name == "agent_help":
            goal = args.get("goal")
            if not isinstance(goal, str) or not goal.strip():
                return "Error: 协助目标不能为空"
            def allocate(session):
                prior = store.get(session, f"help-action:{identity}")
                if prior:
                    return prior["context_id"]
                tasks = store.records(session, "task_state")
                helpers = store.records(session, "agent_run")
                if any(t["status"] == "queued" for t in tasks):
                    return None
                pending = sum(1 for h in helpers if h["status"] == "queued")
                active = sum(1 for w in self.runtime.workers.values() if not w.done())
                if active + pending >= 3:
                    return None
                context_id, run_id = new_id(), new_id()
                helper = {"type": "agent_run", "task_id": self.task_id, "context_id": context_id,
                          "run_id": run_id, "caller": self.context_id, "goal": goal,
                          "status": "queued", "steps": 0}
                store.put(session, f"helper:{context_id}", helper)
                store.put(session, f"help-action:{identity}", {"type": "help_action", "context_id": context_id})
                store.append_history(session, context_id, "user", f"协助任务：{goal}", key=f"help:{context_id}")
                return context_id
            helper_id = await self.runtime.change(allocate)
            return f"协助已受理，助手 ID：{helper_id}" if helper_id else "没有空闲助手，未分配协助"
        if name == "agent_wait":
            return await self.wait_for_agent(args.get("context_id"), identity)
        return f"Error: 未知通信工具 {name}"

    async def wait_for_agent(self, target: str, identity: str) -> str:
        key = f"agent-wait:{identity}"

        def check(session):
            previous = store.get(session, f"wait-result:{identity}")
            if previous:
                return previous["content"]
            tasks, agents = store.records(session, "task_state"), store.records(session, "agent_run")
            task = store.get(session, f"task:{self.task_id}")
            if not (any(t["context_id"] == target for t in tasks) or any(a["context_id"] == target for a in agents)) or target == self.context_id:
                return "Error: 目标助手无效"
            messages = store.records(session, "agent_message", context_id=self.context_id, delivery="pending")
            reply = next((message for message in messages if message["source"] == target), None)
            other = next((t for t in tasks if t["context_id"] == target), None)
            helper = next((a for a in agents if a["context_id"] == target), None)
            if self.stop.is_set() or task.get("stop_requested") or task.get("dependency"):
                result = "Error: 等待已被停止或依赖变更打断。"
            elif any(message["source"] == "main" for message in messages):
                result = "Error: 收到新要求，请先处理再继续等待。"
            elif reply:
                agent_bus.processed(session, reply)
                result = reply["content"]
            elif ((other and other["status"] in {"completed", "failed", "stopped"})
                  or (helper and helper["status"] in {"completed", "failed", "stopped"})):
                result = "Error: 对方助手已结束，且没有待处理回复。"
            else:
                result = None
            if result is not None:
                wait = store.get(session, key)
                if wait:
                    wait["status"] = "completed"
                    store.put(session, key, wait)
                store.put(session, f"wait-result:{identity}", {"type": "agent_wait_result",
                          "context_id": self.context_id, "content": result})
                return result
            target_task = other or next((t for t in tasks if helper and t["id"] == helper["task_id"]), None)
            if target_task and target_task["status"] not in {"completed", "failed", "stopped"}:
                target_task["order"] = min(target_task["order"], task["order"])
                store.put(session, f"task:{target_task['id']}", target_task)
            store.put(session, key, {"type": "agent_wait", "key": key, "context_id": self.context_id,
                      "target": target, "task_id": self.task_id, "status": "waiting"})
            state_key = f"helper:{self.helper_id}" if self.helper_id else f"task:{self.task_id}"
            state = store.get(session, state_key)
            state["status"] = "waiting_agent"
            store.put(session, state_key, state)
            cycles = agent_bus.find_cycles(agent_bus.wait_graph(session))
            if cycles:
                agent_bus.send(session, source="runtime", target="main", task_id=self.task_id,
                               content=f"检测到助手等待环，请判断：{cycles}", kind="cycle", message_id=f"cycle:{identity}")
            return None
        result = await self.runtime.change(check)
        if result is None:
            raise AgentWaiting
        return result

    async def finish(self, outcome: str, content: str) -> None:
        def save(session):
            task = store.get(session, f"task:{self.task_id}")
            if not task:
                return
            result = "stopped" if task.get("stop_requested") or self.stop.is_set() else outcome
            if self.helper_id:
                helper = store.get(session, f"helper:{self.helper_id}")
                helper["status"] = "completed" if result == "result" else "stopped" if result == "stopped" else "failed"
                store.put(session, f"helper:{self.helper_id}", helper)
                if task["status"] not in {"completed", "failed", "stopped"}:
                    agent_bus.send(session, source=self.context_id, target=helper["caller"], task_id=self.task_id,
                                   content=f"协助助手结果（{result}）：{content}", message_id=f"finish:{self.helper_id}:{self.run_id}",
                                   run_id=self.run_id)
                return
            if task["status"] in {"completed", "failed", "stopped"}:
                return
            if result == "stopped":
                task["status"] = "stopped"
                self.runtime._release_dependents(session, task)
            else:
                task["status"] = "reviewing"
            store.put(session, f"task:{self.task_id}", task)
            agent_bus.send(session, source=self.context_id, target="main", task_id=self.task_id,
                           content=f"任务【{task['title']}】一轮结束（{result}）：{content}",
                           kind="result" if result == "stopped" else "review",
                           message_id=f"finish:{self.task_id}:{self.run_id}", run_id=self.run_id)
        await self.runtime.change(save)


def file_version(path: str) -> str:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except FileNotFoundError:
        return "missing"
