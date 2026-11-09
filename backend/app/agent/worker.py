"""独立任务循环。每次模型步骤和工具状态在执行前持久化。"""

import asyncio
import hashlib
import json
import math
import os
import re
import time
from types import SimpleNamespace

from jsonschema import ValidationError, validate

from app import db
from app.agent import agent_bus, hooks, tools
from app.agent.rag_client import RAG_NAMES
from app.agent.runtime import read_output
from app.config import settings
from app.models import new_id
from app.sessions import approvals, chat_store as store, execution_state, session_permissions
from app.user_context import scoped

_PROMPT = """你是后台工作助手，专注完成给定任务。独立保存工作上下文；先读后改，工具执行失败不能当作成功。
知识库由所有用户共享。查询用 rag_search；仅在用户明确要求时调用 rag_ingest 或 rag_remove。
已指定文件时直接入库；移出前用 rag_list 查文件 ID。普通知识问答完整返回工具答案。
bash 仅在用户共享沙箱内运行，并按权限审批；宿主命令不可用。
文件工具使用容器内绝对路径，工作区为 /mnt/user-data/workspace；知识库入库仍使用工作区相对路径。
必须遵守用户限定的操作和顺序；不确定时询问主助手。
关键进展或需要用户确认的问题使用 task_report 告诉主助手。完成后简短汇报结果、验证及未解决的问题。
收到用户修改、停止或同伴消息时按最新要求调整；不要冒充主助手或其他助手。"""


def _limit_result(text: str) -> str:
    data = text.encode("utf-8")
    if len(data) <= 16 * 1024:
        return text
    marker = "\n[结果过长，已截断]"
    budget = 16 * 1024 - len(marker.encode("utf-8"))
    return data[:budget].decode("utf-8", errors="ignore") + marker


def _read_page(content: str, start: int, end: int | None) -> str:
    lines = content.split("\n") if content else []
    if lines and not lines[-1]:
        lines.pop()
    stop = min(end if end is not None else len(lines), start + 1999, len(lines))
    rows, size, next_line = [], 0, None
    for index in range(start - 1, stop):
        line = lines[index].rstrip("\r")
        row = f"{index + 1}: {line[:2000]}"
        if len(line) > 2000:
            row += " …（该行超过 2000 字符，余下内容已截断）"
        if size + len(row) + 1 > 99900:
            next_line = index + 1
            break
        rows.append(row)
        size += len(row) + 1
    if next_line is None and stop < len(lines) and (end is None or stop < end):
        next_line = stop + 1
    if next_line is not None:
        rows.append(f"[继续读取：start_line={next_line}]")
    return "\n".join(rows) if rows else "(文件为空或已到文件末尾)"


def _read_archive_page(content: str, offset: int, limit: int) -> str:
    result = content[offset:offset + limit]
    if result.startswith("Error:"):
        result = "[工具结果正文]\n" + result
    if offset + limit < len(content):
        result += f"\n[继续读取：offset={offset + limit}]"
    return result or "(文件为空或已到文件末尾)"


def _replace_content(content: str, old: str, new: str, replace_all: bool) -> str | None:
    if not old:
        return content
    if "\r\n" in content:
        crlf_old = old.replace("\r\n", "\n").replace("\n", "\r\n")
        if content.count("\r\n") == content.count("\n") or (old not in content and crlf_old in content):
            old = crlf_old
            new = new.replace("\r\n", "\n").replace("\n", "\r\n")
    return content.replace(old, new, -1 if replace_all else 1) if old in content else None


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
        self.sandbox = None
        self.sandbox_scope = ""
        self.keep_sandbox = False

    def task(self) -> dict:
        with db.new_session() as session:
            task = store.get(session, f"task:{self.task_id}")
        if not task:
            raise ValueError("任务不存在")
        return task

    async def run(self) -> None:
        from app.agent.worker_graph import run

        await run(self)

    async def deliver_messages(self) -> bool:
        with db.new_session() as session:
            messages = store.records(session, "agent_message", context_id=self.context_id)
        for message in messages:
            if message["delivery"] in {"pending", "applying", "unknown"}:
                if not await self.runtime.deliver_message(message):
                    return False

        def deliver(session):
            for message in store.records(session, "agent_message", context_id=self.context_id):
                if message["delivery"] == "delivered":
                    agent_bus.processed(session, message)
        await self.runtime.change(deliver)
        return True

    async def resume_groups(self) -> None:
        with db.new_session() as session:
            groups = [g["id"] for g in store.records(session, "tool_group", context_id=self.context_id)
                      if g["status"] == "pending"]
        for group_id in groups:
            await self.process_group(group_id)

    async def process_group(self, group_id: str) -> None:
        from app.agent.worker_graph import process_group

        await process_group(self, group_id)

    async def process_execution(self, identity: str) -> str:
        with db.new_session() as session:
            execution = store.get(session, f"execution:{identity}")
        if execution["status"] == "completed":
            return execution["output"]
        if execution["status"] == "started":
            output = "Error: 工具结果未知，可能已产生副作用；请先检查，不要自动重跑。"
            def unknown(session):
                current = store.get(session, f"execution:{identity}")
                current.update(status="completed", output=output, result_unknown=True)
                store.put(session, f"execution:{identity}", current)
            await self.runtime.change(unknown)
            return output
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
        rag_answer = None
        result_unknown = output.startswith("Error:") and ("结果未知" in output or "结果可能未知" in output)
        if execution["name"] == "rag_search":
            try:
                result = json.loads(output)
                if result.get("status") in {"success", "insufficient"} and isinstance(result.get("answer"), str):
                    rag_answer = _limit_result(result["answer"])
            except (TypeError, ValueError, AttributeError):
                pass
        from app.agent.tool_output import archive
        listing_path = execution["input"].get("path")
        archive_root = "/mnt/user-data/workspace/.tool-results"
        listing = (execution["name"] == "ls" and isinstance(listing_path, str)
                   and (listing_path.rstrip("/") == archive_root or listing_path.startswith(archive_root + "/")))
        if not listing:
            output = await archive(self.runtime, self.task_id, execution["name"], identity, output)
        hooks.trigger("PostToolUse", {**context, "status": "error" if output.startswith("Error:") else "completed",
                                       "duration_ms": (time.monotonic() - started_at) * 1000})
        def save_result(session):
            current = store.get(session, f"execution:{identity}")
            current.update(status="completed", output=output)
            if rag_answer is not None:
                current["rag_answer"] = rag_answer
            if result_unknown:
                current["result_unknown"] = True
            store.put(session, f"execution:{identity}", current)
        await self.runtime.change(save_result)
        return output

    async def finish_group(self, group_id: str) -> None:
        def finish_group(session):
            current = store.get(session, f"group:{group_id}")
            outputs = []
            for identity in current["calls"]:
                execution = store.get(session, f"execution:{identity}")
                output = execution["output"]
                outputs.append({"type": "tool_result", "tool_use_id": execution["call_id"], "content": output,
                                **({"is_error": True} if output.startswith("Error:") else {})})
                if execution["name"] == "agent_wait":
                    wait = store.get(session, f"agent-wait:{identity}")
                    if wait:
                        wait["status"] = "completed"
                        store.put(session, f"agent-wait:{identity}", wait)
            store.stage_context(session, self.context_id, "user", outputs, key=f"result:{group_id}")
            current["status"] = "completed"
            store.put(session, f"group:{group_id}", current)
        await self.runtime.change(finish_group)

    async def execute(self, execution: dict) -> str:
        if self.stop.is_set() or self.task().get("stop_requested"):
            return "Error: 用户已停止任务，工具未执行。"
        name, args = execution["name"], execution["input"]
        if not isinstance(args, dict):
            return "Error: 工具参数必须是对象"
        if name in {"read_file", "write_file", "str_replace", "glob", "ls", "grep"}:
            schema = next(spec["input_schema"] for spec in tools.TOOLS if spec["name"] == name)
            try:
                validate(args, {**schema, "additionalProperties": False})
            except ValidationError:
                return "Error: 文件工具参数无效。"
        from app.agent import coordination
        from app.agent.task_tools import MAIN_TOOLS, SHARED_TOOLS

        if self.task().get("direct") and name in {tool["name"] for tool in MAIN_TOOLS + SHARED_TOOLS}:
            source_seq = (execution.get("decision_seq") if name in {"task_update", "task_stop", "write_todos"}
                          else execution.get("user_seq"))
            return await self.runtime.change(lambda session: self.runtime._main_tool(
                session, {"id": execution["id"], "name": name, "input": args}, source_seq))

        if name in coordination.NAMES:
            try:
                return await self.runtime.change(lambda session: coordination.call(
                    session, name, args, actor=self.context_id, task_id=self.task_id, request_id=execution["id"]))
            except (KeyError, TypeError, ValueError) as exc:
                return f"Error: {exc}"
        if name in {"task_report", "agent_list", "agent_send", "agent_wait", "agent_help", "read_tool_output"}:
            return await self.communicate(name, args, execution["id"])
        if name in RAG_NAMES:
            return await self.rag_tool(execution)
        if name == "bash":
            return await self.bash_tool(execution)
        if name in {"read_file", "write_file", "str_replace"}:
            return await self.file_tool(execution)
        if name in {"ls", "glob", "grep"}:
            return await self.search_tool(execution)
        if name == "write_todos":
            schema = next(spec["input_schema"] for spec in tools.TOOLS if spec["name"] == name)
            try:
                validate(args, {**schema, "additionalProperties": False})
            except ValidationError:
                return "Error: TODO 参数无效。"
            return "已更新 TODO。"
        if name not in tools.TOOL_HANDLERS:
            return f"Error: 未知工具 {name}"
        error = await self.authorize(execution)
        if error:
            return error
        error = await self.started(execution)
        if error:
            return error
        return await self.runtime.run_tool(tools.execute_tool, name, args)

    async def bash_tool(self, execution: dict) -> str:
        from app.agent.permission_service import evaluate

        args = execution["input"]
        command = args.get("command")
        if (set(args) - {"command", "description"} or not isinstance(command, str) or not command.strip()
                or ("description" in args and not isinstance(args["description"], str))):
            return "Error: 命令参数无效。"
        if self.sandbox is None or not self.sandbox_scope:
            return "Error: 沙箱不可用，禁止宿主命令。"
        if (type(settings.bash_timeout) not in {int, float} or not math.isfinite(settings.bash_timeout)
                or settings.bash_timeout <= 0):
            return "Error: 命令超时配置无效。"
        before = execution_state.load_execution_state(settings.data_dir)["mode"]
        error = await self.authorize(execution, sandboxed=True)
        if error:
            return error
        current = execution_state.load_execution_state(settings.data_dir)["mode"]
        if evaluate(settings.data_dir, "bash", args, sandboxed=True)["decision"] == "deny":
            return "Error: 命令已被安全策略拒绝。"
        if before == "yolo" and current != "yolo":
            return "Error: 权限模式已变化，请重新发起命令并审批。"
        error = await self.started(execution, execution.get("user_seq"))
        if error:
            return error
        command_task = asyncio.create_task(self.runtime.run_tool(
            self.sandbox.execute_command_in_scope, command, timeout=settings.bash_timeout, scope_id=self.sandbox_scope))
        stop_task = asyncio.create_task(self.stop.wait())
        try:
            await asyncio.wait([command_task, stop_task], return_when=asyncio.FIRST_COMPLETED)
            if self.stop.is_set():
                self.sandbox.request_command_cancel(self.sandbox_scope)
            return await asyncio.shield(command_task)
        except asyncio.CancelledError:
            self.sandbox.request_command_cancel(self.sandbox_scope)
            await asyncio.shield(command_task)
            raise
        finally:
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)

    async def rag_tool(self, execution: dict) -> str:
        from app.agent.permission_service import evaluate

        name, args = execution["name"], execution["input"]
        fields = {"rag_search": {"query"}, "rag_list": set(), "rag_ingest": {"path"}, "rag_remove": {"document_id"}}
        if set(args) != fields[name] or any(not isinstance(value, str) or not value.strip() for value in args.values()):
            return "Error: 知识库工具参数无效。"
        if self.runtime.rag is None:
            return "Error: 知识库服务不可用。"
        writing = name in {"rag_ingest", "rag_remove"}
        if writing and not self.knowledge_authorized(name):
            return "Error: 需要用户明确指定知识库入库或移出操作。"
        with db.new_session() as session:
            task = store.get(session, f"task:{self.task_id}")
            pending = [item for item in store.records(session, "agent_message", context_id=self.context_id)
                       if item.get("source") == "main" and item["delivery"] in {"pending", "applying", "unknown", "delivered"}]
            previous = [item for item in store.records(session, "tool_execution") if item.get("id") != execution["id"]
                        and item.get("name") == name and item.get("input") == args and item.get("result_unknown")]
        if pending:
            return "Error: 收到新的用户要求，请先处理后再调用知识库。"
        if writing and previous and not self.knowledge_authorized(name, after=max(
                item.get("user_seq", task["order"]) for item in previous)):
            return "Error: 此前知识库操作结果未知，请先检查并等待用户明确要求重试。"
        initial = evaluate(settings.data_dir, name, args)["decision"]
        error = await self.authorize(execution)
        if error:
            return error
        sequence = task.get("latest_user_seq", task["order"])

        async def before_send():
            if self.stop.is_set():
                return "Error: 用户已停止任务，工具未执行。"
            current = evaluate(settings.data_dir, name, args)["decision"]
            if current == "deny" or (initial == "allow" and current == "ask"
                                      and not session_permissions.allows(scoped("main"), name, args)):
                return "Error: 知识库操作权限已变化，请重新确认。"
            return await self.started(execution, sequence)

        operation_id = hashlib.sha256(f"{self.runtime.user_id}:{execution['id']}".encode()).hexdigest()
        sent = asyncio.Event()
        call = asyncio.create_task(self.runtime.rag.perform(self.runtime, name, args, operation_id, before_send, sent))
        if writing:
            self.runtime.rag_writes[call] = sent
        stopped = asyncio.create_task(self.stop.wait())
        try:
            await asyncio.wait([call, stopped], return_when=asyncio.FIRST_COMPLETED)
            if self.stop.is_set() and not (writing and sent.is_set()):
                if sent.is_set():
                    await self.runtime.rag.cancel(operation_id)
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
                return "Error: 操作已停止。"
            try:
                result = await asyncio.shield(call)
            except Exception:
                if writing and sent.is_set():
                    result = "Error: 知识库操作结果未知，请先检查。"
                else:
                    result = "Error: 知识库操作失败，请检查后重试。"
            if self.stop.is_set() and writing and sent.is_set():
                def report(session):
                    if not store.get(session, "meta").get("reset"):
                        agent_bus.send(session, source=self.context_id, target="main", task_id=self.task_id,
                                       content=_limit_result(f"停止前已提交的知识库操作结果：{result}"), kind="progress",
                                       message_id=f"rag-stop:{execution['id']}", run_id=self.run_id)
                await self.runtime.change(report)
            if name == "rag_search" and not result.startswith("Error:"):
                try:
                    answer = json.loads(result)
                    if (not isinstance(answer, dict) or answer.get("status") not in {"success", "insufficient"}
                            or not isinstance(answer.get("answer"), str)):
                        return "Error: 知识库返回格式无效。"
                    result = json.dumps(answer, ensure_ascii=False)
                except (TypeError, ValueError):
                    return "Error: 知识库返回格式无效。"
            return result
        except asyncio.CancelledError:
            if writing and sent.is_set():
                try:
                    result = await asyncio.shield(call)
                except Exception:
                    result = "Error: 知识库操作结果未知，请先检查。"
                def save(session):
                    current = store.get(session, f"execution:{execution['id']}")
                    if current and current["status"] != "completed":
                        current.update(status="completed", output=result)
                        if "结果未知" in result or "结果可能未知" in result:
                            current["result_unknown"] = True
                        store.put(session, f"execution:{execution['id']}", current)
                await self.runtime.change(save)
            else:
                if sent.is_set():
                    await asyncio.shield(self.runtime.rag.cancel(operation_id))
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
                if writing:
                    def save_unsent(session):
                        current = store.get(session, f"execution:{execution['id']}")
                        if current and current["status"] == "started":
                            current.update(status="completed", output="Error: 操作已停止，工具未执行。")
                            store.put(session, f"execution:{execution['id']}", current)
                    await self.runtime.change(save_unsent)
            raise
        finally:
            self.runtime.rag_writes.pop(call, None)
            stopped.cancel()
            await asyncio.gather(stopped, return_exceptions=True)

    def knowledge_authorized(self, name: str, after: int = -1) -> bool:
        # shortcut: 只识别常见明确指令，误拒频繁时扩展表达。
        task = self.task()
        with db.new_session() as session:
            sequences = {task["order"]} | {item.get("user_seq") for item in store.records(session, "agent_message")
                                          if item.get("task_id") == self.task_id and item.get("source") == "main"}
            messages = sorted((item for item in store.records(session, "user_event")
                               if item.get("event_type") == "message" and item.get("user_seq") in sequences
                               and item["user_seq"] > after), key=lambda item: item["user_seq"], reverse=True)
        actions = {
            "rag_ingest": r"入库|导入|加入|加进|添加|收录|存入|写入|放入|放进|放到|上传到|上传至|更新索引|重建索引|\b(?:ingest|import|add|put|store|reindex|update)\b",
            "rag_remove": r"移出|移除|删除|去除|去掉|剔除|\b(?:remove|delete|drop)\b",
            "rag_search": r"查询|检索|搜索|查找|\b(?:search|query|find)\b",
            "rag_list": r"列表|列出|查看|有哪些|\b(?:list|show)\b",
        }[name]
        for item in messages:
            text = item.get("user_text", item.get("text", "")).casefold()
            action = re.search(actions, text)
            scope = re.search(r"知识库|入库|索引|knowledge\s*base|\brag\b|\bindex\b", text)
            retry = after >= 0 and re.search(r"重试|再试|重新执行|\bretry\b|\btry again\b", text)
            if (action and scope) or retry:
                return not bool(re.search(r"(?:不要|禁止|不许|别|无需|不必|do not|don't|never).{0,30}(?:" + actions + r"|重试|再试|retry)", text))
        return False

    async def authorize(self, execution: dict, *, sandboxed: bool = False,
                        canonical_path: str | None = None) -> str | None:
        name, args = execution["name"], execution["input"]
        deps = SimpleNamespace(recorder=SimpleNamespace(session_id=scoped("main")),
                               stop_flag=lambda: self.stop.is_set(),
                               approver=lambda tool, inputs, reason: self.approve(
                                   execution, reason, sandboxed=sandboxed, canonical_path=canonical_path))
        return await approvals.authorize_tool(deps, name, args, data_dir=settings.data_dir, sandboxed=sandboxed,
                                              canonical_path=canonical_path)

    async def started(self, execution: dict, expected_user_seq: int | None = None) -> str | None:
        def started(session):
            current = store.get(session, f"execution:{execution['id']}")
            task = store.get(session, f"task:{self.task_id}")
            if task.get("stop_requested"):
                return "Error: 用户已停止任务，工具未执行。"
            if (expected_user_seq is not None and
                    task.get("latest_user_seq", task["order"]) != expected_user_seq):
                return "Error: 收到新的用户要求，请先判断后再执行。"
            if expected_user_seq is not None and any(
                    item.get("event_type") == "message" and item.get("status") == "pending"
                    and item["user_seq"] > expected_user_seq for item in store.records(session, "user_event")):
                return "Error: 收到新的用户消息，请先判断后再执行。"
            if expected_user_seq is not None and any(
                    item.get("source") == "main" and item["delivery"] in {"pending", "applying", "unknown", "delivered"}
                    for item in store.records(session, "agent_message", context_id=self.context_id)):
                return "Error: 收到新的用户要求，请先判断后再执行。"
            if task.get("direct"):
                decisions = [a.get("user_seq", 0) for a in store.records(session, "approval")
                             if a.get("task_id") == self.task_id and a.get("tool_call_id") == execution["call_id"]
                             and a.get("status") == "resolved"]
                latest = max(decisions, default=0)
                if any(item.get("user_seq", 0) > latest for item in store.records(
                        session, "agent_message", context_id=self.context_id, delivery="pending")
                       if item["source"] == "main"):
                    return "Error: 收到新的用户要求，请先判断后再执行。"
            current["status"] = "started"
            store.put(session, f"execution:{execution['id']}", current)
            return None
        return await self.runtime.change(started)

    async def file_tool(self, execution: dict) -> str:
        name, args = execution["name"], execution["input"]
        if self.sandbox is None:
            return "Error: 沙箱不可用，文件操作未执行。"
        requested_path = args.get("path")
        if not isinstance(requested_path, str) or not requested_path.startswith("/") or "\0" in requested_path:
            return "Error: 路径必须是容器内绝对路径。"
        if name == "read_file" and args.get("end_line") is not None and args.get("start_line", 1) > args["end_line"]:
            return "Error: 读取行范围无效。"
        if name == "read_file" and ({"offset", "limit"} & args.keys()) and ({"start_line", "end_line"} & args.keys()):
            return "Error: 字符分页不能与行分页同时使用。"
        if name == "write_file" and not args.get("append", False) and len(args["content"].encode("utf-8")) > 80 * 1024:
            return "Error: 单次写入不能超过 80 KiB，请分段追加。"
        from app.agent.permission_service import evaluate
        if evaluate(settings.data_dir, name, args, sandboxed=True)["decision"] == "deny":
            return "Error: 已被安全策略拒绝。"
        try:
            path = await self.runtime.run_tool(self.sandbox.real_path, requested_path)
        except Exception as exc:
            return f"Error: 文件路径无效：{exc}"
        archive_root = "/mnt/user-data/workspace/.tool-results/"
        archive_file = requested_path.startswith(archive_root) and path.startswith(archive_root)
        if execution.get("archive_only") and not archive_file:
            return "Error: 当前只能读取工具结果归档。"
        if name == "read_file" and ({"offset", "limit"} & args.keys()) and not archive_file:
            return "Error: 字符分页仅用于工具结果归档。"
        writing = name != "read_file"
        if writing:
            waited = await self.wait_for_file(path, execution["id"])
            if waited:
                return waited
        lock = self.runtime.file_locks.setdefault(path, asyncio.Lock())
        async with lock:
            if self.stop.is_set():
                return "Error: 用户已停止任务，工具未执行。"
            error = await self.authorize(execution, sandboxed=True, canonical_path=path)
            if error:
                return error
            try:
                if await self.runtime.run_tool(self.sandbox.real_path, requested_path) != path:
                    return "Error: 文件路径已变化，请重新发起操作。"
            except Exception:
                return "Error: 文件路径已变化，请重新发起操作。"
            content = await self.runtime.run_tool(self.sandbox.read_file, requested_path)
            uninspectable = content.startswith("Error:")
            if uninspectable and archive_file and not writing:
                from app.agent.tool_output import matches_file
                if await self.runtime.run_tool(matches_file, requested_path, content):
                    uninspectable = False
            if not writing and uninspectable:
                return content
            initial_version = None if uninspectable else hashlib.sha256(content.encode("utf-8")).hexdigest()
            version_key = f"version:{self.context_id}:{hashlib.sha256(path.encode()).hexdigest()}"
            if writing:
                with db.new_session() as session:
                    read_version = store.get(session, version_key)
                    owners = [task for task in store.records(session, "task_state")
                              if path in task.get("files", []) and task["id"] != self.task_id
                              and task["status"] not in {"completed", "failed", "stopped"}]
                if owners:
                    return "Error: 文件已被其他任务占用，请等待并重新读取。"
                # shortcut: DeerFlow 对读取失败放行，若上游能可靠区分不存在再收紧。
                if not uninspectable and (not read_version or read_version["version"] != initial_version):
                    return "Error: 请先读取文件的当前版本，再重新发起操作。"
            if writing:
                def claim(session):
                    task = store.get(session, f"task:{self.task_id}")
                    if path not in task["files"]:
                        task["files"].append(path)
                        store.put(session, f"task:{self.task_id}", task)
                await self.runtime.change(claim)
                latest = await self.runtime.run_tool(self.sandbox.read_file, requested_path)
                if (not uninspectable and latest != content) or (uninspectable and not latest.startswith("Error:")):
                    return "Error: 审批期间文件已被修改，请重新读取并重新发起操作。"
            error = await self.started(execution)
            if error:
                return error
            if not writing:
                latest = await self.runtime.run_tool(self.sandbox.read_file, requested_path)
                if latest != content:
                    return "Error: 读取期间文件发生变化，请重新读取。"
                await self.runtime.change(lambda session: store.put(session, version_key, {
                    "type": "file_version", "context_id": self.context_id, "path": path, "version": initial_version}))
                if archive_file and not ({"start_line", "end_line"} & args.keys()):
                    return _read_archive_page(content, args.get("offset", 0), args.get("limit", 10000))
                return _read_page(content, args.get("start_line", 1), args.get("end_line"))
            try:
                if name == "write_file":
                    await self.runtime.run_tool(self.sandbox.write_file, requested_path, args["content"],
                                                append=args.get("append", False))
                else:
                    current = await self.runtime.run_tool(self.sandbox.read_file, requested_path)
                    if current.startswith("Error:"):
                        return current
                    replacement = _replace_content(current, args["old_str"], args["new_str"], args.get("replace_all", False))
                    if replacement is None:
                        return "Error: 文件中没有找到要替换的内容。"
                    if args["old_str"]:
                        await self.runtime.run_tool(self.sandbox.write_file, requested_path, replacement)
            except Exception as exc:
                return f"Error: 写入失败或结果未知，请先核查：{type(exc).__name__}"
            await self.runtime.change(lambda session: store.put(session, version_key, {
                "type": "file_version", "context_id": self.context_id, "path": path, "version": None}))
            return "OK"

    async def search_tool(self, execution: dict) -> str:
        name, args = execution["name"], execution["input"]
        if self.sandbox is None:
            return "Error: 沙箱不可用，文件操作未执行。"
        path = args.get("path")
        if not isinstance(path, str) or not path.startswith("/") or "\0" in path:
            return "Error: 路径必须是容器内绝对路径。"
        try:
            canonical = await self.runtime.run_tool(self.sandbox.real_path, path)
            archive_root = "/mnt/user-data/workspace/.tool-results"
            archive_dir = (path.rstrip("/") == archive_root or path.startswith(archive_root + "/")) and (
                canonical == archive_root or canonical.startswith(archive_root + "/"))
            if execution.get("archive_only") and not archive_dir:
                return "Error: 当前只能列出工具结果归档。"
            if ({"offset", "limit"} & args.keys()) and (name != "ls" or not archive_dir):
                return "Error: 目录分页仅用于工具结果归档。"
            error = await self.authorize(execution, sandboxed=True, canonical_path=canonical)
            if error:
                return error
            if await self.runtime.run_tool(self.sandbox.real_path, path) != canonical:
                return "Error: 文件路径已变化，请重新发起操作。"
            error = await self.started(execution)
            if error:
                return error
            if name == "ls":
                if archive_dir:
                    from app.agent.tool_output import list_directory
                    return await self.runtime.run_tool(list_directory, canonical, args.get("offset", 0), args.get("limit", 50))
                entries = await self.runtime.run_tool(self.sandbox.list_dir, path)
                return "\n".join(entries) or "(空目录)"
            if name == "glob":
                matches, truncated = await self.runtime.run_tool(
                    self.sandbox.glob, path, args["pattern"], include_dirs=args.get("include_dirs", False),
                    max_results=args.get("max_results", 200))
                rows = matches
            else:
                matches, truncated = await self.runtime.run_tool(
                    self.sandbox.grep, path, args["pattern"], glob=args.get("glob"), literal=args.get("literal", False),
                    case_sensitive=args.get("case_sensitive", False), max_results=args.get("max_results", 100))
                rows = [f"{match.path}:{match.line_number}: {match.line}" for match in matches]
            if truncated:
                rows.append("[结果已截断]")
            return "\n".join(rows) or "(无匹配结果)"
        except Exception as exc:
            return f"Error: 文件搜索失败：{exc}"

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

    async def approve(self, execution: dict, reason: str, *, sandboxed: bool = False,
                      canonical_path: str | None = None) -> dict:
        request_id = new_id()
        scope = session_permissions.scope_for(execution["name"], execution["input"], canonical_path=canonical_path)
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
                      "scope": scope, "reason": reason, "options": approvals.build_options(
                          execution["name"], execution["input"], canonical_path=canonical_path),
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
            if canonical_path is not None:
                try:
                    canonical_path = await self.runtime.run_tool(self.sandbox.real_path, execution["input"]["path"])
                except Exception:
                    canonical_path = None
            allowed = (evaluate(settings.data_dir, execution["name"], execution["input"], sandboxed=sandboxed)["decision"] != "deny"
                       and scope == session_permissions.scope_for(execution["name"], execution["input"], canonical_path=canonical_path))
        if allowed and record["option_id"] == "allowSession" and scope is not None:
            session_permissions.grant(scoped("main"), scope)
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
                if active + pending >= 6:
                    return None
                context_id, run_id = new_id(), new_id()
                helper = {"type": "agent_run", "task_id": self.task_id, "context_id": context_id,
                          "run_id": run_id, "caller": self.context_id, "goal": goal,
                          "status": "queued", "steps": 0}
                store.put(session, f"helper:{context_id}", helper)
                store.put(session, f"help-action:{identity}", {"type": "help_action", "context_id": context_id})
                store.stage_context(session, context_id, "user", f"协助任务：{goal}", key=f"help:{context_id}")
                return context_id
            helper_id = await self.runtime.change(allocate)
            return f"协助已受理，助手 ID：{helper_id}" if helper_id else "没有空闲助手，未分配协助"
        if name == "agent_wait":
            return await self.wait_for_agent(args.get("context_id"), identity)
        return f"Error: 未知通信工具 {name}"

    async def wait_for_agent(self, target: str, identity: str) -> str:
        key = f"agent-wait:{identity}"
        with db.new_session() as session:
            incoming = store.records(session, "agent_message", context_id=self.context_id)
        for message in incoming:
            if message["delivery"] in {"pending", "applying", "unknown"}:
                if not await self.runtime.deliver_message(message):
                    raise AgentWaiting

        def check(session):
            previous = store.get(session, f"wait-result:{identity}")
            if previous:
                return previous["content"]
            tasks, agents = store.records(session, "task_state"), store.records(session, "agent_run")
            task = store.get(session, f"task:{self.task_id}")
            if not (any(t["context_id"] == target for t in tasks) or any(a["context_id"] == target for a in agents)) or target == self.context_id:
                return "Error: 目标助手无效"
            messages = store.records(session, "agent_message", context_id=self.context_id, delivery="delivered")
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
                    execution = store.get(session, f"execution:{identity}")
                    wait["status"] = "ready" if execution and execution["status"] != "completed" else "completed"
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
                                   content=_limit_result(f"协助助手结果（{result}）：{content}"), message_id=f"finish:{self.helper_id}:{self.run_id}",
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
            if result == "result" and task.get("knowledge_question"):
                calls = [item for item in store.records(session, "tool_execution", context_id=self.context_id)
                         if item.get("run_id") == self.run_id and item.get("name") == "rag_search"]
                if len(calls) == 1 and calls[0].get("status") == "completed":
                    if isinstance(calls[0].get("rag_answer"), str):
                        agent_bus.send(session, source=self.context_id, target="main", task_id=self.task_id,
                                       content=calls[0]["rag_answer"], kind="rag_answer",
                                       message_id=f"finish:{self.task_id}:{self.run_id}", run_id=self.run_id)
                        return
                    try:
                        answer = json.loads(calls[0].get("output", ""))
                    except (ValueError, TypeError):
                        answer = {}
                    if answer.get("status") in {"success", "insufficient"} and isinstance(answer.get("answer"), str):
                        agent_bus.send(session, source=self.context_id, target="main", task_id=self.task_id,
                                       content=_limit_result(answer["answer"]), kind="rag_answer",
                                       message_id=f"finish:{self.task_id}:{self.run_id}", run_id=self.run_id)
                        return
            agent_bus.send(session, source=self.context_id, target="main", task_id=self.task_id,
                           content=_limit_result(f"任务【{task['title']}】一轮结束（{result}）：{content}"),
                           kind="result" if result == "stopped" else "review",
                           message_id=f"finish:{self.task_id}:{self.run_id}", run_id=self.run_id)
        await self.runtime.change(save)
