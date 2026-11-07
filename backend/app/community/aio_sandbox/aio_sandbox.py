import json
import math
import os
import shlex
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx
from agent_sandbox import Sandbox as AioClient

from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.sandbox.sandbox import Sandbox


_REQUEST = {"timeout_in_seconds": 5, "max_retries": 0}
_OUTPUT_LIMIT = 20000
_OUTPUT_MARKER = "\n...（输出过长，中段已截断）...\n"


@dataclass
class CommandResult:
    output: str
    status: str
    exit_code: int | None

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.exit_code == 0


@dataclass
class _Scope:
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    cancelled: threading.Event = field(default_factory=threading.Event)


def _preview(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    remaining = _OUTPUT_LIMIT - len(_OUTPUT_MARKER)
    head = remaining // 2
    return text[:head] + _OUTPUT_MARKER + text[-(remaining - head):]


class AioSandbox(Sandbox):
    def __init__(self, info: SandboxInfo, *, on_uncertain=None) -> None:
        super().__init__(info.sandbox_id)
        self.info = info
        self._http = httpx.Client(trust_env=False, timeout=30)
        self._client = AioClient(base_url=info.sandbox_url,
                                 headers={"X-DeerFlow-Relay-Token": info.relay_token}, httpx_client=self._http)
        self._on_uncertain = on_uncertain
        self._lock = threading.RLock()
        self._scopes: dict[str, _Scope] = {}
        self._closed = False

    def _uncertain(self, reason: str) -> None:
        if self._on_uncertain:
            self._on_uncertain(reason)

    def _scope(self, scope_id: str) -> _Scope:
        if not isinstance(scope_id, str) or not scope_id:
            raise ValueError("命令运行身份无效")
        with self._lock:
            if self._closed:
                raise RuntimeError("沙箱已关闭")
            return self._scopes.setdefault(scope_id, _Scope())

    def _shell_session(self, scope: _Scope) -> str:
        if scope.session_id:
            return scope.session_id
        session_id = str(uuid.uuid4())
        try:
            result = self._client.shell.create_session(id=session_id, exec_dir="/mnt/user-data/workspace",
                                                       request_options=_REQUEST)
            if not result.success or not result.data or result.data.session_id != session_id:
                raise RuntimeError("沙箱 shell 会话创建失败")
        except Exception:
            self._uncertain("shell 会话创建结果未知")
            raise
        scope.session_id = session_id
        return session_id

    def execute_command(self, command: str, env: dict[str, str] | None = None,
                        timeout: float | None = None) -> str:
        return self.execute_command_in_scope(command, env=env, timeout=timeout, scope_id="default")

    def execute_command_in_scope(self, command: str, env: dict[str, str] | None = None,
                                 timeout: float | None = None, *, scope_id: str | None = None) -> str:
        result = self.execute_command_result_in_scope(command, env=env, timeout=timeout, scope_id=scope_id)
        if result.ok:
            return _preview(result.output) or "(无输出)"
        if result.status == "unknown":
            return _preview(f"Error: 命令结果未知，请先核查；{result.output}")
        if result.status == "cancelled":
            return _preview(f"Error: 命令已停止；{result.output}")
        if result.status == "timeout":
            return _preview(f"Error: 命令超时；{result.output}")
        return _preview(f"Error: 命令退出码 {result.exit_code}；{result.output}")

    def execute_command_result_in_scope(self, command: str, env: dict[str, str] | None = None,
                                        timeout: float | None = None, *, scope_id: str | None = None) -> CommandResult:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("命令不能为空")
        if env:
            raise ValueError("当前沙箱命令不接受额外环境变量")
        if timeout is not None and (type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("命令超时无效")
        duration = float(120 if timeout is None else timeout)
        scope = self._scope(scope_id or "default")
        with scope.lock:
            if scope.cancelled.is_set():
                return CommandResult("", "cancelled", None)
            session_id = self._shell_session(scope)
            prepared = f"export OPENWORKBOT_SCOPE_TOKEN={scope.token}; {command}"
            try:
                result = self._client.shell.exec_command(id=session_id, command=prepared, async_mode=True,
                                                         hard_timeout=duration,
                                                         no_change_timeout=max(600, math.ceil(duration + 5)),
                                                         request_options=_REQUEST)
                if not result.success or not result.data:
                    raise RuntimeError("沙箱未确认命令受理")
                deadline = time.monotonic() + duration
                while True:
                    if scope.cancelled.is_set() or time.monotonic() >= deadline:
                        self._stop_scope(scope)
                        return CommandResult("", "cancelled" if scope.cancelled.is_set() else "timeout", None)
                    wait = self._client.shell.wait_for_process(id=session_id, seconds=1, max_wait_seconds=1,
                                                               request_options=_REQUEST)
                    status = getattr(wait.data, "status", None) if wait.success else None
                    if status == "running":
                        continue
                    view = self._client.shell.view(id=session_id, request_options=_REQUEST)
                    if not view.success or not view.data:
                        raise RuntimeError("沙箱命令结果不可用")
                    data = view.data
                    output = next((item.output or "" for item in reversed(data.console or [])
                                   if item.command == prepared), None)
                    if output is None:
                        raise RuntimeError("沙箱未返回当前命令结果")
                    if data.status in {"hard_timeout", "no_change_timeout", "terminated"}:
                        self._stop_scope(scope)
                        return CommandResult(_preview(output), "timeout" if data.status == "hard_timeout" else "unknown",
                                             getattr(data, "exit_code", None))
                    if data.status != "completed" or getattr(data, "exit_code", None) is None:
                        raise RuntimeError("沙箱命令状态不完整")
                    return CommandResult(_preview(output), "completed", data.exit_code)
            except Exception:
                self._uncertain("命令执行结果未知")
                return CommandResult("", "unknown", None)

    def request_command_cancel(self, scope_id: str) -> None:
        with self._lock:
            scope = self._scopes.get(scope_id)
        if scope:
            scope.cancelled.set()

    def cancel_command_scope(self, scope_id: str) -> None:
        with self._lock:
            scope = self._scopes.get(scope_id)
        if not scope:
            return
        scope.cancelled.set()
        with scope.lock:
            self._stop_scope(scope)

    def _stop_scope(self, scope: _Scope) -> None:
        session_id = scope.session_id
        try:
            if session_id:
                result = self._client.shell.kill_process(id=session_id, request_options=_REQUEST)
                if not result.success:
                    raise RuntimeError("无法停止 shell")
            response = self._bash_helper("python /opt/openworkbot-process-cleanup.py " + shlex.quote(scope.token), None)
            if not response.get("ok"):
                raise RuntimeError("不能确认残留进程已结束")
            if session_id:
                self._cleanup_shell_session(session_id)
        except Exception:
            self._uncertain("无法确认残留进程已结束")
            raise
        finally:
            scope.session_id = None

    def _cleanup_shell_session(self, session_id: str) -> None:
        result = self._client.shell.cleanup_session(session_id, request_options=_REQUEST)
        if result.success:
            return
        sessions = self._client.shell.list_sessions(request_options=_REQUEST)
        if not sessions.success or not sessions.data or session_id in sessions.data.sessions:
            raise RuntimeError("shell 会话清理失败")

    def _bash_helper(self, command: str, input_text: str | None) -> dict:
        session_id = str(uuid.uuid4())
        try:
            session = self._client.bash.create_session(session_id=session_id, exec_dir="/mnt/user-data/workspace",
                                                       request_options=_REQUEST)
            if not session.success or not session.data or session.data.session_id != session_id:
                raise RuntimeError("沙箱辅助会话创建失败")
            execution = self._client.bash.exec(session_id=session_id, command=command, async_mode=True,
                                               hard_timeout=120, request_options=_REQUEST)
            if not execution.success or not execution.data or not execution.data.command_id:
                raise RuntimeError("沙箱辅助程序未确认受理")
            command_id = execution.data.command_id
            if input_text is not None:
                written = self._client.bash.write(session_id=session_id, command_id=command_id,
                                                   input=input_text, request_options=_REQUEST)
                if not written.success:
                    raise RuntimeError("沙箱辅助程序输入未确认")
            output, error, offset, error_offset = "", "", 0, 0
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                response = self._client.bash.output(session_id=session_id, command_id=command_id,
                                                    offset=offset, stderr_offset=error_offset,
                                                    wait=True, wait_timeout=1, request_options=_REQUEST)
                if not response.success or not response.data:
                    raise RuntimeError("沙箱辅助程序结果不可用")
                data = response.data
                output += data.stdout or ""
                error += data.stderr or ""
                if len(output) > 1024 * 1024 or len(error) > 30000:
                    raise RuntimeError("沙箱辅助程序输出异常")
                offset, error_offset = data.offset, data.stderr_offset
                if data.command and data.command.status != "running":
                    if data.command.status != "completed" or data.command.exit_code != 0:
                        raise RuntimeError("沙箱辅助程序执行失败")
                    result = json.loads(output)
                    if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                        raise RuntimeError("沙箱辅助程序返回无效结果")
                    return result
            raise TimeoutError("沙箱辅助程序超时")
        except Exception:
            self._uncertain("沙箱辅助程序结果未知")
            raise
        finally:
            try:
                result = self._client.bash.close_session(session_id, request_options=_REQUEST)
                if not result.success:
                    raise RuntimeError("沙箱辅助会话清理失败")
            except Exception:
                self._uncertain("沙箱辅助会话清理失败")

    def file_operation(self, operation: str, args: dict) -> str:
        if operation not in {"read_file", "write_file", "edit_file", "glob", "delete_file", "version"}:
            raise ValueError("未知文件操作")
        request = json.dumps({"operation": operation, "args": args, "windows_newline": os.name == "nt"},
                             ensure_ascii=True) + "\n"
        result = self._bash_helper("python /opt/openworkbot-file-ops.py", request)
        if not result["ok"]:
            return f"Error: {result.get('error', '文件操作失败')}"
        if not isinstance(result.get("output"), str):
            raise RuntimeError("沙箱文件操作返回无效结果")
        return result["output"]

    @staticmethod
    def _workspace_path(path: str) -> str:
        root = "/mnt/user-data/workspace"
        if path == root:
            return "."
        if path.startswith(root + "/"):
            return path[len(root) + 1:]
        if path.startswith("/"):
            raise ValueError("路径必须位于工作区内")
        return path

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        offset = start_line or 1
        limit = min(2000, end_line - offset + 1) if end_line is not None else 2000
        return self.file_operation("read_file", {"path": self._workspace_path(path), "offset": offset, "limit": limit})

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        output = self.file_operation("write_file", {"path": self._workspace_path(path), "content": content,
                                                    "append": append})
        if output.startswith("Error:"):
            raise OSError(output)

    def glob(self, path: str, pattern: str, *, include_dirs: bool = False,
             max_results: int = 200) -> tuple[list[str], bool]:
        if include_dirs:
            raise ValueError("仅支持工作区内文件查找")
        directory = self._workspace_path(path) if path else "."
        prefix = "" if directory == "." else directory.rstrip("/") + "/"
        output = self.file_operation("glob", {"pattern": prefix + pattern})
        if output.startswith("Error:"):
            raise OSError(output)
        lines = [] if output == "(无匹配文件)" else output.splitlines()
        return lines[:max_results], len(lines) > max_results

    def release_command_scope(self, scope_id: str) -> None:
        with self._lock:
            scope = self._scopes.get(scope_id)
        if not scope:
            return
        with scope.lock:
            if scope.session_id:
                try:
                    result = self._bash_helper("python /opt/openworkbot-process-cleanup.py " + shlex.quote(scope.token), None)
                    if not result.get("ok"):
                        raise RuntimeError("不能确认残留进程已结束")
                    self._cleanup_shell_session(scope.session_id)
                except Exception:
                    self._uncertain("无法确认残留进程已结束")
                    raise
            with self._lock:
                self._scopes.pop(scope_id, None)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            scopes = list(self._scopes)
        error = None
        for scope_id in scopes:
            try:
                self.release_command_scope(scope_id)
            except Exception as exc:
                error = exc
        self._http.close()
        if error:
            raise RuntimeError("沙箱会话清理失败") from error
