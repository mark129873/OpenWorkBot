import base64
import errno
import json
import math
import re
import shlex
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx
from agent_sandbox import Sandbox as AioClient
from pydantic import ValidationError

from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.sandbox.sandbox import Sandbox
from app.sandbox.search import GrepMatch, ignored, path_matches


_REQUEST = {"timeout_in_seconds": 5, "max_retries": 0}


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
    tokens: set[str] = field(default_factory=set)
    session_id: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    cancelled: threading.Event = field(default_factory=threading.Event)


class AioSandbox(Sandbox):
    artifact_root = "/mnt/user-data"
    persistent_shell_sessions = True
    supports_command_cancellation = True

    def __init__(self, info: SandboxInfo, *, on_uncertain=None) -> None:
        super().__init__(info.sandbox_id)
        self.info = info
        self._http = httpx.Client(trust_env=False, timeout=30)
        self._client = AioClient(base_url=info.sandbox_url, headers=info.request_headers, httpx_client=self._http)
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
            return result.output or "(无输出)"
        if result.status == "unknown":
            return f"Error: 命令结果未知，请先核查；{result.output}"
        if result.status == "cancelled":
            return f"Error: 命令已停止；{result.output}"
        if result.status == "timeout":
            return f"Error: 命令超时；{result.output}"
        return f"Error: 命令退出码 {result.exit_code}；{result.output}"

    def execute_command_result_in_scope(self, command: str, env: dict[str, str] | None = None,
                                        timeout: float | None = None, *, scope_id: str | None = None) -> CommandResult:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("命令不能为空")
        if env is not None and (not isinstance(env, dict) or any(
                not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                or not isinstance(value, str) or "\0" in value for key, value in env.items())):
            raise ValueError("环境变量无效")
        if timeout is not None and (type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("命令超时无效")
        duration = float(120 if timeout is None else timeout)
        scope = self._scope(scope_id or "default")
        with scope.lock:
            if scope.cancelled.is_set():
                return CommandResult("", "cancelled", None)
            if env:
                return self._execute_extra_env(command, env, duration, scope)
            session_id = self._shell_session(scope)
            scope.tokens.add(scope.token)
            prepared = f"export OPENWORKBOT_SCOPE_{scope.token.upper()}=1; {command}"
            try:
                result = self._client.shell.exec_command(id=session_id, command=prepared, async_mode=True,
                                                         hard_timeout=duration,
                                                         no_change_timeout=max(600, math.ceil(duration + 5)),
                                                         truncate=False,
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
                        return CommandResult(output, "timeout" if data.status == "hard_timeout" else "unknown",
                                             getattr(data, "exit_code", None))
                    if data.status != "completed" or getattr(data, "exit_code", None) is None:
                        raise RuntimeError("沙箱命令状态不完整")
                    return CommandResult(output, "completed", data.exit_code)
            except Exception:
                self._uncertain("命令执行结果未知")
                return CommandResult("", "unknown", None)

    def _execute_extra_env(self, command: str, env: dict[str, str], duration: float, scope: _Scope) -> CommandResult:
        token = scope.token
        while f"OPENWORKBOT_SCOPE_{token.upper()}" in env:
            token = uuid.uuid4().hex
        scope.tokens.add(token)
        variables = {**env, f"OPENWORKBOT_SCOPE_{token.upper()}": "1"}
        session_id = str(uuid.uuid4())
        created = False
        result = CommandResult("", "unknown", None)
        try:
            session = self._client.bash.create_session(session_id=session_id, exec_dir="/mnt/user-data/workspace",
                                                       request_options=_REQUEST)
            if not session.success or not session.data or session.data.session_id != session_id:
                raise RuntimeError("沙箱临时会话创建失败")
            created = True
            execution = self._client.bash.exec(session_id=session_id, command=command, env=variables,
                                               async_mode=True, hard_timeout=duration, request_options=_REQUEST)
            if not execution.success or not execution.data or not execution.data.command_id:
                raise RuntimeError("沙箱未确认命令受理")
            command_id = execution.data.command_id
            output, offset, error_offset = "", 0, 0
            deadline = time.monotonic() + duration
            while True:
                if scope.cancelled.is_set() or time.monotonic() >= deadline:
                    stopped = self._client.bash.kill(session_id=session_id, request_options=_REQUEST)
                    if not stopped.success:
                        raise RuntimeError("无法停止临时命令")
                    self._stop_scope(scope)
                    result = CommandResult(output, "cancelled" if scope.cancelled.is_set() else "timeout", None)
                    break
                response = self._client.bash.output(session_id=session_id, command_id=command_id,
                                                    offset=offset, stderr_offset=error_offset,
                                                    wait=True, wait_timeout=1, request_options=_REQUEST)
                if not response.success or not response.data:
                    raise RuntimeError("沙箱命令结果不可用")
                data = response.data
                output += (data.stdout or "") + (data.stderr or "")
                offset, error_offset = data.offset, data.stderr_offset
                if data.command and data.command.status not in {"pending", "running"}:
                    status = data.command.status
                    exit_code = data.command.exit_code
                    if status == "completed" and exit_code is not None:
                        result = CommandResult(output, "completed", exit_code)
                    elif status == "timed_out":
                        self._stop_scope(scope)
                        result = CommandResult(output, "timeout", exit_code)
                    else:
                        raise RuntimeError("沙箱命令状态不完整")
                    break
        except Exception:
            self._uncertain("临时命令执行结果未知")
            result = CommandResult("", "unknown", None)
        finally:
            if created:
                try:
                    closed = self._client.bash.close_session(session_id, request_options=_REQUEST)
                    if not closed.success:
                        raise RuntimeError("沙箱临时会话清理失败")
                except Exception:
                    self._uncertain("沙箱临时会话清理失败")
                    result = CommandResult("", "unknown", None)
        return result

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
            self._cleanup_scope_processes(scope)
            if session_id:
                self._cleanup_shell_session(session_id)
        except Exception:
            self._uncertain("无法确认残留进程已结束")
            raise
        finally:
            scope.session_id = None

    def _cleanup_scope_processes(self, scope: _Scope) -> None:
        for token in scope.tokens:
            response = self._bash_helper("python /opt/openworkbot-process-cleanup.py " + shlex.quote(token), None)
            if not response.get("ok"):
                raise RuntimeError("不能确认残留进程已结束")

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
                if data.command and data.command.status not in {"pending", "running"}:
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
            except Exception as exc:
                self._uncertain("沙箱辅助会话清理失败")
                raise RuntimeError("沙箱辅助会话清理失败") from exc

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        try:
            kwargs = {}
            if start_line is not None:
                kwargs["start_line"] = max(start_line - 1, 0)
            if end_line is not None:
                kwargs["end_line"] = max(end_line, 0)
            result = self._client.file.read_file(file=path, **kwargs)
            return result.data.content if result.data else ""
        except ValidationError:
            return f"Error: file not found or unreadable: {path}"
        except Exception as exc:
            return f"Error: {exc}"

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        result = self._client.file.write_file(file=path, content=content, append=append)
        if not result.success:
            raise OSError("沙箱写入失败")

    def download_file(self, path: str, *, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("下载大小限制无效")
        normal = path.replace("\\", "/")
        if ".." in normal.split("/") or not (normal == "/mnt/user-data" or normal.startswith("/mnt/user-data/")):
            raise PermissionError("下载路径必须位于 /mnt/user-data")
        limit = min(max_bytes, 100 * 1024 * 1024) if max_bytes is not None else 100 * 1024 * 1024
        chunks, size = [], 0
        stream = None
        try:
            stream = self._client.file.download_file(path=path)
            for chunk in stream:
                size += len(chunk)
                if size > limit:
                    raise OSError(errno.EFBIG, "文件超过下载大小限制", path)
                chunks.append(chunk)
            return b"".join(chunks)
        except OSError:
            raise
        except Exception as exc:
            raise OSError(f"无法下载文件：{path}") from exc
        finally:
            if stream is not None and hasattr(stream, "close"):
                stream.close()

    def update_file(self, path: str, content: bytes) -> None:
        result = self._client.file.write_file(file=path, content=base64.b64encode(content).decode(), encoding="base64")
        if not result.success:
            raise OSError("沙箱写入失败")

    def real_path(self, path: str) -> str:
        program = "import json,os,sys; print(json.dumps({'ok':True,'path':os.path.realpath(sys.argv[1])}))"
        result = self._bash_helper(f"python -c {shlex.quote(program)} {shlex.quote(path)}", None)
        resolved = result.get("path")
        if not isinstance(resolved, str) or not resolved.startswith("/"):
            raise OSError("沙箱路径解析失败")
        return resolved

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("目录深度无效")
        result = self._client.file.list_path(path=path, recursive=True, show_hidden=True, max_depth=max_depth)
        if not result.success or not result.data:
            raise OSError(f"无法列出目录：{path}")
        entries = [entry.path for entry in result.data.files or [] if not ignored(entry.path, path)]
        return [path, *[entry for entry in entries if entry != path]][:500]

    def glob(self, path: str, pattern: str, *, include_dirs: bool = False,
             max_results: int = 200) -> tuple[list[str], bool]:
        root = path.rstrip("/") or "/"
        if not include_dirs:
            result = self._client.file.find_files(path=path, glob=pattern)
            if not result.success:
                raise OSError("沙箱文件搜索失败")
            files = result.data.files if result.data and result.data.files else []
            matches = [item for item in files if not ignored(item, root)]
        else:
            result = self._client.file.list_path(path=path, recursive=True, show_hidden=False)
            if not result.success:
                raise OSError("沙箱文件搜索失败")
            entries = result.data.files if result.data and result.data.files else []
            prefix = "/" if root == "/" else root + "/"
            matches = [entry.path for entry in entries if entry.path.startswith(prefix)
                       and not ignored(entry.path, root) and path_matches(pattern, entry.path[len(prefix):])]
        return matches[:max_results], len(matches) > max_results

    def grep(self, path: str, pattern: str, *, glob: str | None = None, literal: bool = False,
             case_sensitive: bool = False, max_results: int = 100) -> tuple[list[GrepMatch], bool]:
        re.compile(re.escape(pattern) if literal else pattern, 0 if case_sensitive else re.IGNORECASE)
        result = self._client.file.grep_files(path=path, pattern=pattern, case_insensitive=not case_sensitive,
                                              fixed_strings=literal, max_results=max(max_results * 4, max_results + 50),
                                              max_file_size="1M", recursive=True)
        if not result.success:
            raise OSError("沙箱内容搜索失败")
        data = result.data
        root = path.rstrip("/") or "/"
        prefix = "/" if root == "/" else root + "/"
        matches = []
        for item in data.matches if data and data.matches else []:
            if ignored(item.file, root):
                continue
            if item.file == root:
                relative = item.file.rsplit("/", 1)[-1]
            elif item.file.startswith(prefix):
                relative = item.file[len(prefix):]
            else:
                continue
            if glob is not None and not path_matches(glob, relative):
                continue
            line = item.line_content.rstrip("\r\n")
            matches.append(GrepMatch(item.file, item.line_number, line if len(line) <= 200 else line[:197] + "..."))
            if len(matches) > max_results:
                return matches[:max_results], True
        return matches, bool(data and data.truncated)

    def release_command_scope(self, scope_id: str) -> None:
        with self._lock:
            scope = self._scopes.get(scope_id)
        if not scope:
            return
        with scope.lock:
            if scope.tokens or scope.session_id:
                try:
                    self._cleanup_scope_processes(scope)
                    if scope.session_id:
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
