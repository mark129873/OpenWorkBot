import hashlib
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.community.aio_sandbox.local_backend import SandboxCleanupError
from app.community.aio_sandbox.quarantine import SandboxQuarantine
from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.sandbox.sandbox_provider import SandboxProvider
from app.user_context import current_user

logger = logging.getLogger(__name__)


def _owner(user_id: str | None = None) -> str:
    bound = current_user()
    user = user_id or bound or "default"
    if not isinstance(user, str) or not re.fullmatch(r"(?:[0-9a-f]{32}|default)", user):
        raise PermissionError("沙箱用户身份无效")
    if (bound and user != bound) or (not bound and user != "default"):
        raise PermissionError("沙箱不属于当前用户")
    return user


@dataclass
class _Entry:
    info: SandboxInfo
    handles: set[str] = field(default_factory=set)
    state: str = "active"
    released_at: float = 0


class AioSandboxProvider(SandboxProvider):
    def __init__(self, backend, data_dir: Path, *, idle_timeout: int = 600, replicas: int = 3) -> None:
        self.backend = backend
        self.quarantine = SandboxQuarantine(data_dir)
        self.idle_timeout = idle_timeout
        self.replicas = replicas
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._creating = 0
        self._reconcile_lock = threading.Lock()
        self._user_locks: dict[str, threading.Lock] = {}
        self._entries: dict[str, _Entry] = {}
        self._handles: dict[str, tuple[str, str | None, object]] = {}
        self._reconciled = False
        self._closed = False
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap_loop, name="sandbox-reaper", daemon=True)
        self._reaper.start()

    def _sandbox_id(self, user_id: str) -> str:
        return hashlib.sha256(f"{self.backend.instance}:{user_id}".encode()).hexdigest()[:32]

    def _handle_id(self, physical_id: str, thread_id: str | None) -> str:
        if thread_id is None:
            return uuid.uuid4().hex
        return hashlib.sha256(f"{physical_id}\0{thread_id}".encode()).hexdigest()[:32]

    def _reconcile(self) -> None:
        with self._reconcile_lock:
            if not self._reconciled:
                self.backend.reconcile()
                self._reconciled = True

    def _make_sandbox(self, info: SandboxInfo, handle_id: str):
        from app.community.aio_sandbox.aio_sandbox import AioSandbox

        return AioSandbox(replace(info, sandbox_id=handle_id),
                          on_uncertain=lambda reason: self._quarantine_entry(info.sandbox_id, reason))

    def _evict_warm(self) -> None:
        with self._lock:
            if len(self._entries) < self.replicas:
                return
            warm = [item for item in self._entries.values() if item.state == "warm"]
            if not warm:
                return
            entry = min(warm, key=lambda item: item.released_at)
            entry.state = "destroying"
        self._destroy(entry)

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        user_id = _owner(user_id)
        if thread_id is not None and (not isinstance(thread_id, str) or not thread_id):
            raise ValueError("沙箱线程身份无效")
        with self._lock:
            user_lock = self._user_locks.setdefault(user_id, threading.Lock())
        with user_lock:
            self._reconcile()
            physical_id = self._sandbox_id(user_id)
            handle_id = self._handle_id(physical_id, thread_id)
            with self._lock:
                if self._closed:
                    raise RuntimeError("沙箱服务正在关闭")
                if self.quarantine.get(physical_id):
                    raise RuntimeError("沙箱已隔离，等待资源清理")
                entry = self._entries.get(physical_id)
                if entry and entry.state not in {"active", "warm"}:
                    raise RuntimeError("沙箱正在回收或已隔离")
            if entry:
                try:
                    alive = self.backend.is_alive(entry.info)
                except Exception:
                    self._quarantine_entry(physical_id, "资源状态无法确认")
                    raise
                if not alive:
                    self._quarantine_entry(physical_id, "资源状态异常")
                    raise RuntimeError("沙箱资源状态异常，已暂停复用")
            else:
                self._evict_warm()
                with self._changed:
                    if self._closed:
                        raise RuntimeError("沙箱服务正在关闭")
                    self._creating += 1
                try:
                    try:
                        info = self.backend.create(thread_id or handle_id, physical_id, user_id=user_id)
                    except SandboxCleanupError as exc:
                        self.quarantine.block(physical_id, user_id, exc.generation, "资源清理结果未知")
                        raise
                    entry = _Entry(info)
                    with self._lock:
                        closing = self._closed
                        if not closing:
                            self._entries[physical_id] = entry
                    if closing:
                        try:
                            self.backend.destroy(info)
                        except Exception as exc:
                            self.quarantine.block(physical_id, user_id, info.generation, "服务关闭时清理失败")
                            raise RuntimeError("沙箱服务关闭时清理失败") from exc
                        raise RuntimeError("沙箱服务正在关闭")
                finally:
                    with self._changed:
                        self._creating -= 1
                        self._changed.notify_all()
            with self._lock:
                if self._closed or entry.state not in {"active", "warm"}:
                    raise RuntimeError("沙箱已不可复用")
                if handle_id not in self._handles:
                    try:
                        sandbox = self._make_sandbox(entry.info, handle_id)
                    except Exception:
                        if not entry.handles:
                            entry.state = "warm"
                            entry.released_at = time.monotonic()
                        raise
                    self._handles[handle_id] = (physical_id, thread_id, sandbox)
                    entry.handles.add(handle_id)
                entry.state = "active"
            return handle_id

    def get(self, sandbox_id: str):
        user = _owner()
        with self._lock:
            handle = self._handles.get(sandbox_id)
            entry = self._entries.get(handle[0]) if handle else None
            if not entry or entry.info.user_id != user or entry.state != "active":
                return None
            return handle[2]

    def get_scoped(self, sandbox_id: str, *, thread_id: str, user_id: str):
        try:
            user = _owner(user_id)
        except PermissionError:
            return None
        with self._lock:
            handle = self._handles.get(sandbox_id)
            entry = self._entries.get(handle[0]) if handle else None
            if (not entry or entry.info.user_id != user or entry.state != "active"
                    or handle[1] != thread_id or self.quarantine.get(handle[0])):
                return None
            return handle[2]

    def release(self, sandbox_id: str) -> None:
        user = _owner()
        with self._lock:
            handle = self._handles.get(sandbox_id)
            entry = self._entries.get(handle[0]) if handle else None
            if not entry or entry.info.user_id != user:
                raise PermissionError("沙箱不属于当前用户")
            user_lock = self._user_locks.setdefault(user, threading.Lock())
        with user_lock:
            with self._lock:
                handle = self._handles.pop(sandbox_id, None)
                if handle is None:
                    return
                entry.handles.remove(sandbox_id)
            close_error = None
            try:
                handle[2].close()
            except Exception as exc:
                close_error = exc
                self._quarantine_entry(entry.info.sandbox_id, "沙箱会话清理失败")
            with self._lock:
                if entry.handles:
                    if close_error:
                        raise RuntimeError("沙箱会话清理失败，已暂停复用") from close_error
                    return
                if entry.state == "quarantined":
                    entry.state = "destroying"
                else:
                    entry.state = "warm"
                    entry.released_at = time.monotonic()
                    return
            self._destroy(entry)

    def _quarantine_entry(self, physical_id: str, reason: str) -> None:
        with self._lock:
            entry = self._entries.get(physical_id)
            if not entry:
                raise RuntimeError("沙箱资源不存在")
            entry.state = "quarantined"
            self.quarantine.block(physical_id, entry.info.user_id, entry.info.generation, reason)

    def quarantine_sandbox(self, sandbox_id: str, reason: str) -> None:
        user = _owner()
        with self._lock:
            handle = self._handles.get(sandbox_id)
            entry = self._entries.get(handle[0]) if handle else None
            if not entry or entry.info.user_id != user:
                raise PermissionError("沙箱不属于当前用户")
        self._quarantine_entry(entry.info.sandbox_id, reason)

    def _destroy(self, entry: _Entry) -> None:
        try:
            self.backend.destroy(entry.info)
        except Exception:
            with self._lock:
                entry.state = "quarantined"
                self.quarantine.block(entry.info.sandbox_id, entry.info.user_id, entry.info.generation, "资源清理失败")
            raise
        with self._lock:
            if self._entries.get(entry.info.sandbox_id) is entry:
                self._entries.pop(entry.info.sandbox_id)
            self.quarantine.clear(entry.info.sandbox_id, entry.info.generation)

    def reset_user(self, user_id: str) -> None:
        user_id = _owner(user_id)
        with self._lock:
            user_lock = self._user_locks.setdefault(user_id, threading.Lock())
        with user_lock:
            self._reconcile()
            with self._lock:
                physical_id = self._sandbox_id(user_id)
                entry = self._entries.get(physical_id)
                if self.quarantine.get(physical_id) and not entry:
                    raise RuntimeError("沙箱资源状态未知，无法重置")
                if entry and entry.handles:
                    raise RuntimeError("仍有运行中的沙箱操作")
                if entry and entry.state != "warm":
                    raise RuntimeError("沙箱尚未完成清理")
                if entry:
                    entry.state = "destroying"
            if entry:
                self._destroy(entry)

    def cleanup_idle(self) -> None:
        with self._lock:
            expired = [entry for entry in self._entries.values() if entry.state == "warm"
                       and time.monotonic() - entry.released_at >= self.idle_timeout]
            for entry in expired:
                entry.state = "destroying"
        for entry in expired:
            self._destroy(entry)

    def _reap_loop(self) -> None:
        while not self._stop.wait(30):
            try:
                self.cleanup_idle()
            except Exception:
                logger.exception("空闲沙箱清理失败")

    def close(self) -> None:
        with self._changed:
            self._closed = True
            while self._creating:
                self._changed.wait()
            if any(entry.handles for entry in self._entries.values()):
                raise RuntimeError("仍有运行中的沙箱操作")
        self._stop.set()
        self._reaper.join(timeout=5)
        if self._reaper.is_alive():
            raise RuntimeError("沙箱清理尚未结束")
        with self._lock:
            if any(entry.state == "destroying" for entry in self._entries.values()):
                raise RuntimeError("沙箱清理尚未结束")
            entries = list(self._entries.values())
            for entry in entries:
                entry.state = "destroying"
        for entry in entries:
            self._destroy(entry)

    def reset(self) -> None:
        self.close()
