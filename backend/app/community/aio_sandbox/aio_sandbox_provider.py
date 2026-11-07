import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.community.aio_sandbox.local_backend import SandboxCleanupError
from app.community.aio_sandbox.quarantine import SandboxQuarantine
from app.community.aio_sandbox.sandbox_info import SandboxInfo
from app.sandbox.sandbox_provider import SandboxProvider
from app.user_context import require_user

logger = logging.getLogger(__name__)


@dataclass
class _Entry:
    info: SandboxInfo
    sandbox: object
    threads: set[str] = field(default_factory=set)
    state: str = "active"
    released_at: float = 0
    network_lock: threading.Lock = field(default_factory=threading.Lock)


class AioSandboxProvider(SandboxProvider):
    def __init__(self, backend, data_dir: Path, *, idle_timeout: int = 600, replicas: int = 3) -> None:
        self.backend = backend
        self.quarantine = SandboxQuarantine(data_dir)
        self.idle_timeout = idle_timeout
        self.replicas = replicas
        self._lock = threading.RLock()
        self._reconcile_lock = threading.Lock()
        self._user_locks: dict[str, threading.Lock] = {}
        self._entries: dict[str, _Entry] = {}
        self._network_grants: dict[str, dict[tuple[str, int], float | None]] = {}
        self._reconciled = False
        self._closed = False
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reap_loop, name="sandbox-reaper", daemon=True)
        self._reaper.start()

    def _sandbox_id(self, user_id: str) -> str:
        return hashlib.sha256(f"{self.backend.instance}:{user_id}".encode("utf-8")).hexdigest()[:32]

    def _reconcile(self) -> None:
        with self._reconcile_lock:
            if not self._reconciled:
                self.backend.reconcile()
                self._reconciled = True

    def _make_sandbox(self, info: SandboxInfo):
        from app.community.aio_sandbox.aio_sandbox import AioSandbox

        return AioSandbox(info, on_uncertain=lambda reason: self.quarantine_sandbox(info.sandbox_id, reason))

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
        if user_id != require_user() or not re.fullmatch(r"[0-9a-f]{32}", user_id or "") or not thread_id:
            raise PermissionError("沙箱用户或线程身份无效")
        with self._lock:
            lock = self._user_locks.setdefault(user_id, threading.Lock())
        with lock:
            self._reconcile()
            sandbox_id = self._sandbox_id(user_id)
            with self._lock:
                if self._closed:
                    raise RuntimeError("沙箱服务正在关闭")
                if self.quarantine.get(sandbox_id):
                    raise RuntimeError("沙箱已隔离，等待资源清理")
                entry = self._entries.get(sandbox_id)
                if entry and entry.state not in {"active", "warm"}:
                    raise RuntimeError("沙箱正在回收或已隔离")
            if entry:
                try:
                    alive = self.backend.is_alive(entry.info)
                except Exception:
                    self.quarantine_sandbox(sandbox_id, "资源状态无法确认")
                    raise
                if not alive:
                    self.quarantine_sandbox(sandbox_id, "资源状态异常")
                    raise RuntimeError("沙箱资源状态异常，已暂停复用")
                with self._lock:
                    if entry.state not in {"active", "warm"}:
                        raise RuntimeError("沙箱已不可复用")
                    entry.state = "active"
                    entry.threads.add(thread_id)
                return sandbox_id
            self._evict_warm()
            try:
                info = self.backend.create(thread_id, sandbox_id, user_id=user_id)
                with self._lock:
                    grants = [(host, port, expiry) for (host, port), expiry in
                              self._network_grants.get(user_id, {}).items()]
                if grants:
                    self.backend.seed_network_grants(info, grants)
                sandbox = self._make_sandbox(info)
            except SandboxCleanupError as exc:
                self.quarantine.block(sandbox_id, user_id, exc.generation, "资源清理结果未知")
                raise
            except Exception:
                if "info" in locals():
                    try:
                        self.backend.destroy(info)
                    except Exception as cleanup_error:
                        self.quarantine.block(sandbox_id, user_id, info.generation, "资源清理结果未知")
                        raise RuntimeError("沙箱初始化失败且资源清理结果未知") from cleanup_error
                raise
            with self._lock:
                if self._closed:
                    try:
                        sandbox.close()
                        self.backend.destroy(info)
                    except Exception as exc:
                        self.quarantine.block(sandbox_id, user_id, info.generation, "服务关闭时清理失败")
                        raise RuntimeError("沙箱服务关闭时清理失败") from exc
                    raise RuntimeError("沙箱服务正在关闭")
                self._entries[sandbox_id] = _Entry(info, sandbox, {thread_id})
            return sandbox_id

    def get(self, sandbox_id: str):
        with self._lock:
            entry = self._entries.get(sandbox_id)
            if not entry or entry.info.user_id != require_user() or entry.state != "active":
                return None
            return entry.sandbox

    def get_scoped(self, sandbox_id: str, *, thread_id: str, user_id: str):
        if user_id != require_user():
            return None
        with self._lock:
            entry = self._entries.get(sandbox_id)
            if (not entry or entry.info.user_id != user_id or entry.state != "active"
                    or thread_id not in entry.threads or self.quarantine.get(sandbox_id)):
                return None
            return entry.sandbox

    def release(self, sandbox_id: str) -> None:
        with self._lock:
            entry = self._entries.get(sandbox_id)
            if not entry or entry.info.user_id != require_user():
                raise PermissionError("沙箱不属于当前用户")
            if entry.state == "quarantined":
                entry.state = "destroying"
            elif entry.state == "active":
                entry.state = "warm"
                entry.released_at = time.monotonic()
                return
            else:
                return
        self._destroy(entry)

    def quarantine_sandbox(self, sandbox_id: str, reason: str) -> None:
        with self._lock:
            entry = self._entries.get(sandbox_id)
            if not entry or entry.info.user_id != require_user():
                raise PermissionError("沙箱不属于当前用户")
            entry.state = "quarantined"
            self.quarantine.block(sandbox_id, entry.info.user_id, entry.info.generation, reason)

    def _destroy(self, entry: _Entry) -> None:
        close_error = None
        try:
            entry.sandbox.close()
        except Exception as exc:
            close_error = exc
        try:
            with entry.network_lock:
                self.backend.destroy(entry.info)
        except Exception:
            with self._lock:
                entry.state = "quarantined"
                self.quarantine.block(entry.info.sandbox_id, entry.info.user_id, entry.info.generation, "资源清理失败")
            raise
        if close_error:
            logger.warning("沙箱客户端关闭失败，但容器已确认清理：%s", type(close_error).__name__)
        with self._lock:
            if self._entries.get(entry.info.sandbox_id) is entry:
                self._entries.pop(entry.info.sandbox_id)
            self.quarantine.clear(entry.info.sandbox_id, entry.info.generation)

    def network_events(self, user_id: str) -> tuple[SandboxInfo, list[dict]] | None:
        if user_id != require_user():
            raise PermissionError("沙箱不属于当前用户")
        with self._lock:
            entry = self._entries.get(self._sandbox_id(user_id))
        if not entry:
            return None
        with entry.network_lock:
            with self._lock:
                if entry.state not in {"active", "warm"}:
                    return None
            return entry.info, self.backend.network_events(entry.info)

    def acknowledge_network_event(self, info: SandboxInfo, request_id: str) -> None:
        if info.user_id != require_user():
            raise PermissionError("沙箱不属于当前用户")
        with self._lock:
            entry = self._entries.get(info.sandbox_id)
        if not entry or entry.info != info:
            return
        with entry.network_lock:
            with self._lock:
                if entry.state not in {"active", "warm"}:
                    return
            self.backend.acknowledge_network_event(info, request_id)

    def decide_network(self, user_id: str, generation: str, events: list[str], host: str,
                       port: int, option_id: str) -> None:
        if user_id != require_user():
            raise PermissionError("沙箱不属于当前用户")
        if option_id not in {"allowOnce", "allowSession", "deny"}:
            raise ValueError("网络审批选项无效")
        host, port = self.backend._network_target(host, port)
        with self._lock:
            lock = self._user_locks.setdefault(user_id, threading.Lock())
        with lock:
            with self._lock:
                entry = self._entries.get(self._sandbox_id(user_id))
                previous = self._network_grants.get(user_id, {}).get((host, port), "missing")
            expiry = time.time() + 300 if option_id == "allowOnce" else None
            if option_id != "deny" and previous != "missing":
                expiry = None if previous is None or expiry is None else max(previous, expiry)
            if entry:
                with entry.network_lock:
                    with self._lock:
                        usable = entry.state in {"active", "warm"}
                    if usable and entry.info.generation == generation:
                        decision = {"allowOnce": "allow_temporary", "allowSession": "allow_sandbox", "deny": "deny"}[option_id]
                        for request_id in events:
                            self.backend.decide_network_event(entry.info, request_id, decision)
                    if usable and option_id != "deny":
                        self.backend.seed_network_grants(entry.info, [(host, port, expiry)])
            if option_id != "deny":
                with self._lock:
                    grants = self._network_grants.setdefault(user_id, {})
                    grants[(host, port)] = expiry

    def reset_user(self, user_id: str, *, clear_grants: bool = True) -> None:
        if user_id != require_user():
            raise PermissionError("沙箱不属于当前用户")
        with self._lock:
            lock = self._user_locks.setdefault(user_id, threading.Lock())
        with lock:
            self._reconcile()
            with self._lock:
                sandbox_id = self._sandbox_id(user_id)
                entry = self._entries.get(sandbox_id)
                if self.quarantine.get(sandbox_id) and not entry:
                    raise RuntimeError("沙箱资源状态未知，无法重置")
                if entry and entry.state == "active":
                    raise RuntimeError("仍有运行中的沙箱操作")
                if entry and entry.state != "warm":
                    raise RuntimeError("沙箱尚未完成清理")
                if entry:
                    entry.state = "destroying"
            if entry:
                self._destroy(entry)
            if clear_grants:
                self.forget_network_grants(user_id)

    def forget_network_grants(self, user_id: str) -> None:
        if user_id != require_user():
            raise PermissionError("沙箱不属于当前用户")
        with self._lock:
            self._network_grants.pop(user_id, None)

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
        with self._lock:
            self._closed = True
            if any(entry.state == "active" for entry in self._entries.values()):
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
