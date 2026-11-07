import json
import os
import tempfile
import threading
from pathlib import Path


class SandboxQuarantine:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "sandbox-quarantine.json"
        self._lock = threading.Lock()
        self._entries = self._read()

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(entries, dict) or any(not isinstance(value, dict) for value in entries.values()):
                raise ValueError("隔离记录格式无效")
            return entries
        except (OSError, ValueError) as exc:
            raise RuntimeError("无法读取沙箱隔离记录") from exc

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".sandbox-quarantine-", delete=False) as temp:
                temp_name = temp.name
                json.dump(self._entries, temp, ensure_ascii=False, sort_keys=True)
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(temp_name, self.path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def block(self, sandbox_id: str, user_id: str, generation: str, reason: str) -> None:
        if not sandbox_id or not user_id or not generation:
            raise ValueError("沙箱隔离身份不完整")
        with self._lock:
            self._entries[sandbox_id] = {"user_id": user_id, "generation": generation, "reason": reason}
            self._write()

    def get(self, sandbox_id: str) -> dict | None:
        with self._lock:
            entry = self._entries.get(sandbox_id)
            return dict(entry) if entry else None

    def clear(self, sandbox_id: str, generation: str) -> bool:
        with self._lock:
            entry = self._entries.get(sandbox_id)
            if not entry or entry.get("generation") != generation:
                return False
            del self._entries[sandbox_id]
            try:
                self._write()
            except Exception:
                self._entries[sandbox_id] = entry
                raise
            return True
