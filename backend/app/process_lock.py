"""同一数据目录只允许运行一个后端。"""

import os
from pathlib import Path


class ProcessLock:
    def __init__(self, data_dir: Path):
        self.path = data_dir / ".backend.lock"
        self.file = None

    def acquire(self) -> None:
        try:
            self.file = self.path.open("a+b")
            self.file.seek(0, os.SEEK_END)
            if self.file.tell() == 0:
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if self.file is not None:
                self.file.close()
            self.file = None
            raise RuntimeError(f"数据目录已有运行中的后端：{self.path.parent}") from exc

    def release(self) -> None:
        if self.file is None:
            return
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()
        self.file = None
