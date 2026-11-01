import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path

from fastapi import UploadFile

from app import db
from app.agent.runtime import Conflict
from app.agent.tools import workspace_root
from app.sessions import chat_store as store

MAX_UPLOAD = 100 * 1024 * 1024


def _filename(value: str) -> str:
    name = value or ""
    if (not name or name in {".", ".."} or len(name) > 240 or name[-1] in {" ", "."}
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", name)):
        raise ValueError("文件名无效")
    return name


async def upload(runtime, assistant_id: str, upload_id: str, file: UploadFile) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", upload_id):
        raise ValueError("上传 ID 无效")
    filename = _filename(file.filename or "")
    lock = runtime.upload_locks.setdefault(upload_id, asyncio.Lock())
    async with lock:
        async with runtime.lock:
            with db.new_session() as session:
                runtime._check_assistant(session, assistant_id)
                old = store.get(session, f"upload:{upload_id}")
                if old and (old["assistant_id"] != assistant_id or old["name"] != filename):
                    raise Conflict("上传 ID 已用于其他文件")
                if old and old["status"] == "started":
                    raise Conflict("上次上传结果未知，请检查文件后使用新 ID")
                if not old or old["status"] == "failed":
                    store.put(session, f"upload:{upload_id}", {"type": "upload", "id": upload_id,
                              "assistant_id": assistant_id, "name": filename, "status": "started"})
                    session.commit()
        temp_dir = runtime.data_dir / "upload-tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="upload-", dir=temp_dir)
        digest, size, replaced = hashlib.sha256(), 0, False
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise ValueError("文件超过 100 MiB")
                    digest.update(chunk)
                    await asyncio.to_thread(out.write, chunk)
            checksum = digest.hexdigest()
            if old and old["status"] == "completed":
                if old["size"] != size or old["sha256"] != checksum:
                    raise Conflict("上传 ID 已用于不同内容")
                return {key: old[key] for key in ("upload_id", "name", "path", "size")}
            root = workspace_root().resolve()
            folder = root / "uploads"
            folder.mkdir(parents=True, exist_ok=True)
            if not folder.resolve().is_relative_to(root):
                raise ValueError("上传目录越出工作区")
            target = folder / filename
            if target.is_symlink() or not target.resolve().is_relative_to(root):
                raise ValueError("上传目标无效")
            canonical = os.path.normcase(str(target.resolve()))
            async with runtime.file_locks.setdefault(canonical, asyncio.Lock()):
                async with runtime.lock:
                    with db.new_session() as session:
                        runtime._check_assistant(session, assistant_id)
                        owners = [task for task in store.records(session, "task_state")
                                  if canonical in task.get("files", [])
                                  and task["status"] not in {"completed", "failed", "stopped"}]
                        if owners:
                            raise ValueError("文件正被后台任务使用，请稍后重试")
                        if target.is_symlink() or not target.resolve().is_relative_to(root):
                            raise ValueError("上传目标已变化")
                        os.replace(temporary, target)
                        replaced = True
                        result = {"type": "upload", "id": upload_id, "upload_id": upload_id,
                                  "assistant_id": assistant_id, "name": filename,
                                  "path": f"uploads/{filename}", "size": size, "sha256": checksum,
                                  "target_key": hashlib.sha256(canonical.encode()).hexdigest(),
                                  "status": "completed"}
                        store.put(session, f"upload:{upload_id}", result)
                        store.put(session, f"upload-target:{result['target_key']}", {
                            "type": "upload_target", "id": upload_id})
                        session.commit()
            return {key: result[key] for key in ("upload_id", "name", "path", "size")}
        except BaseException:
            if not replaced and not (old and old["status"] == "completed"):
                async with runtime.lock:
                    with db.new_session() as session:
                        current = store.get(session, f"upload:{upload_id}")
                        if current and current.get("assistant_id") == assistant_id:
                            current["status"] = "failed"
                            store.put(session, f"upload:{upload_id}", current)
                            session.commit()
            raise
        finally:
            Path(temporary).unlink(missing_ok=True)
            await file.close()
