import hashlib
import logging
import os
import re
import uuid
from pathlib import Path

from app import db
from app.agent.tools import workspace_root
from app.config import settings
from app.sessions import chat_store as store

logger = logging.getLogger(__name__)
_DIRECTORY = ".tool-results"
_VIRTUAL_ROOT = "/mnt/user-data/workspace/.tool-results/"
_SCOPE = re.compile(r"(?:main|[0-9a-f]{32})\Z")
_FILE = re.compile(r"[a-zA-Z0-9_-]+-[0-9a-f]{64}\.txt\Z")
_REFERENCE = re.compile(r"\[完整工具结果：(/mnt/user-data/workspace/\.tool-results/[^\]\n]+)\]")


def _link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _archive_path(scope: str, filename: str, *, create: bool = False) -> Path:
    if not _SCOPE.fullmatch(scope) or not _FILE.fullmatch(filename):
        raise ValueError("工具输出路径无效")
    workspace = workspace_root()
    root = workspace / _DIRECTORY
    directory = root / scope
    target = directory / filename
    for path in (workspace, root, directory, target):
        if _link(path):
            raise ValueError("工具输出路径不能是链接")
    if not target.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("工具输出路径越界")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return target


def _write(scope: str, filename: str, content: str) -> None:
    target = _archive_path(scope, filename, create=True)
    if target.exists():
        with target.open("rb") as stream:
            existing = hashlib.file_digest(stream, "sha256").hexdigest()
        if existing != hashlib.sha256(content.encode("utf-8")).hexdigest():
            raise FileExistsError("工具结果文件已被修改")
        return
    temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _description(scope: str, tool: str, call_id: str, content: str) -> tuple[dict, str]:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", tool)[:60] or "tool"
    identity, data = call_id.encode("utf-8"), content.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(identity).to_bytes(8, "big"))
    digest.update(identity)
    digest.update(data)
    filename = f"{safe_name}-{digest.hexdigest()}.txt"
    _archive_path(scope, filename)
    path = f"{_VIRTUAL_ROOT}{scope}/{filename}"
    return {"type": "tool_archive", "scope": scope, "filename": filename,
            "digest": hashlib.sha256(data).hexdigest()}, path


def _preview(content: str, path: str) -> str:
    head = min(len(content), max(0, settings.tool_output_preview_head))
    tail = min(len(content) - head, max(0, settings.tool_output_preview_tail))
    prefix = content[:head]
    if content.startswith("Error:") and not prefix.startswith("Error:"):
        prefix = "Error: "
    suffix = content[-tail:] if tail else ""
    return (f"{prefix}\n…（工具结果过长，已保存原文）…\n{suffix}"
            f"\n[完整工具结果：{path}]\n使用 read_file 按字符偏移分段读取。")


async def archive(runtime, scope: str, tool: str, call_id: str, content: str) -> str:
    if (not isinstance(content, str) or tool == "read_file" or settings.tool_output_threshold <= 0
            or len(content) <= settings.tool_output_threshold):
        return content
    try:
        entry, path = await runtime.run_tool(_description, scope, tool, call_id, content)
        with db.new_session() as session:
            meta = store.get(session, "meta")
            assistant_id = meta["assistant_id"]
        entry.update(assistant_id=assistant_id, status="pending")
        key = f"tool-archive:{entry['filename']}"

        def reserve(session):
            meta = store.get(session, "meta")
            if not meta or meta["assistant_id"] != assistant_id or meta.get("reset"):
                raise ValueError("助手正在重置")
            store.put(session, key, entry)

        await runtime.change(reserve)
        await runtime.run_tool(_write, scope, entry["filename"], content)

        def publish(session):
            current = store.get(session, key)
            meta = store.get(session, "meta")
            if not current or not meta or meta["assistant_id"] != assistant_id or meta.get("reset"):
                raise ValueError("助手正在重置")
            current["status"] = "ready"
            store.put(session, key, current)

        await runtime.change(publish)
        return _preview(content, path)
    except Exception:
        logger.exception("工具结果归档失败，保留原始结果")
        return content


def read_reference(content: str, *, max_chars: int) -> str | None:
    matches = list(_REFERENCE.finditer(content))
    if not matches:
        return None
    reference = matches[-1].group(1)
    parts = reference.removeprefix(_VIRTUAL_ROOT).split("/")
    if len(parts) != 2:
        return None
    try:
        path = _archive_path(parts[0], parts[1])
        if not path.is_file() or path.stat().st_size > max_chars * 4:
            return None
        with path.open("r", encoding="utf-8", newline="") as stream:
            text = stream.read()
        return text if len(text) <= max_chars else None
    except (OSError, ValueError, UnicodeError):
        return None


def matches_file(reference: str, content: str) -> bool:
    if not reference.startswith(_VIRTUAL_ROOT):
        return False
    parts = reference.removeprefix(_VIRTUAL_ROOT).split("/")
    if len(parts) != 2:
        return False
    try:
        path = _archive_path(parts[0], parts[1])
        if not path.is_file():
            return False
        with path.open("r", encoding="utf-8", newline="") as stream:
            return stream.read() == content
    except (OSError, ValueError, UnicodeError):
        return False


def list_directory(path: str, offset: int, limit: int) -> str:
    root_path = _VIRTUAL_ROOT.rstrip("/")
    if path.rstrip("/") == root_path:
        scope = None
    elif path.startswith(_VIRTUAL_ROOT):
        scope = path.removeprefix(_VIRTUAL_ROOT).rstrip("/")
        if not _SCOPE.fullmatch(scope):
            raise ValueError("工具输出目录无效")
    else:
        raise ValueError("工具输出目录无效")
    workspace = workspace_root()
    root = workspace / _DIRECTORY
    directory = root / scope if scope else root
    if any(_link(item) for item in (workspace, root, directory)):
        raise ValueError("工具输出目录不能是链接")
    if not directory.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("工具输出路径越界")
    if not directory.is_dir():
        return "(无工具结果)"
    entries = sorted(item.name for item in directory.iterdir() if not _link(item) and
                     ((_SCOPE.fullmatch(item.name) and item.is_dir()) if scope is None
                      else (_FILE.fullmatch(item.name) and item.is_file())))
    rows = [f"{path.rstrip('/')}/{name}" for name in entries[offset:offset + limit]]
    if offset + limit < len(entries):
        rows.append(f"[继续读取：offset={offset + limit}]")
    return "\n".join(rows) or "(无工具结果)"


def clear(entries: list[dict]) -> None:
    for entry in entries:
        try:
            path = _archive_path(entry["scope"], entry["filename"])
        except (KeyError, ValueError):
            logger.warning("跳过无法确认归属的工具输出")
            continue
        if not path.exists():
            continue
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest == entry.get("digest"):
            path.unlink()
