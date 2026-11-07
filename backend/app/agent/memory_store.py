import json
import math
import os
import re
import shutil
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import yaml

from app.config import settings
from app.models import new_id


_SECTIONS = {
    "user": ("workContext", "personalContext", "topOfMind", "cognitiveStyle"),
    "history": ("recentMonths", "earlierContext", "longTermBackground"),
}


def empty_memory() -> dict:
    return {"version": "2.0", "revision": 0, "lastUpdated": "", **{
        section: {name: {"summary": "", "updatedAt": ""} for name in names}
        for section, names in _SECTIONS.items()}}


def _directory(user_id: str) -> Path:
    root = (settings.data_dir / "memory").resolve()
    users = root / "users"
    if users.is_symlink() or (hasattr(users, "is_junction") and users.is_junction()):
        raise ValueError("记忆目录不能是链接")
    target = users / user_id
    if (target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction())
            or target.resolve().parent != users.resolve()):
        raise ValueError("记忆目录无效")
    return target


def _path(directory: Path, *parts: str) -> Path:
    current = directory
    for part in parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("记忆路径不能是链接")
    if not current.resolve().is_relative_to(directory.resolve()):
        raise ValueError("记忆路径无效")
    return current


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{new_id()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _file_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = _path(directory, ".memory.lock")
    deadline = time.monotonic() + 10
    stream = None
    while stream is None:
        try:
            stream = path.open("a+b")
            if os.name == "nt":
                import msvcrt
                stream.seek(0, os.SEEK_END)
                if not stream.tell():
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if stream is not None:
                stream.close()
                stream = None
            if time.monotonic() >= deadline:
                raise TimeoutError("记忆文件锁超时") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        if os.name == "nt":
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _fact_path(directory: Path, identity: str) -> Path:
    if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identity):
        raise ValueError("记忆事实 ID 无效")
    shard = sha256(identity.encode("utf-8")).hexdigest()[:2]
    return _path(directory, "agents", "__default__", "facts", shard, f"{identity}.md")


def _render_fact(user_id: str, fact: dict) -> str:
    metadata = {key: value for key, value in fact.items() if key not in {"content", "title", "scope"}}
    metadata.update(user_id=user_id, agent_name="__default__")
    title = str(fact.get("title") or fact["content"].splitlines()[0]).replace("\n", " ")[:160]
    return f"---\n{yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip()}\n---\n\n# {title}\n\n{fact['content'].rstrip()}\n"


def _parse_fact(path: Path, user_id: str) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    if not text.startswith("---\n"):
        raise ValueError("记忆事实格式无效")
    front, body = text[4:].split("\n---\n", 1)
    fact = yaml.safe_load(front)
    if not isinstance(fact, dict) or fact.get("id") != path.stem:
        raise ValueError("记忆事实 ID 无效")
    if fact.pop("user_id", None) != user_id or fact.pop("agent_name", None) != "__default__":
        raise ValueError("记忆事实归属无效")
    lines = body.lstrip("\n").splitlines()
    if lines and lines[0].startswith("# "):
        fact["title"] = lines.pop(0)[2:].strip()
        if lines and not lines[0].strip():
            lines.pop(0)
    fact["content"] = "\n".join(lines).rstrip("\n")
    confidence = fact.get("confidence")
    if (not fact["content"].strip() or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("记忆事实内容无效")
    return fact


def _facts(directory: Path, user_id: str) -> list[dict]:
    root = _path(directory, "agents", "__default__", "facts")
    if not root.exists():
        return []
    result = []
    for path in root.glob("**/*.md"):
        checked = _path(directory, *path.relative_to(directory).parts)
        fact = _parse_fact(checked, user_id)
        if _fact_path(directory, fact["id"]) != checked:
            raise ValueError("记忆事实路径无效")
        result.append(fact)
    return sorted(result, key=lambda fact: fact["id"])


def _read_unlocked(directory: Path, user_id: str) -> tuple[dict, list[dict]]:
    path = _path(directory, "memory.json")
    summary = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else empty_memory()
    _validate_summary(summary)
    return summary, _facts(directory, user_id)


def _validate_summary(summary: dict) -> None:
    if not isinstance(summary, dict) or type(summary.get("revision")) is not int:
        raise ValueError("记忆摘要格式无效")
    for section, names in _SECTIONS.items():
        if not isinstance(summary.get(section), dict):
            raise ValueError("记忆摘要格式无效")
        for name in names:
            item = summary[section].get(name)
            if not isinstance(item, dict) or not isinstance(item.get("summary"), str):
                raise ValueError("记忆摘要格式无效")


def _recover(directory: Path) -> None:
    path = _path(directory, ".memory.journal.json")
    if not path.exists():
        return
    journal = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(journal, dict):
        raise ValueError("记忆恢复日志无效")
    operation = journal.get("operation")
    if not isinstance(operation, str) or not re.fullmatch(r"[0-9a-f]+", operation):
        raise ValueError("记忆恢复日志无效")
    backup = _path(directory, ".recovery", operation)
    if journal.get("state") == "prepared":
        summary = _path(backup, "memory.json")
        target = _path(directory, "memory.json")
        old_ids = journal.get("old_ids")
        fact_ids = journal.get("fact_ids")
        if (type(journal.get("had_summary")) is not bool or not isinstance(old_ids, list) or not isinstance(fact_ids, list)
                or any(not isinstance(identity, str) for identity in old_ids + fact_ids)
                or len(set(fact_ids)) != len(fact_ids) or not set(old_ids).issubset(fact_ids)):
            raise ValueError("记忆恢复日志无效")
        for identity in fact_ids:
            _fact_path(directory, identity)
            if identity in old_ids:
                _parse_fact(_path(backup, f"{identity}.md"), directory.name)
        if journal["had_summary"]:
            previous = summary.read_text(encoding="utf-8")
            if not isinstance(json.loads(previous), dict):
                raise ValueError("记忆恢复摘要无效")
            _atomic_write(target, previous)
        else:
            target.unlink(missing_ok=True)
        for identity in fact_ids:
            target = _fact_path(directory, identity)
            if identity in old_ids:
                source = _path(backup, f"{identity}.md")
                _atomic_write(target, source.read_text(encoding="utf-8"))
            else:
                target.unlink(missing_ok=True)
    elif journal.get("state") != "committed":
        raise ValueError("记忆恢复日志无效")
    if backup.exists():
        shutil.rmtree(backup)
    path.unlink(missing_ok=True)


def _read_files(user_id: str) -> tuple[dict, list[dict]]:
    directory = _directory(user_id)
    with _file_lock(directory):
        _recover(directory)
        return _read_unlocked(directory, user_id)


def _store(user_id: str, summary: dict, facts: list[dict], expected_revision: int | None = None) -> None:
    directory = _directory(user_id)
    _validate_summary(summary)
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("记忆事实无效")
        _fact_path(directory, fact.get("id"))
        confidence = fact.get("confidence")
        if (not isinstance(fact.get("content"), str) or not fact["content"].strip() or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("记忆事实内容无效")
    with _file_lock(directory):
        _recover(directory)
        current, old_facts = _read_unlocked(directory, user_id)
        if expected_revision is not None and current["revision"] != expected_revision:
            raise ValueError("记忆已更新，请重试")
        old = {fact["id"]: fact for fact in old_facts}
        desired = {fact["id"]: fact for fact in facts}
        if len(desired) != len(facts):
            raise ValueError("记忆事实 ID 重复")
        changed = [identity for identity in set(old) | set(desired)
                   if identity not in old or identity not in desired or old[identity] != desired[identity]]
        operation = new_id()
        backup = _path(directory, ".recovery", operation)
        journal_path = _path(directory, ".memory.journal.json")
        summary_path = _path(directory, "memory.json")
        journal = {"operation": operation, "state": "prepared", "had_summary": summary_path.exists(),
                   "fact_ids": changed, "old_ids": [identity for identity in changed if identity in old]}
        try:
            if summary_path.exists():
                _atomic_write(_path(backup, "memory.json"), summary_path.read_text(encoding="utf-8"))
            for identity in journal["old_ids"]:
                _atomic_write(_path(backup, f"{identity}.md"), _fact_path(directory, identity).read_text(encoding="utf-8"))
            _atomic_write(journal_path, json.dumps(journal, ensure_ascii=False))
            for identity in changed:
                target = _fact_path(directory, identity)
                if identity in desired:
                    _atomic_write(target, _render_fact(user_id, desired[identity]))
                else:
                    target.unlink(missing_ok=True)
            summary = {**summary, "version": "2.0", "revision": current["revision"] + 1,
                       "lastUpdated": datetime.now(UTC).isoformat().replace("+00:00", "Z")}
            _atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))
            journal["state"] = "committed"
            _atomic_write(journal_path, json.dumps(journal, ensure_ascii=False))
            _recover(directory)
        except BaseException:
            if journal_path.exists():
                _recover(directory)
            elif backup.exists():
                shutil.rmtree(backup)
            raise


def _state(user_id: str, activation_seq: int | None = None, watermark: str | None = None) -> dict:
    directory = _directory(user_id)
    with _file_lock(directory):
        _recover(directory)
        path = _path(directory, "agents", "__default__", ".metadata", "watermark.json")
        if not path.exists():
            if activation_seq is None:
                raise ValueError("记忆提取水位未初始化")
            if _path(directory, "memory.json").exists():
                raise ValueError("记忆提取水位丢失")
            value = {"activation_seq": activation_seq, "watermark": watermark}
            _atomic_write(path, json.dumps(value, ensure_ascii=False))
            return value
        value = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(value, dict) or type(value.get("activation_seq")) is not int
                or value.get("watermark") is not None and not isinstance(value["watermark"], str)):
            raise ValueError("记忆提取水位无效")
        return value


def _advance(user_id: str, watermark: str) -> None:
    directory = _directory(user_id)
    with _file_lock(directory):
        path = _path(directory, "agents", "__default__", ".metadata", "watermark.json")
        value = json.loads(path.read_text(encoding="utf-8"))
        value["watermark"] = watermark
        _atomic_write(path, json.dumps(value, ensure_ascii=False))


def _delete(user_id: str) -> None:
    directory = _directory(user_id)
    if not directory.exists():
        return
    with _file_lock(directory):
        for name in ("memory.json", ".memory.journal.json"):
            _path(directory, name).unlink(missing_ok=True)
        for name in ("agents", ".recovery"):
            target = _path(directory, name)
            if target.exists():
                shutil.rmtree(target)
