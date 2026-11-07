import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path

import jieba
import tiktoken

from app import db
from app.config import settings
from app.models import User, new_id
from app.user_context import bind_user, require_user


logger = logging.getLogger(__name__)
_DEBOUNCE = 30
_MAX_FACTS = 100
_MAX_TOKENS = 2000


def _directory(user_id: str) -> Path:
    parent = (settings.data_dir / "memory").resolve()
    target = parent / user_id
    if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
        raise ValueError("记忆目录不能是链接")
    if target.resolve().parent != parent:
        raise ValueError("记忆目录无效")
    return target


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{new_id()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_files(user_id: str) -> tuple[str, list[dict]]:
    directory = _directory(user_id)
    summary_file, facts_file = directory / "summary.json", directory / "facts.md"
    if summary_file.is_symlink() or facts_file.is_symlink():
        raise ValueError("记忆文件不能是链接")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))["summary"] if summary_file.exists() else ""
    facts = []
    if facts_file.exists():
        for line in facts_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("- "):
                facts.append(json.loads(line[2:]))
    if not isinstance(summary, str) or any(not isinstance(item, dict) for item in facts):
        raise ValueError("记忆文件格式无效")
    return summary, facts


def _tokens(text: str) -> list[str]:
    return list(dict.fromkeys(piece.strip() for piece in jieba.cut_for_search(text) if piece.strip()))


def _index(user_id: str, facts: list[dict], query: str = "") -> list[str]:
    index_path = settings.data_dir / "memory-index.sqlite"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(("search:" + serialized).encode("utf-8")).hexdigest()
    with sqlite3.connect(index_path, timeout=5) as conn:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(user_id UNINDEXED, fact_id UNINDEXED, terms)")
        conn.execute("CREATE TABLE IF NOT EXISTS memory_index_version (user_id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
        row = conn.execute("SELECT digest FROM memory_index_version WHERE user_id=?", (user_id,)).fetchone()
        if not row or row[0] != digest:
            conn.execute("DELETE FROM memory_fts WHERE user_id=?", (user_id,))
            conn.executemany("INSERT INTO memory_fts(user_id, fact_id, terms) VALUES (?, ?, ?)",
                             [(user_id, fact["id"], " ".join(_tokens(fact["content"]))) for fact in facts])
            conn.execute("INSERT OR REPLACE INTO memory_index_version(user_id, digest) VALUES (?, ?)",
                         (user_id, digest))
        terms = _tokens(query)[:64]
        if not terms:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        return [row[0] for row in conn.execute(
            "SELECT fact_id FROM memory_fts WHERE user_id=? AND memory_fts MATCH ? ORDER BY bm25(memory_fts) LIMIT 100",
            (user_id, expression))]


def _snapshot(user_id: str, query: str) -> str:
    summary, facts = _read_files(user_id)
    if not summary and not facts:
        return ""
    ranked_ids = _index(user_id, facts, query)
    rank = {identity: index for index, identity in enumerate(ranked_ids)}
    facts.sort(key=lambda item: (rank.get(item["id"], len(rank)), -float(item.get("confidence", 0))))
    text = "用户摘要：" + summary + "\n事实：\n" + "\n".join(f"- {item['content']}" for item in facts)
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text, disallowed_special=())
    return encoding.decode(tokens[:_MAX_TOKENS], errors="ignore")


def _store(user_id: str, summary: str, facts: list[dict]) -> None:
    directory = _directory(user_id)
    _atomic_write(directory / "summary.json", json.dumps({"summary": summary}, ensure_ascii=False))
    lines = ["# 用户事实"] + ["- " + json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in facts]
    _atomic_write(directory / "facts.md", "\n".join(lines) + "\n")
    _index(user_id, facts)


def _delete(user_id: str) -> None:
    directory = _directory(user_id)
    for name in ("summary.json", "facts.md"):
        path = directory / name
        if path.is_symlink():
            raise ValueError("记忆文件不能是链接")
        path.unlink(missing_ok=True)
    index_path = settings.data_dir / "memory-index.sqlite"
    if index_path.exists():
        with sqlite3.connect(index_path, timeout=5) as conn:
            conn.execute("DELETE FROM memory_fts WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM memory_index_version WHERE user_id=?", (user_id,))


async def _io(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


class Memory:
    def __init__(self) -> None:
        self.pending: dict[str, list[tuple[int, list[dict]]]] = {}
        self.events: dict[str, asyncio.Event] = {}
        self.workers: dict[str, asyncio.Task] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def snapshot(self, query: str) -> str:
        user_id = require_user()
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            return await _io(_snapshot, user_id, query)

    def enqueue(self, client, messages: list[dict], generation: int, *, role: str) -> None:
        if role != "main":
            raise PermissionError("仅主 Agent 可以写入记忆")
        user_id = require_user()
        self.pending.setdefault(user_id, []).append((generation, messages))
        self.events.setdefault(user_id, asyncio.Event()).set()
        if user_id not in self.workers or self.workers[user_id].done():
            self.workers[user_id] = asyncio.create_task(self._worker(user_id, client))

    def discard_pending(self, user_id: str) -> None:
        self.pending[user_id] = []
        self.events.setdefault(user_id, asyncio.Event()).set()

    async def clear(self) -> None:
        user_id = require_user()
        self.discard_pending(user_id)
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            await _io(_delete, user_id)

    async def close(self) -> None:
        for task in self.workers.values():
            task.cancel()
        await asyncio.gather(*self.workers.values(), return_exceptions=True)

    async def _worker(self, user_id: str, client) -> None:
        with bind_user(user_id):
            event = self.events[user_id]
            while self.pending.get(user_id):
                event.clear()
                while True:
                    try:
                        await asyncio.wait_for(event.wait(), timeout=_DEBOUNCE)
                        event.clear()
                        if not self.pending.get(user_id):
                            return
                    except asyncio.TimeoutError:
                        break
                batch, self.pending[user_id] = self.pending[user_id], []
                if not batch:
                    continue
                generation = batch[-1][0]
                messages = [message for item_generation, items in batch if item_generation == generation for message in items]
                try:
                    await self._extract(user_id, client, messages, generation)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("长期记忆提取失败 user=%s", user_id)

    async def _extract(self, user_id: str, client, messages: list[dict], generation: int) -> None:
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            summary, facts = await _io(_read_files, user_id)
        prompt = ("从对话提取长期有用的用户信息，勿猜测。只返回 JSON 对象：summary 为更新后的简短摘要，"
                  "facts 为数组，每项包含 content、0 到 1 的 confidence、category，可用已有 id 更新事实；"
                  "remove_ids 为应删除的已有事实 ID。无变化时保留原内容。")
        content = json.dumps({"summary": summary, "facts": facts, "messages": messages}, ensure_ascii=False)
        model_messages = [{"role": "user", "content": content}]
        output = []
        async for event in client.stream(system=prompt, messages=model_messages, tools=[]):
            if event.get("type") == "text_delta":
                output.append(event.get("text", ""))
            if event.get("type") == "stop_reason" and event.get("reason") == "max_tokens":
                raise ValueError("记忆提取结果不完整")
        text = "".join(output).strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        if not isinstance(result, dict) or not isinstance(result.get("facts", []), list):
            raise ValueError("记忆提取结果无效")
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            with db.new_session() as session:
                user = session.get(User, user_id)
                if not user or user.generation != generation:
                    return
            current_summary, current = await _io(_read_files, user_id)
            by_id = {item["id"]: item for item in current}
            removed = result.get("remove_ids", [])
            for identity in removed if isinstance(removed, list) else []:
                if isinstance(identity, str):
                    by_id.pop(identity, None)
            for fact in result.get("facts", []):
                if not isinstance(fact, dict) or not isinstance(fact.get("content"), str):
                    continue
                confidence = fact.get("confidence")
                if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not 0.7 <= confidence <= 1:
                    continue
                value = fact["content"].strip()
                if not value:
                    continue
                identity = fact.get("id") if fact.get("id") in by_id else new_id()
                duplicate = next((key for key, item in by_id.items() if item["content"].casefold() == value.casefold()), None)
                identity = duplicate or identity
                by_id[identity] = {"id": identity, "content": value, "confidence": float(confidence),
                                   "category": str(fact.get("category") or "fact")}
            kept = sorted(by_id.values(), key=lambda item: item["confidence"], reverse=True)[:_MAX_FACTS]
            updated_summary = result.get("summary")
            await _io(_store, user_id, updated_summary if isinstance(updated_summary, str) else current_summary, kept)
