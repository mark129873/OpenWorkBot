import asyncio
import json
import logging
import math
import re
import time
from datetime import UTC, datetime

from app import db
from app.agent.memory_recall import _snapshot
from app.agent.memory_store import _advance, _delete, _read_files, _state, _store
from app.models import User, new_id
from app.user_context import bind_user, require_user


logger = logging.getLogger(__name__)
_DEBOUNCE = 30
_MAX_WAIT = 300
_MAX_FACTS = 100
_TRIVIAL = re.compile(r"(?:嗯+|ok(?:ay)?|好的[呢]?|好[的呀]?|谢谢|感谢|多谢|对|是的|收到|明白|了解|知道了|got it|thanks|thank you|cheers)", re.I)
_PROMPT = """从对话提取跨任务长期有用的用户信息，不猜测。项目专属信息、临时要求和本次授权不可写入用户记忆。
只返回 JSON 对象，包含 user、history、newFacts、factsToRemove。
user 可更新 workContext、personalContext、topOfMind、cognitiveStyle；history 可更新 recentMonths、earlierContext、longTermBackground。
每个需要更新的摘要包含 summary、shouldUpdate=true、scope=user、authority=descriptive；无变化可省略。
newFacts 每项包含 content、category、confidence（0 到 1）、scope、durability、authority；
只保存 scope=user、durability=durable、authority=descriptive 的事实。category 可为 preference、knowledge、context、behavior、cognitive、goal、correction。
factsToRemove 仅包含用户明确更正或撤回的已有事实，每项包含 id、scope=user、reason；若删除依赖新事实，replacementFactIndex 指向 newFacts 的下标。
明确陈述的事实置信度可为 0.9 至 1，强烈暗示可为 0.7 至 0.8。不要把本次任务、文件、上传或操作授权记为长期信息。"""


async def _io(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def _conversation(records: list[dict], events: dict[str, dict], state: dict) -> list[dict]:
    start = 0
    watermark = state.get("watermark")
    if watermark:
        start = next((index + 1 for index, row in enumerate(records) if row.get("id") == watermark), 0)
    result = []
    eligible = False
    skip_reply = False
    for index, row in enumerate(records):
        identity, role, content = row.get("id", ""), row.get("role"), row.get("content")
        if role == "user" and identity.startswith("input:"):
            event = events.get(identity[6:])
            eligible = bool(event and event.get("user_seq", 0) > state["activation_seq"])
            text = event.get("user_text", "") if eligible else ""
            text = text.strip() if isinstance(text, str) else ""
            skip_reply = not text or bool(_TRIVIAL.fullmatch(text.rstrip(" \t\n\r.。,，!！?？;；")))
            if eligible and not skip_reply and index >= start:
                result.append({"id": identity, "role": "user", "content": _shorten(text)})
        elif role == "assistant" and eligible and not identity.startswith(("interrupted:", "failed:")):
            if isinstance(content, list):
                if any(isinstance(block, dict) and block.get("type") in {"tool_use", "tool_call"} for block in content):
                    continue
                text = " ".join(block.get("text", "") for block in content
                                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str))
            else:
                text = content if isinstance(content, str) else ""
            if skip_reply:
                skip_reply = False
                continue
            if text.strip() and index >= start:
                result.append({"id": identity, "role": "assistant", "content": _shorten(text.strip())})
    return result


def _shorten(text: str) -> str:
    return text if len(text) <= 1000 else text[:500] + "\n...[truncated]...\n" + text[-500:]


def _label(value) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _apply(summary: dict, facts: list[dict], result: dict) -> tuple[dict, list[dict]]:
    if (not isinstance(result.get("user"), dict) or not isinstance(result.get("history"), dict)
            or not isinstance(result.get("newFacts"), list)):
        raise ValueError("记忆提取结果不完整")
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for section in ("user", "history"):
        updates = result.get(section)
        if not isinstance(updates, dict):
            continue
        for name, item in updates.items():
            if (name in summary[section] and isinstance(item, dict) and item.get("shouldUpdate") is True
                    and _label(item.get("scope")) == "user" and _label(item.get("authority")) == "descriptive"
                    and isinstance(item.get("summary"), str)):
                summary[section][name] = {"summary": item["summary"], "updatedAt": now}

    by_id = {fact["id"]: fact for fact in facts}
    proposals = result.get("newFacts", [])
    if not isinstance(proposals, list):
        raise ValueError("记忆提取事实格式无效")
    accepted = {}
    for index, fact in enumerate(proposals):
        if (not isinstance(fact, dict) or _label(fact.get("scope")) != "user" or _label(fact.get("durability")) != "durable"
                or _label(fact.get("authority")) != "descriptive" or not isinstance(fact.get("content"), str)):
            continue
        confidence = fact.get("confidence")
        if isinstance(confidence, bool):
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(confidence) or not 0.7 <= confidence <= 1:
            continue
        content = fact["content"].strip()
        if not content:
            continue
        duplicate = next((identity for identity, item in by_id.items()
                          if item["content"].casefold() == content.casefold()), None)
        identity = duplicate or new_id()
        if not duplicate:
            category = fact.get("category")
            item = {"id": identity, "content": content,
                    "category": category.strip() if isinstance(category, str) and category.strip() else "context",
                    "confidence": float(confidence), "createdAt": now, "source": "main"}
            if isinstance(fact.get("sourceError"), str) and fact["sourceError"].strip():
                item["sourceError"] = fact["sourceError"].strip()
            by_id[identity] = item
        accepted[index] = identity

    kept = sorted(by_id.values(), key=lambda item: item["confidence"], reverse=True)[:_MAX_FACTS]
    kept_ids = {fact["id"] for fact in kept}
    removals = result.get("factsToRemove", [])
    if not isinstance(removals, list):
        raise ValueError("记忆删除结果无效")
    removal_ids = {item.get("id") for item in removals if isinstance(item, dict) and isinstance(item.get("id"), str)
                   and _label(item.get("scope")) == "user" and isinstance(item.get("reason"), str) and item["reason"].strip()}
    for item in removals:
        if (not isinstance(item, dict) or _label(item.get("scope")) != "user" or not isinstance(item.get("id"), str)
                or not isinstance(item.get("reason"), str) or not item["reason"].strip()):
            continue
        if "replacementFactIndex" in item:
            replacement = item["replacementFactIndex"]
            if (type(replacement) is not int or accepted.get(replacement) not in kept_ids
                    or accepted[replacement] == item["id"] or accepted[replacement] in removal_ids):
                continue
        by_id.pop(item["id"], None)
    return summary, sorted(by_id.values(), key=lambda item: item["confidence"], reverse=True)[:_MAX_FACTS]


class Memory:
    def __init__(self) -> None:
        self.pending: dict[str, dict] = {}
        self.events: dict[str, asyncio.Event] = {}
        self.workers: dict[str, asyncio.Task] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def initialize(self, activation_seq: int, watermark: str | None) -> None:
        user_id = require_user()
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            await _io(_state, user_id, activation_seq, watermark)

    async def snapshot(self, query: str) -> str:
        user_id = require_user()
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            return await _io(_snapshot, user_id, query)

    def enqueue(self, client, records: list[dict], events: dict[str, dict], generation: int, *, role: str) -> None:
        if role != "main":
            raise PermissionError("仅主 Agent 可以写入记忆")
        user_id = require_user()
        now = time.monotonic()
        first = self.pending.get(user_id, {}).get("first", now)
        self.pending[user_id] = {"generation": generation, "records": records, "events": events,
                                 "first": first, "last": now}
        self.events.setdefault(user_id, asyncio.Event()).set()
        if user_id not in self.workers or self.workers[user_id].done():
            self.workers[user_id] = asyncio.create_task(self._worker(user_id, client))

    def discard_pending(self, user_id: str) -> None:
        self.pending.pop(user_id, None)
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
            while user_id in self.pending:
                event.clear()
                batch = self.pending[user_id]
                wait = max(0, min(batch["last"] + _DEBOUNCE, batch["first"] + _MAX_WAIT) - time.monotonic())
                try:
                    await asyncio.wait_for(event.wait(), timeout=wait)
                    continue
                except asyncio.TimeoutError:
                    pass
                batch = self.pending.pop(user_id, None)
                if batch is None:
                    continue
                try:
                    await self._extract(user_id, client, batch["records"], batch["events"], batch["generation"])
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("长期记忆提取失败 user=%s", user_id)

    async def _extract(self, user_id: str, client, records: list[dict], events: dict[str, dict], generation: int) -> None:
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            state = await _io(_state, user_id)
            summary, facts = await _io(_read_files, user_id)
        conversation = _conversation(records, events, {**state, "watermark": None})
        if not {"user", "assistant"}.issubset({item["role"] for item in conversation}):
            return
        messages = _conversation(records, events, state)
        if not messages:
            return
        content = json.dumps({"memory": {**summary, "facts": facts},
                              "messages": [{"role": item["role"], "content": item["content"]} for item in messages]},
                             ensure_ascii=False)
        output = []
        async for event in client.stream(system=_PROMPT, messages=[{"role": "user", "content": content}], tools=[]):
            if event.get("type") == "text_delta":
                output.append(event.get("text", ""))
            if event.get("type") == "stop_reason" and event.get("reason") == "max_tokens":
                raise ValueError("记忆提取结果不完整")
        text = "".join(output).strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("记忆提取结果无效")
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            with db.new_session() as session:
                user = session.get(User, user_id)
                if not user or user.generation != generation:
                    return
            updated, kept = _apply(summary, facts, result)
            await _io(_store, user_id, updated, kept, summary["revision"])
            await _io(_advance, user_id, messages[-1]["id"])
