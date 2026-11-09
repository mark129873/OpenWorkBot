import asyncio
import json
import logging
import math
import re
import time
from datetime import UTC, datetime

from app import db
from app.agent.memory_recall import _snapshot
from app.agent.memory_store import MemoryRevisionConflict, _advance, _delete, _read_files, _state, _store
from app.models import User, new_id
from app.user_context import bind_user, require_user


logger = logging.getLogger(__name__)
_DEBOUNCE = 30
_MAX_WAIT = 300
_MAX_FACTS = 100
_MAX_PENDING = 1000
_SHUTDOWN_WAIT = 30
_BATCH_DELAY = 0.5
_TRIVIAL = re.compile(r"(?:嗯+|ok(?:ay)?|好的[呢]?|好[的呀]?|谢谢|感谢|多谢|对|是的|收到|明白|了解|知道了|got it|thanks|thank you|cheers)", re.I)
_SIGNALS = {name: re.compile(pattern, re.I) for name, pattern in {
    "correction": r"\bthat(?:'s| is) (?:wrong|incorrect)\b|\byou misunderstood\b|\btry again\b|\bredo\b|不对|你理解错了|你理解有误|重试|重新来|换一种|改用",
    "reinforcement": r"\byes[,.]?\s+(?:exactly|perfect|that(?:'s| is) (?:right|correct|it))\b|\bperfect(?:[.!?]|$)|\bexactly\s+(?:right|correct)\b|\bthat(?:'s| is)\s+(?:exactly\s+)?(?:right|correct|what i (?:wanted|needed|meant))\b|\bkeep\s+(?:doing\s+)?that\b|\bjust\s+(?:like\s+)?(?:that|this)\b|\bthis is (?:great|helpful)\b(?:[.!?]|$)|\bthis is what i wanted\b(?:[.!?]|$)|对[，,]?\s*就是这样(?:[。！？!?.]|$)|完全正确(?:[。！？!?.]|$)|(?:对[，,]?\s*)?就是这个意思(?:[。！？!?.]|$)|正是我想要的(?:[。！？!?.]|$)|继续保持(?:[。！？!?.]|$)|(?:以后|今后)(?:都|一直)?保持这样(?:[。！？!?.]|$)",
    "preference": r"\bI (?:really )?(?:prefer|like|love|enjoy|favor|hate|dislike|don.t like|cannot stand)\b|\bmy favorite\b|\bI.d rather\b|我喜欢|我偏好|我更喜欢|我偏爱|我讨厌|我不喜欢|我钟爱|我最喜欢",
    "identity": r"\bI am (?:a |an )?[a-z][a-z -]{2,}\b|\bI work as\b|\bI.m (?:a |an )?[a-z][a-z -]{2,}\b|\bmy job is\b|我是|我的职业是|我的工作是|我担任|我是一名",
    "goal": r"\bI (?:plan to|want to|am going to|aim to|intend to)\b|\bmy goal is\b|\bI.m going to\b|\bnext I.ll\b|我的目标是|我打算|我准备|接下来我要|接下来我打算|我想做",
    "decision": r"\blet.s (?:go with|use|pick|choose)\b|\bI.ll (?:go with|use|pick|choose)\b|\bwe (?:should|will) (?:go with|use|pick|choose)\b|\bI (?:decide|chose|selected) to\b|就用|决定用|决定采用|我们选|我选|就采用",
}.items()}
_HINTS = {
    "correction": "检测到纠错；仅长期用户偏好可记录为 correction，明确纠错的置信度可至少为 0.95。",
    "reinforcement": "检测到正面反馈；仅用户长期认可的方式可记录。",
    "preference": "检测到偏好；仅跨任务持续有效的用户偏好可记录。",
    "identity": "检测到身份；仅跨任务持续有效的用户身份可记录。",
    "goal": "检测到目标；仅跨任务持续有效的用户目标可记录。",
    "decision": "检测到决定；仅跨任务持续有效的用户决定可记录。",
}
_PROMPT = """从对话提取跨任务长期有用的用户信息，不猜测。项目专属信息、临时要求和本次授权不可写入用户记忆。
只返回 JSON 对象，包含 user、history、newFacts、factsToRemove。
user 可更新 workContext、personalContext、topOfMind、cognitiveStyle；history 可更新 recentMonths、earlierContext、longTermBackground。
每个需要更新的摘要包含 summary、shouldUpdate=true、scope=user、authority=descriptive；无变化可省略。
摘要长度建议：workContext 2-3 句；personalContext 1-2 句；topOfMind 3-5 句；cognitiveStyle 2-4 句；
recentMonths 4-6 句或 1-2 段；earlierContext 3-5 句或 1 段；longTermBackground 2-4 句。
摘要须非常简洁且不失信息；更新后的七类摘要正文（包括未更新的已有摘要）合计不得超过 1000 token；必要时精简并更新相关摘要，保留跨任务长期有效的重要信息。
newFacts 每项包含 content、category、confidence（0 到 1）、scope、durability、authority；
每条事实 content 须非常简洁且不失信息，不得超过 1000 token。
只保存 scope=user、durability=durable、authority=descriptive 的事实。category 可为 preference、knowledge、context、behavior、cognitive、goal、correction。
factsToRemove 仅包含用户明确更正或撤回的已有事实，每项包含 id、scope=user、reason；若删除依赖新事实，replacementFactIndex 指向 newFacts 的下标。
仅在 correction 的错误明确时记录 sourceError；不得把当前任务的纠错写成长期用户偏好。
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
    watermark = state.get("main")
    if isinstance(watermark, list) and len(watermark) == 2 and watermark[0] == "id" and isinstance(watermark[1], str):
        watermark = watermark[1]
        start = next((index + 1 for index, row in enumerate(records) if row.get("id") == watermark), 0)
    result = []
    eligible = False
    skip_reply = False
    for index, row in enumerate(records):
        identity, role, content = row.get("id", ""), row.get("role"), row.get("content")
        if role == "user" and identity.startswith("input:"):
            event = events.get(identity[6:])
            eligible = bool(event)
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


def _signals(messages: list[dict]) -> set[str]:
    return {name for name, pattern in _SIGNALS.items() for message in messages[-6:]
            if message.get("role") == "user" and pattern.search(message.get("content", ""))}


def _parse(text: str) -> dict:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            result, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and {"user", "history", "newFacts"}.issubset(result):
            return result
    raise ValueError("记忆提取结果无效")


def _fact_confidence(fact) -> float | None:
    if not isinstance(fact, dict) or not isinstance(fact.get("content"), str) or not fact["content"].strip():
        return None
    confidence = fact.get("confidence", 0.5)
    if isinstance(confidence, bool):
        return None
    try:
        confidence = float(confidence)
    except (TypeError, ValueError, OverflowError):
        return None
    return confidence if math.isfinite(confidence) and 0 <= confidence <= 1 else None


def _apply(summary: dict, facts: list[dict], result: dict) -> tuple[dict, list[dict]]:
    if (not isinstance(result.get("user"), dict) or not isinstance(result.get("history"), dict)
            or not isinstance(result.get("newFacts"), list)):
        raise ValueError("记忆提取结果不完整")
    proposals = result["newFacts"]
    removals = result.get("factsToRemove", [])
    if not isinstance(removals, list):
        raise ValueError("记忆删除结果无效")
    if removals and any(_fact_confidence(fact) is None for fact in proposals):
        raise ValueError("删除请求包含格式无效的新事实")
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for section in ("user", "history"):
        updates = result.get(section)
        if not isinstance(updates, dict):
            continue
        for name, item in updates.items():
            if (name in summary[section] and isinstance(item, dict) and item.get("shouldUpdate") is True
                    and _label(item.get("scope")) == "user" and _label(item.get("authority")) == "descriptive"
                    and isinstance(item.get("summary"), str) and item["summary"].strip()
                    and item["summary"].strip() != summary[section][name]["summary"]):
                summary[section][name] = {"summary": item["summary"].strip(), "updatedAt": now}

    by_id = {fact["id"]: fact for fact in facts}
    accepted = {}
    for index, fact in enumerate(proposals):
        if (not isinstance(fact, dict) or _label(fact.get("scope")) != "user" or _label(fact.get("durability")) != "durable"
                or _label(fact.get("authority")) != "descriptive" or not isinstance(fact.get("content"), str)):
            continue
        confidence = _fact_confidence(fact)
        if confidence is None or confidence < 0.7:
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
            if item["category"] == "correction" and isinstance(fact.get("sourceError"), str) and fact["sourceError"].strip():
                item["sourceError"] = fact["sourceError"].strip()
            by_id[identity] = item
        accepted[index] = identity

    kept = sorted(by_id.values(), key=lambda item: item["confidence"], reverse=True)[:_MAX_FACTS]
    kept_ids = {fact["id"] for fact in kept}
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
        self.compactions: list[tuple[str, dict]] = []
        self.event = asyncio.Event()
        self.worker: asyncio.Task | None = None
        self.draining = False
        self.locks: dict[str, asyncio.Lock] = {}

    async def snapshot(self, query: str) -> str:
        user_id = require_user()
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            return await _io(_snapshot, user_id, query)

    def enqueue(self, client, records: list[dict], events: dict[str, dict], generation: int, *, role: str) -> None:
        if role != "main":
            raise PermissionError("仅主 Agent 可以写入记忆")
        user_id = require_user()
        if self.draining:
            return
        now = time.monotonic()
        old = self.pending.get(user_id)
        signals = _signals(_conversation(records, events, {}))
        if old is None and len(self.pending) >= _MAX_PENDING and not signals:
            logger.warning("长期记忆队列已满，等待下次处理 user=%s", user_id)
            return
        first = old["first"] if old else now
        due = min(now + _DEBOUNCE, first + _MAX_WAIT)
        if old and old["due"] <= now:
            due = old["due"]
        self.pending.pop(user_id, None)
        self.pending[user_id] = {"client": client, "generation": generation, "records": records, "events": events,
                                 "first": first, "due": due, "signals": signals | (old["signals"] if old else set())}
        self.event.set()
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._worker())

    def enqueue_compaction(self, client, records: list[dict], events: dict[str, dict], generation: int) -> None:
        if self.draining:
            return
        user_id = require_user()
        self.compactions.append((user_id, {"client": client, "generation": generation, "records": records,
                                           "events": events, "signals": _signals(_conversation(records, events, {}))}))
        self.event.set()
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._worker())

    def discard_pending(self, user_id: str) -> None:
        self.pending.pop(user_id, None)
        self.compactions = [item for item in self.compactions if item[0] != user_id]
        self.event.set()

    async def clear(self) -> None:
        user_id = require_user()
        self.discard_pending(user_id)
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            await _io(_delete, user_id)

    async def close(self) -> None:
        self.draining = True
        self.event.set()
        if (self.pending or self.compactions) and (self.worker is None or self.worker.done()):
            self.worker = asyncio.create_task(self._worker())
        if self.worker is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.worker), timeout=_SHUTDOWN_WAIT)
            except asyncio.TimeoutError:
                self.worker.cancel()
                await asyncio.gather(self.worker, return_exceptions=True)

    async def _worker(self) -> None:
        while self.pending or self.compactions:
            self.event.clear()
            now = time.monotonic()
            due = [user_id for user_id, item in self.pending.items() if self.draining or item["due"] <= now]
            if not due and not self.compactions:
                wait = max(0, min(item["due"] for item in self.pending.values()) - now)
                try:
                    await asyncio.wait_for(self.event.wait(), timeout=wait)
                except asyncio.TimeoutError:
                    pass
                continue
            batches = [(user_id, batch, True) for user_id, batch in self.compactions]
            self.compactions = []
            batches.extend((user_id, self.pending.pop(user_id), False) for user_id in due)
            for user_id, batch, bypass in batches:
                try:
                    with bind_user(user_id):
                        args = (user_id, batch["client"], batch["records"], batch["events"],
                                batch["generation"], batch["signals"])
                        if bypass:
                            await self._extract(*args, bypass_watermark=True)
                        else:
                            await self._extract(*args)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("长期记忆提取失败 user=%s", user_id)
                if len(batches) > 1 and not self.draining:
                    await asyncio.sleep(_BATCH_DELAY)

    async def _extract(self, user_id: str, client, records: list[dict], events: dict[str, dict], generation: int,
                       signals: set[str] | frozenset[str] = frozenset(), *, bypass_watermark: bool = False) -> None:
        with db.new_session() as session:
            user = session.get(User, user_id)
            if not user or user.generation != generation:
                return
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            state = await _io(_state, user_id)
            summary, facts = await _io(_read_files, user_id)
        conversation = _conversation(records, events, {})
        if not {"user", "assistant"}.issubset({item["role"] for item in conversation}):
            return
        messages = _conversation(records, events, {} if bypass_watermark else state)
        if not messages:
            return
        content = json.dumps({"memory": {**summary, "facts": facts},
                              "messages": [{"role": item["role"], "content": item["content"]} for item in messages]},
                             ensure_ascii=False)
        signals = signals | _signals(messages)
        hints = "\n".join(_HINTS[name] for name in _HINTS if name in signals)
        output = []
        async for event in client.stream(system=_PROMPT + ("\n" + hints if hints else ""),
                                         messages=[{"role": "user", "content": content}], tools=[]):
            if event.get("type") == "text_delta":
                output.append(event.get("text", ""))
            if event.get("type") == "stop_reason" and event.get("reason") == "max_tokens":
                raise ValueError("记忆提取结果不完整")
        result = _parse("".join(output))
        async with self.locks.setdefault(user_id, asyncio.Lock()):
            with db.new_session() as session:
                user = session.get(User, user_id)
                if not user or user.generation != generation:
                    return
            for attempt in range(3):
                revision = summary["revision"]
                updated, kept = _apply(summary, facts, result)
                try:
                    await _io(_store, user_id, updated, kept, revision)
                    break
                except MemoryRevisionConflict:
                    if attempt == 2:
                        raise
                    summary, facts = await _io(_read_files, user_id)
            if not bypass_watermark:
                await _io(_advance, user_id, messages[-1]["id"])
