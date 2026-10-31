import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from app.modelio import record_llm_call

_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens")


@dataclass
class ModelReply:
    text: str = ""
    tool_uses: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    stop_reason: str | None = None

    def accept(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "text_delta":
            self.text += event.get("text", "")
        elif kind == "tool_use":
            self.tool_uses.append(event)
        elif kind == "usage":
            self.usage = {key: event.get(key, 0) for key in _USAGE_KEYS}
        elif kind == "stop_reason":
            self.stop_reason = event.get("reason")

    def assistant_blocks(self) -> list[dict]:
        blocks = [{"type": "text", "text": self.text}] if self.text else []
        for tool in self.tool_uses:
            blocks.append({
                "type": "tool_use", "id": tool["id"], "name": tool["name"],
                "input": tool.get("input") or {},
            })
        return blocks


@contextmanager
def record_model_step(deps, system: str, messages: list, tools: list,
                      *, subtask_id: str | None = None) -> Iterator[ModelReply]:
    """正常、异常和流中停止均记录调用快照。"""
    reply = ModelReply()
    snapshot = [dict(message) for message in messages]
    started = time.monotonic()
    error = None
    try:
        yield reply
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        record_llm_call(
            deps.recorder.session_id, deps.recorder.turn_id,
            getattr(deps.client, "model", "unknown"), system, snapshot,
            response_text=reply.text,
            tool_calls=[
                {"id": t["id"], "name": t["name"], "input": t.get("input") or {}}
                for t in reply.tool_uses
            ],
            usage=reply.usage,
            stop_reason=reply.stop_reason,
            duration_ms=(time.monotonic() - started) * 1000,
            tool_names=[t["name"] for t in tools],
            subtask_id=subtask_id,
            error=error,
        )
