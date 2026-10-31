"""按块聚合工具参数；块结束后才发出 tool_use。"""

import json
import logging
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import Settings

logger = logging.getLogger(__name__)


class AnthropicStreamClient:
    def __init__(self, s: Settings) -> None:
        kwargs: dict = {"api_key": s.anthropic_api_key}
        if s.anthropic_base_url:
            kwargs["base_url"] = s.anthropic_base_url
        self._client = AsyncAnthropic(**kwargs)
        self.model = s.model_id
        self._max_tokens = s.max_tokens

    async def stream(
        self, *, system: str, messages: list, tools: list
    ) -> AsyncIterator[dict]:
        async with self._client.messages.stream(
            model=self.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=self._max_tokens,
            thinking={"type": "disabled"},
        ) as stream:
            tool_meta: dict[int, dict] = {}
            json_fragments: dict[int, str] = {}
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
            }

            async for event in stream:
                etype = event.type
                if etype == "message_start":
                    # 输入用量包含缓存读写。
                    msg_usage = getattr(event.message, "usage", None)
                    if msg_usage is not None:
                        usage["input_tokens"] = getattr(msg_usage, "input_tokens", 0) or 0
                        usage["cache_read_tokens"] = getattr(msg_usage, "cache_read_input_tokens", 0) or 0
                        usage["cache_creation_tokens"] = getattr(msg_usage, "cache_creation_input_tokens", 0) or 0
                elif etype == "content_block_start":
                    block = event.content_block
                    if getattr(block, "type", "") == "tool_use":
                        tool_meta[event.index] = {"id": block.id, "name": block.name}
                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text_delta", "text": delta.text}
                    elif delta.type == "input_json_delta":
                        json_fragments[event.index] = (
                            json_fragments.get(event.index, "") + delta.partial_json
                        )
                elif etype == "content_block_stop":
                    meta = tool_meta.pop(event.index, None)
                    if meta is not None:
                        raw = json_fragments.pop(event.index, "")
                        try:
                            args = json.loads(raw) if raw else {}
                        except ValueError:
                            logger.warning("tool_use 参数 JSON 解析失败，按空参数处理")
                            args = {}
                        yield {
                            "type": "tool_use",
                            "id": meta["id"],
                            "name": meta["name"],
                            "input": args,
                        }
                elif etype == "message_delta":
                    msg_usage = getattr(event, "usage", None)
                    if msg_usage is not None:
                        usage["output_tokens"] = getattr(msg_usage, "output_tokens", 0) or 0
                    reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                    if reason:
                        yield {"type": "stop_reason", "reason": reason}

            yield {
                "type": "usage",
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cache_read_tokens": usage["cache_read_tokens"],
                "cache_creation_tokens": usage["cache_creation_tokens"],
            }


def build_client(s: Settings) -> AnthropicStreamClient:
    return AnthropicStreamClient(s)
