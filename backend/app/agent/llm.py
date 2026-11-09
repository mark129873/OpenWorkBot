"""按块聚合工具参数；块结束后才发出 tool_use。"""

import json
import logging
from typing import AsyncIterator

from anthropic import AsyncAnthropic
from jsonschema import ValidationError, validate
from openai import APIError, AsyncOpenAI

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
        self, *, system: str, messages: list, tools: list, tool_choice: dict | None = None
    ) -> AsyncIterator[dict]:
        options = {"tool_choice": tool_choice} if tool_choice is not None else {}
        async with self._client.messages.stream(
            model=self.model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=self._max_tokens,
            temperature=0,
            thinking={"type": "disabled"},
            **options,
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


def openai_messages(system: str, messages: list) -> list:
    converted = [{"role": "system", "content": system}]
    pending: set[str] = set()
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            if pending or role not in {"user", "assistant"}:
                raise ValueError("历史消息或工具结果不完整")
            converted.append({"role": role, "content": content})
        elif role == "assistant" and isinstance(content, list):
            if pending or any(block.get("type") not in {"text", "tool_use"} for block in content):
                raise ValueError("历史助手消息无效")
            text = "".join(block.get("text", "") for block in content if block["type"] == "text")
            calls = [{"id": block["id"], "type": "function", "function": {
                "name": block["name"], "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False)}}
                for block in content if block["type"] == "tool_use"]
            pending = {call["id"] for call in calls}
            if len(pending) != len(calls) or any(not call_id for call_id in pending):
                raise ValueError("历史工具调用 ID 无效")
            item = {"role": "assistant", "content": text or None}
            if calls:
                item["tool_calls"] = calls
            converted.append(item)
        elif role == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                    call_id = block["tool_use_id"]
                    output = block.get("content", "")
                    if not isinstance(output, str):
                        raise ValueError("历史工具结果无效")
                    if block.get("is_error") and not output.startswith("Error:"):
                        output = f"Error: {output}"
                    converted.append({"role": "tool", "tool_call_id": call_id, "content": output})
                    pending.remove(call_id)
                elif block.get("type") == "text" and not pending:
                    converted.append({"role": "user", "content": block.get("text", "")})
                else:
                    raise ValueError("历史工具结果无效或未配对")
        else:
            raise ValueError("历史消息格式无效")
    if pending:
        raise ValueError("历史工具调用缺少结果")
    return converted


class OpenAIStreamClient:
    def __init__(self, s: Settings) -> None:
        self._client = AsyncOpenAI(api_key=s.openai_api_key,
                                   base_url=s.openai_base_url or "https://api.openai.com/v1")
        self.model = s.model_id
        self._max_tokens = s.max_tokens

    async def stream(
        self, *, system: str, messages: list, tools: list, tool_choice: dict | None = None
    ) -> AsyncIterator[dict]:
        schemas = {tool["name"]: tool["input_schema"] for tool in tools}
        options = {}
        if tools:
            options["tools"] = [{"type": "function", "function": {
                "name": tool["name"], "description": tool["description"],
                "parameters": tool["input_schema"], "strict": False}} for tool in tools]
        if tool_choice is not None:
            options["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}
        calls: dict[int, dict] = {}
        usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0}
        finish_reason = None
        try:
            stream = await self._client.chat.completions.create(
                model=self.model, messages=openai_messages(system, messages), max_completion_tokens=self._max_tokens,
                temperature=0, reasoning_effort="none", store=False, stream=True,
                stream_options={"include_usage": True}, **options)
            async with stream:
                async for chunk in stream:
                    if chunk.usage is not None:
                        details = chunk.usage.prompt_tokens_details
                        usage = {"input_tokens": chunk.usage.prompt_tokens, "output_tokens": chunk.usage.completion_tokens,
                                 "cache_read_tokens": getattr(details, "cached_tokens", 0) or 0,
                                 "cache_creation_tokens": getattr(details, "cache_write_tokens", 0) or 0}
                    for choice in chunk.choices:
                        if choice.index != 0:
                            continue
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                        if choice.delta.content:
                            yield {"type": "text_delta", "text": choice.delta.content}
                        if choice.delta.refusal:
                            yield {"type": "text_delta", "text": choice.delta.refusal}
                        for part in choice.delta.tool_calls or []:
                            call = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                            if part.id:
                                if call["id"] and call["id"] != part.id:
                                    raise ValueError("工具调用 ID 不一致")
                                call["id"] = part.id
                            if part.function:
                                call["name"] += part.function.name or ""
                                call["arguments"] += part.function.arguments or ""
        except APIError as exc:
            from app.agent.compact import is_context_limit_error
            status = getattr(exc, "status_code", None)
            reason = "context_length_exceeded" if is_context_limit_error(exc) else status
            raise RuntimeError(f"OpenAI 请求失败：{reason or '连接错误'}") from None
        if finish_reason == "length":
            yield {"type": "stop_reason", "reason": "max_tokens"}
        elif finish_reason not in {"stop", "tool_calls"} or bool(calls) != (finish_reason == "tool_calls"):
            raise ValueError("模型响应中断或未完成")
        else:
            parsed = []
            for index in sorted(calls):
                call = calls[index]
                try:
                    args = json.loads(call["arguments"])
                    schema = schemas[call["name"]]
                    validate(args, {**schema, "additionalProperties": False})
                except (ValueError, KeyError, ValidationError):
                    raise ValueError("模型返回了无效工具参数") from None
                if not call["id"] or any(item["id"] == call["id"] for item in parsed):
                    raise ValueError("模型返回了无效或重复的工具调用 ID")
                parsed.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": args})
            for call in parsed:
                yield call
            yield {"type": "stop_reason", "reason": "tool_use" if parsed else "end_turn"}
        yield {"type": "usage", **usage}


def build_client(s: Settings) -> AnthropicStreamClient | OpenAIStreamClient:
    if s.llm_provider == "anthropic":
        return AnthropicStreamClient(s)
    if s.llm_provider == "openai":
        return OpenAIStreamClient(s)
    raise ValueError("不支持的 LLM_PROVIDER")
