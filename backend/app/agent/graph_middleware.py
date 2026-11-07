import asyncio

from langchain.agents.middleware import AgentMiddleware


class OrderedToolMiddleware(AgentMiddleware):
    async def awrap_tool_call(self, request, handler):
        context = request.runtime.context
        if not isinstance(context, dict):
            raise ValueError("Agent 运行上下文无效")
        messages = request.state.get("messages") or []
        message = next((message for message in reversed(messages)
                        if getattr(message, "type", None) == "ai" and getattr(message, "tool_calls", None)), None)
        if not message:
            raise ValueError("工具调用缺少模型消息")
        calls = message.tool_calls
        identities = tuple(call.get("id") for call in calls)
        if any(not isinstance(identity, str) or not identity for identity in identities) or len(set(identities)) != len(identities):
            raise ValueError("工具调用 ID 无效或重复")
        call_id = request.tool_call.get("id")
        if call_id not in identities:
            raise ValueError("工具调用不属于当前消息")
        groups = context.setdefault("_ordered_tool_groups", {})
        group = groups.setdefault((message.id or id(message), identities),
                                  {"condition": asyncio.Condition(), "done": set(), "next": 0})
        condition = group["condition"]

        def finish():
            group["done"].add(call_id)
            while group["next"] < len(identities) and identities[group["next"]] in group["done"]:
                group["next"] += 1
            condition.notify_all()

        try:
            async with condition:
                while group["next"] < len(identities) and identities[group["next"]] != call_id:
                    await condition.wait()
                if call_id in group["done"] or group["next"] >= len(identities):
                    raise ValueError("工具调用已处理")
                try:
                    return await handler(request)
                finally:
                    finish()
        except asyncio.CancelledError:
            async with condition:
                finish()
            raise
