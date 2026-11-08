import asyncio

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


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
                                  {"condition": asyncio.Condition(), "done": set(), "next": 0, "failed": False})
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
                    if group.get("waiting"):
                        from app.agent.worker import AgentWaiting
                        raise AgentWaiting
                    duplicate = (request.tool_call.get("name") == "write_todos"
                                 and sum(call.get("name") == "write_todos" for call in calls) > 1)
                    if group["failed"] or duplicate:
                        output = "Error: 同一轮不能多次调用 write_todos。" if duplicate else "Error: 前一项工具失败，本项未执行。"
                        skip = context.get("skip_tool_call")
                        if skip:
                            await skip(request.tool_call, context, output)
                        group["failed"] = True
                        return ToolMessage(content=output, tool_call_id=call_id, name=request.tool_call["name"], status="error")
                    result = await handler(request)
                    if isinstance(result, ToolMessage) and (result.status == "error" or str(result.content).startswith("Error:")):
                        group["failed"] = True
                    return result
                except asyncio.CancelledError:
                    group["waiting"] = True
                    raise
                except Exception as exc:
                    from app.agent.worker import AgentWaiting
                    if isinstance(exc, AgentWaiting):
                        group["waiting"] = True
                    else:
                        group["failed"] = True
                    raise
                finally:
                    finish()
        except asyncio.CancelledError:
            async with condition:
                finish()
            raise
