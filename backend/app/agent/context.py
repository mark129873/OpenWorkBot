from langchain.agents import AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import db
from app.sessions import chat_store as store


class ConversationState(AgentState):
    summary_text: str


def messages_from_records(records: list[dict]) -> list:
    messages = []
    for row in records:
        identity, role, content = row["id"], row["role"], row["content"]
        if role == "user" and isinstance(content, str):
            messages.append(HumanMessage(content=content, id=identity))
        elif role == "user" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    call_id = block["tool_use_id"]
                    messages.append(ToolMessage(content=block.get("content", ""), tool_call_id=call_id,
                                                id=f"{identity}:{call_id}",
                                                status="error" if block.get("is_error") else "success"))
        elif role == "assistant":
            if isinstance(content, str):
                messages.append(AIMessage(content=content, id=identity))
                continue
            text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
            calls = [{"id": block["id"], "name": block["name"], "args": block.get("input") or {},
                      "type": "tool_call"} for block in content if block.get("type") == "tool_use"]
            messages.append(AIMessage(content=text, tool_calls=calls, id=identity))
    return messages


def records_from_messages(messages: list, *, for_memory: bool = False) -> list[dict]:
    records = []
    for message in messages:
        identity = getattr(message, "id", None)
        if not identity:
            continue
        if isinstance(message, HumanMessage):
            records.append({"id": identity, "role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            status = message.additional_kwargs.get("memory_status")
            if status in {"pending", "interrupted", "failed"} and (for_memory or not message.tool_calls):
                continue
            if not message.content and not message.tool_calls:
                continue
            blocks = [{"type": "text", "text": message.content}] if message.content else []
            blocks.extend({"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}
                          for call in message.tool_calls)
            records.append({"id": identity, "role": "assistant", "content": blocks})
        elif isinstance(message, ToolMessage):
            records.append({"id": identity, "role": "user", "content": [{"type": "tool_result",
                            "tool_use_id": message.tool_call_id, "content": message.content,
                            **({"is_error": True} if message.status == "error" else {})}]})
    return records


def projected_history(messages: list) -> tuple[list[dict], list[list[str]]]:
    history, identities = [], []
    for row in records_from_messages(messages):
        if (row["role"] == "user" and isinstance(row["content"], list) and history
                and history[-1]["role"] == "user" and isinstance(history[-1]["content"], list)):
            history[-1]["content"].extend(row["content"])
            identities[-1].append(row["id"])
        else:
            history.append({"role": row["role"], "content": row["content"]})
            identities.append([row["id"]])
    return history, identities


def repair_main_history(messages: list) -> tuple[list, bool]:
    repaired, changed, index = [], False, 0

    def unpaired(result):
        return HumanMessage(id=result.id, content=f"此前未配对的工具结果（数据）：{result.content}",
                            additional_kwargs={"hide_from_ui": True})

    with db.new_session() as session:
        while index < len(messages):
            message = messages[index]
            if isinstance(message, AIMessage) and message.tool_calls:
                repaired.append(message)
                calls = {call["id"]: call for call in message.tool_calls}
                seen, extra = set(), []
                index += 1
                while index < len(messages) and isinstance(messages[index], ToolMessage):
                    result = messages[index]
                    if result.tool_call_id in calls and result.tool_call_id not in seen:
                        repaired.append(result)
                        seen.add(result.tool_call_id)
                    else:
                        extra.append(unpaired(result))
                        changed = True
                    index += 1
                if len(seen) < len(calls):
                    group_id = f"main:{message.id[5:]}" if message.id and message.id.startswith("step:") else None
                    group = store.get(session, f"group:{group_id}") if group_id else None
                    outputs = group.get("outputs", {}) if group and group.get("type") == "main_tool_group" else {}
                    for call_id, call in calls.items():
                        if call_id in seen:
                            continue
                        output = outputs.get(call_id)
                        if not isinstance(output, str):
                            output = "Error: 此前工具调用的结果未确认，请先检查，禁止自动重跑。"
                        repaired.append(ToolMessage(id=f"recovered:{message.id}:{call_id}", tool_call_id=call_id,
                                                    name=call["name"], content=output,
                                                    status="error" if output.startswith("Error:") else "success"))
                        changed = True
                repaired.extend(extra)
            elif isinstance(message, ToolMessage):
                repaired.append(unpaired(message))
                changed = True
                index += 1
            elif (isinstance(message, AIMessage) and not message.tool_calls
                  and message.additional_kwargs.get("memory_status") in {"pending", "interrupted", "failed"}):
                changed = True
                index += 1
            else:
                repaired.append(message)
                index += 1
    return repaired, changed


def pending(context_id: str) -> list[dict]:
    with db.new_session() as session:
        return store.records(session, "context_pending", context_id=context_id)


def confirmed(records: list[dict], messages: list) -> list[str]:
    current = {message.id: message for message in messages if getattr(message, "id", None)}
    identities = []
    for row in records:
        expected = messages_from_records([row])
        if not expected:
            continue
        def matches(item):
            actual = current.get(item.id)
            if isinstance(item, ToolMessage) and isinstance(actual, HumanMessage):
                return actual.content == f"此前未配对的工具结果（数据）：{item.content}"
            return (type(actual) is type(item) and actual.content == item.content
                    and (not isinstance(item, AIMessage) or actual.tool_calls == item.tool_calls)
                    and (not isinstance(item, ToolMessage) or (actual.tool_call_id == item.tool_call_id
                                                               and actual.status == item.status)))

        if all(matches(item) for item in expected):
            identities.append(row["id"])
    return identities


async def acknowledge(runtime, context_id: str, identities: list[str]) -> None:
    if identities:
        async with runtime.lock, runtime.graph.write_lock:
            with db.new_session() as session:
                store.clear_pending(session, context_id, identities)
                session.commit()
