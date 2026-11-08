import json
from typing import Annotated

from langchain.agents.middleware import AgentState, TodoListMiddleware


PROMPT = """通常在需要三个及以上步骤的复杂任务中使用 write_todos；用户明确要求列清单时，即使少于三步也可使用。简单任务可直接处理。
顺序执行时通常只保留一项 in_progress；确有并行任务时可以多项同时 in_progress。
开始步骤时标记 in_progress，完成后及时标记 completed；根据新要求调整清单，不得把未完成的事项标为完成。"""


def merge_todos(existing, new):
    return existing if new is None else new


class TodoState(AgentState):
    todos: Annotated[list | None, merge_todos]


class TodoMiddleware(TodoListMiddleware):
    state_schema = TodoState

    def __init__(self):
        super().__init__(system_prompt=PROMPT, tool_description="全量更新当前助手的 TODO 清单。")

    async def aafter_model(self, state, runtime):
        return None


def remind(messages: list, state: dict) -> list:
    todos = state.get("todos")
    if not todos or any(
        block.get("type") == "tool_use" and block.get("name") == "write_todos"
        for message in messages if isinstance(message.get("content"), list)
        for block in message["content"] if isinstance(block, dict)
    ):
        return messages
    return [*messages, {"role": "user", "content":
            "<system_reminder>此前的 TODO 已离开上下文，当前清单：\n"
            + json.dumps(todos, ensure_ascii=False) + "\n请根据工作进展继续更新。</system_reminder>"}]
