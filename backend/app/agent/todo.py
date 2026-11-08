import json
from typing import Annotated

from langchain.agents.middleware import AgentState, TodoListMiddleware


PROMPT = """你可以使用 write_todos 管理复杂的多步骤目标。

关键规则：
- 每完成一个步骤，立即将其标记为 completed，不要攒到一起批量完成。
- 同一时间保持一项 in_progress；可以并行的任务除外。
- 工作过程中实时更新清单，以跟踪执行进展。
- 简单任务或少于三个步骤的任务通常直接处理；用户明确要求 TODO 清单时除外。

适用场景：
- 三个及以上独立步骤的复杂任务。
- 需要仔细规划和执行的非简单任务。
- 用户明确要求 TODO 清单。
- 用户一次提供多个任务。
- 计划可能随着中间结果变化而调整。

不适用场景（用户明确要求 TODO 清单时除外）：
- 单一、简单、直接的任务。
- 少于三个步骤的琐碎任务。
- 纯对话或信息查询。
- 方法明确的简单工具调用。

实践建议：
- 拆分为较小、可执行的步骤。
- 使用清晰具体的事项名称。
- 移除不再相关的事项。
- 加入执行时新发现的事项。
- 根据新信息调整计划。

编写 TODO 会消耗时间和 token，应在确有帮助时使用。"""


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
