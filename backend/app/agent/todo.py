import json
from typing import Annotated

from langchain.agents.middleware import AgentState, TodoListMiddleware


PROMPT = """<todo_list_system>
你可以使用 write_todos 管理复杂的多步骤目标。

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

编写 TODO 会消耗时间和 token，应在确有帮助时使用。
</todo_list_system>"""


TOOL_DESCRIPTION = """使用此工具为复杂工作会话创建和管理结构化任务清单。每次调用全量替换当前助手的清单；空列表用于清空。

重要：通常用于需要三个及以上步骤的复杂任务。用户明确要求 TODO 清单时，即使少于三步也可以使用；其他简单请求直接完成。

何时使用：
1. 复杂的多步骤任务：任务需要三个或以上独立步骤或操作。
2. 非简单任务：需要仔细规划或多个操作。
3. 用户明确要求 TODO 清单：用户直接要求跟踪任务。
4. 多个任务：用户提供了一组需要完成的事项。
5. 动态规划：计划可能需要根据中间结果更新。

何时不使用（用户明确要求 TODO 清单时除外）：
1. 任务很直接，且少于三个步骤。
2. 任务很简单，使用清单没有帮助。
3. 请求只是对话或信息查询。
4. 需要做什么很明确，直接执行即可。

如何使用：
1. 开始任务：开始工作之前，将任务标记为 in_progress。
2. 完成任务：完成后立即将任务标记为 completed。
3. 更新清单：按需添加新任务、移除不相关任务或更新描述。
4. 多项更新：可以一次更新多个状态，例如完成一项并开始下一项。

任务状态：
- pending：任务尚未开始。
- in_progress：正在处理；并行任务可以有多项。
- completed：任务已成功完成。

任务完成要求：
重要：只有任务已完全完成时，才能标记为 completed。
以下情况不能标记为 completed：
- 还有未解决的问题或错误。
- 工作只完成了一部分，或尚未完成。
- 遇到阻碍，无法完成工作。
- 找不到必要的资源或依赖。
- 尚未达到质量标准。

如果遇到阻塞，保持该事项为 in_progress，并在清单中新建一项描述需要解决的问题。

实践建议：
- 创建具体、可执行的事项。
- 将复杂任务拆分为更小、更容易管理的步骤。
- 使用清晰、具体的任务名称。
- 工作过程中实时更新任务状态。
- 任务完成后立即标记，不要批量完成。
- 移除不再相关的任务。
- 写入 TODO 清单时，立即将第一项或前几项标记为 in_progress。
- 除非所有任务都已完成，否则始终至少保留一项 in_progress，以跟踪进展。

主动管理任务体现工作的周全性，并确保所有要求成功完成。
请记住：如果只需调用几个工具就能完成任务，且做法明确，通常直接执行；用户明确要求 TODO 清单时除外。"""


def merge_todos(existing, new):
    return existing if new is None else new


class TodoState(AgentState):
    todos: Annotated[list | None, merge_todos]


class TodoMiddleware(TodoListMiddleware):
    state_schema = TodoState

    def __init__(self):
        super().__init__(system_prompt=PROMPT, tool_description=TOOL_DESCRIPTION)

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
