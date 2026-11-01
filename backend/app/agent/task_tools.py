from app.agent.tools import TOOLS, _schema


ROUTE = _schema("route_input", "判断用户消息与任务的关系。", {
    "message_id": {"type": "string"},
    "user_seq": {"type": "integer"},
    "intents": {"type": "array", "items": {"type": "object", "properties": {
        "type": {"type": "string", "enum": ["chat", "create", "query", "update", "stop", "unclear"],
                 "description": "create 用于需要读取、搜索、分析或其他实际工作的独立新任务；chat 仅用于无需操作的简单问答。"},
        "task_id": {"type": "string"}, "instruction": {"type": "string"},
        "approval_ids": {"type": "array", "items": {"type": "string"}},
    }, "required": ["type", "instruction"]}},
    "needs_clarification": {"type": "boolean"},
}, ["message_id", "user_seq", "intents", "needs_clarification"])

MAIN_TOOLS = [
    _schema("task_create", "创建需要后台执行的任务。", {
        "title": {"type": "string"}, "goal": {"type": "string"}}, ["title", "goal"]),
    _schema("task_inspect", "查看任务状态。", {"task_id": {"type": "string"}}, ["task_id"]),
    _schema("task_update", "向任务传达修改要求。", {
        "task_id": {"type": "string"}, "instruction": {"type": "string"}}, ["task_id", "instruction"]),
    _schema("task_stop", "停止一个任务。", {"task_id": {"type": "string"}}, ["task_id"]),
    _schema("task_review", "判断一轮工作完成、继续、失败或停止。", {
        "task_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["complete", "continue", "fail", "stop"]},
        "instruction": {"type": "string"}}, ["task_id", "decision"]),
    _schema("task_wait_for", "让任务等待另一个任务的结果。", {
        "task_id": {"type": "string"}, "dependency_id": {"type": "string"}},
        ["task_id", "dependency_id"]),
]

SHARED_TOOLS = [_schema("read_tool_output", "分段读取已保存的命令输出。", {
    "output_id": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
    "limit": {"type": "integer", "minimum": 1, "maximum": 30000}}, ["output_id"])]

WORKER_TOOLS = [tool for tool in TOOLS if tool["name"] not in {"subtask"}] + SHARED_TOOLS + [
    _schema("agent_list", "查看可联系的后台助手。", {}, []),
    _schema("agent_send", "向后台助手发送信息。", {
        "context_id": {"type": "string"}, "content": {"type": "string"}}, ["context_id", "content"]),
    _schema("agent_wait", "等待另一位助手的回复。", {
        "context_id": {"type": "string"}}, ["context_id"]),
    _schema("agent_help", "请求空闲助手协助；没有空位时立即返回。", {
        "goal": {"type": "string"}}, ["goal"]),
    _schema("task_report", "向主助手报告关键进展或提出问题。", {
        "content": {"type": "string"}}, ["content"]),
]
