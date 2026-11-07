from app.agent.tools import TOOLS, _schema


MAIN_TOOLS = [
    _schema("task_create", "创建需要后台执行的任务。", {
        "title": {"type": "string"}, "goal": {"type": "string"},
        "knowledge_question": {"type": "boolean", "description": "仅当任务是单纯的知识库问答时为 true。"}}, ["title", "goal"]),
    _schema("task_inspect", "分页查看任务详情和报告；继续读取文本时保留相同 report_offset。", {
        "task_id": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
        "report_offset": {"type": "integer", "minimum": 0}}, ["task_id"]),
    _schema("task_search", "按关键词查找历史任务，关键词为空时列出最近任务；结果分页。", {
        "keyword": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}}, ["keyword"]),
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
SHARED_TOOLS += [
    _schema("state_write", "创建或更新共享状态；新建时版本填 0，更新时填读取到的版本。", {
        "state_id": {"type": "string"}, "task_id": {"type": "string"}, "kind": {"type": "string"},
        "content": {"type": "string"}, "expected_version": {"type": "integer", "minimum": 0}},
        ["kind", "content", "expected_version"]),
    _schema("state_read", "读取一条共享状态。", {"state_id": {"type": "string"}}, ["state_id"]),
    _schema("state_list", "按任务或类型分页读取共享状态。", {
        "task_id": {"type": "string"}, "kind": {"type": "string"},
        "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, []),
    _schema("agent_subscribe", "订阅协作消息主题。", {"topic": {"type": "string"}}, ["topic"]),
    _schema("agent_unsubscribe", "取消主题订阅。", {"topic": {"type": "string"}}, ["topic"]),
    _schema("agent_publish", "向当前订阅者发布消息。", {
        "topic": {"type": "string"}, "content": {"type": "string"}}, ["topic", "content"]),
]

DIRECT_TOOLS = [tool for tool in TOOLS if tool["name"] != "subtask"]
FIRST_ACTION_TOOLS = DIRECT_TOOLS + [tool for tool in MAIN_TOOLS if tool["name"] in {"task_search", "task_inspect"}] + SHARED_TOOLS

ROUTE = _schema("route_input", "判断用户消息的完整意图和首个动作。", {
    "message_id": {"type": "string"},
    "user_seq": {"type": "integer"},
    "intents": {"type": "array", "items": {"type": "object", "properties": {
        "type": {"type": "string", "enum": ["chat", "create", "query", "update", "stop", "unclear"],
                 "description": "chat 用于简单问答和预计可快速完成的简单任务；create 用于复杂、耗时或适合并行的独立任务。"},
        "task_id": {"type": "string"}, "instruction": {"type": "string"},
        "knowledge_question": {"type": "boolean", "description": "仅当 create 任务是单纯的知识库问答时为 true。"},
        "approval_ids": {"type": "array", "items": {"type": "string"},
                         "description": "受本意图影响的审批 ID，包括已决定的审批；确认无影响时为空数组。"},
    }, "required": ["type", "instruction"]}},
    "needs_clarification": {"type": "boolean"},
    "first_action": {"type": "object", "properties": {
        "intent_index": {"type": "integer", "minimum": 0, "description": "该动作对应的 intents 下标，从 0 开始。"},
        "name": {"type": "string", "enum": [tool["name"] for tool in FIRST_ACTION_TOOLS]},
        "input": {"type": "object", "description": "工具参数，必须符合提供的工具定义。"},
    }, "required": ["intent_index", "name", "input"]},
}, ["message_id", "user_seq", "intents", "needs_clarification"])

RAG_TOOLS = [
    _schema("rag_search", "查询知识库并返回答案。", {"query": {"type": "string"}}, ["query"]),
    _schema("rag_ingest", "用户明确要求时将工作区文件加入知识库或更新索引。", {"path": {"type": "string"}}, ["path"]),
    _schema("rag_list", "查看知识库中已入库的文件。", {}, []),
    _schema("rag_remove", "用户明确要求时从知识库移出文件；不删除工作区原文件。",
            {"document_id": {"type": "string"}}, ["document_id"]),
]

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
