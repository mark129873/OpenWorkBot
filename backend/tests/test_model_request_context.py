import asyncio
import json

from langchain_core.messages import HumanMessage

from app import db
from app.agent import compact, shared_state
from app.agent.checkpoint import SqliteSaver
from app.agent.runtime import Runtime
from app.sessions import chat_store as store
from test_chat_runtime import accept, isolated
from test_checkpoint_context import MemorySpy, _main_state, _wait_for_reply
from test_openai import chunk, make_client, response


def test_internal_report_is_labeled_only_for_the_model():
    report = HumanMessage(id="bus:report", content="后台结果")
    state = {"messages": [HumanMessage(id="input:one", content="用户请求"), report]}
    history = compact.history_for_state(state)
    assert history[0]["content"] == "用户请求"
    assert history[1]["content"] == "内部消息（非用户直接输入）：\n后台结果"
    assert report.content == "后台结果"


def test_request_context_stays_out_of_multiturn_checkpoint(isolated):
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        last = payload["messages"][-1]
        if last["role"] == "tool":
            return response([chunk({"content": "已查到任务。"}), chunk(finish="stop")])
        if last["content"] == "查找当前任务":
            call = {"index": 0, "id": "search", "type": "function",
                    "function": {"name": "task_search", "arguments": '{"keyword":""}'}}
            return response([chunk({"tool_calls": [call]}), chunk(finish="tool_calls")])
        return response([chunk({"content": "收到。"}), chunk(finish="stop")])

    class Memory(MemorySpy):
        async def snapshot(self, query):
            return "用户偏好：简洁"

    async def run():
        client, memory = make_client(handler), Memory()
        try:
            async with SqliteSaver.from_conn_string(str(isolated / "app.db")) as saver:
                runtime = Runtime(client=client, checkpointer=saver, memory=memory)
                await runtime.start()
                try:
                    def setup(session):
                        task = runtime.create_task(session, "测试后台任务", 1)
                        task.update(status="waiting_user", run_id="run", delivery_unknown=True)
                        store.put(session, f"task:{task['id']}", task)
                        shared_state.replace(session, task["context_id"], task["id"], "run",
                                             [{"content": "正在检查", "status": "in_progress"}])
                        return task
                    task = await runtime.change(setup)
                    await accept(runtime, "one", "你好", None)
                    await _wait_for_reply(runtime, "one")
                    assert requests[0]["messages"][1]["content"].startswith("运行状态（由系统提供，仅供参考）：")
                    assert "正在检查" in requests[0]["messages"][1]["content"]
                    assert "用户偏好：简洁" in requests[0]["messages"][2]["content"]
                    assert requests[0]["messages"][-1] == {"role": "user", "content": "你好"}

                    async with runtime.graph.write_lock:
                        thread = runtime.graph.thread_id("main:loop", parent_context="main")
                    await runtime.agent_loop.graph.aupdate_state({"configurable": {"thread_id": thread}},
                                                                 {"summary_text": "以前的摘要"})
                    await runtime.change(lambda session: shared_state.replace(
                        session, task["context_id"], task["id"], "run",
                        [{"content": "检查已更新", "status": "in_progress"}]))
                    await accept(runtime, "two", "查找当前任务", None)
                    await _wait_for_reply(runtime, "two")

                    before_tool, after_tool = requests[1]["messages"], requests[2]["messages"]
                    assert "检查已更新" in before_tool[1]["content"]
                    assert "用户偏好：简洁" in before_tool[2]["content"]
                    assert "以前的摘要" in before_tool[3]["content"]
                    assert before_tool[-1] == {"role": "user", "content": "查找当前任务"}
                    assert after_tool[-2]["role"] == "assistant"
                    assert after_tool[-2]["tool_calls"][0]["id"] == "search"
                    assert after_tool[-1]["role"] == "tool" and after_tool[-1]["tool_call_id"] == "search"
                    assert "检查已更新" in after_tool[1]["content"]

                    state = await _main_state(runtime)
                    assert [m.content for m in state["messages"] if m.type == "human"] == ["你好", "查找当前任务"]
                    assert state["summary_text"] == "以前的摘要"
                    assert "<shared_state>" not in str(state["messages"])
                    assert "用户偏好：简洁" not in str(state["messages"])
                finally:
                    await runtime.close()

                runtime = Runtime(client=client, checkpointer=saver, memory=memory)
                await runtime.start()
                try:
                    assert (await accept(runtime, "one", "你好", None))["replayed"] is True
                    await accept(runtime, "three", "继续", None)
                    await _wait_for_reply(runtime, "three")
                    state = await _main_state(runtime)
                    assert [m.content for m in state["messages"] if m.type == "human"] == ["你好", "查找当前任务", "继续"]
                    assert "<shared_state>" not in str(state["messages"])
                    with db.new_session() as session:
                        assert [row["text"] for row in store.visible(session) if row["role"] == "user"] == [
                            "你好", "查找当前任务", "继续"]
                finally:
                    await runtime.close()
        finally:
            await client._client.close()

    asyncio.run(run())
