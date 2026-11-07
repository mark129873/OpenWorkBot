import asyncio
import logging

from app import db
from app.sessions import chat_store as store

logger = logging.getLogger(__name__)


class NetworkApprovals:
    def __init__(self, runtime, provider):
        self.runtime = runtime
        self.provider = provider
        self.lock = asyncio.Lock()

    async def poll(self):
        async with self.lock:
            with db.new_session() as session:
                if store.get(session, "meta").get("reset"):
                    return
            result = await self.runtime.run_tool(self.provider.network_events, self.runtime.user_id)
            if result is None:
                return
            info, events = result
            for event in events:
                request_id = f"network:{info.generation}:{event['request_id']}"

                def save(session):
                    meta = store.get(session, "meta")
                    if meta.get("reset"):
                        return False
                    if not store.get(session, f"approval:{request_id}"):
                        host, port = event["host"], event["port"]
                        store.put(session, f"approval:{request_id}", {
                            "type": "approval", "kind": "sandbox_network", "request_id": request_id,
                            "task_id": None, "task_title": None, "tool": "网络访问",
                            "input": {"host": host, "port": port}, "scope": {"host": host, "port": port},
                            "reason": "当前用户的共享沙箱访问了未放行的域名；原命令不会自动重跑。",
                            "options": [{"option_id": "allowOnce", "label": "允许 5 分钟"},
                                        {"option_id": "allowSession", "label": "本会话内始终允许"},
                                        {"option_id": "deny", "label": "拒绝"}],
                            "generation": info.generation, "network_event_id": event["request_id"],
                            "status": "requested", "order": meta["revision"] + 1})
                    return True

                if await self.runtime.change(save, "approval"):
                    await self.runtime.run_tool(self.provider.acknowledge_network_event, info, event["request_id"])

    async def run(self):
        warned = False
        while not self.runtime.closing:
            try:
                await self.poll()
                warned = False
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("读取沙箱网络审批失败")
                if not warned:
                    await self.runtime.change(lambda session: store.add_visible(
                        session, "assistant", "共享沙箱的网络审批暂时不可用，请检查 Docker 和沙箱状态。", status="failed"))
                    warned = True
            await asyncio.sleep(1)

    async def decide(self, request_id, option_id, feedback, assistant_id):
        from app.agent.runtime import Conflict

        async with self.lock:
            def begin(session):
                self.runtime._check_assistant(session, assistant_id)
                if store.get(session, "meta").get("reset"):
                    raise Conflict("助手正在重置")
                record = store.get(session, f"approval:{request_id}")
                if not record or record.get("kind") != "sandbox_network":
                    raise Conflict("审批不存在或已失效")
                if record["status"] != "requested":
                    if (record["status"] == "resolved" and record["option_id"] == option_id
                            and record.get("feedback") == feedback):
                        return None
                    raise Conflict("审批已失效或正在处理")
                if option_id not in {option["option_id"] for option in record["options"]}:
                    raise Conflict("审批选项不属于当前请求")
                meta = store.bump(session, user_event=True)
                record.update(status="applying", option_id=option_id, feedback=feedback, user_seq=meta["user_seq"])
                store.put(session, f"approval:{request_id}", record)
                store.put(session, f"decision:{request_id}", {"type": "user_event", "event_type": "approval",
                          "id": request_id, "task_id": None, "user_seq": meta["user_seq"],
                          "option_id": option_id, "status": "processed"})
                return record

            record = await self.runtime.change(begin, "approval")
            if record is None:
                return {"ok": True, "replayed": True}

            async def apply():
                try:
                    await self.runtime.run_tool(self.provider.decide_network, self.runtime.user_id,
                                                record["generation"], [record["network_event_id"]],
                                                record["input"]["host"], record["input"]["port"], option_id)
                except Exception:
                    logger.exception("沙箱网络审批结果未知")
                    def failed(session):
                        current = store.get(session, f"approval:{request_id}")
                        if current and current["status"] == "applying":
                            current["status"] = "failed"
                            store.put(session, f"approval:{request_id}", current)
                            store.add_visible(session, "assistant", "共享沙箱网络审批结果未知，请检查后再处理；原命令不会自动重跑。", status="failed")
                    await self.runtime.change(failed, "approval")
                    raise Conflict("网络审批结果未知，请检查后再处理") from None
                def finish(session):
                    current = store.get(session, f"approval:{request_id}")
                    if current and current["status"] == "applying":
                        current["status"] = "resolved"
                        store.put(session, f"approval:{request_id}", current)
                await self.runtime.change(finish, "approval")
                return {"ok": True, "replayed": False}

            task = asyncio.create_task(apply())
            self.runtime.active_tools.add(task)
            task.add_done_callback(self.runtime.active_tools.discard)
            return await asyncio.shield(task)

    async def reset(self):
        async with self.lock:
            await self.runtime.run_tool(self.provider.reset_user, self.runtime.user_id, clear_grants=False)

    async def forget_grants(self):
        await self.runtime.run_tool(self.provider.forget_network_grants, self.runtime.user_id)
