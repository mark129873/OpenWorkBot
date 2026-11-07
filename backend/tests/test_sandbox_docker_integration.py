import asyncio
import os
import re

import pytest

from app import db
from app.agent.runtime import Runtime
from app.agent.skills import SKILLS_DIR
from app.agent.worker import Worker
from app.community.aio_sandbox.aio_sandbox_provider import AioSandboxProvider
from app.community.aio_sandbox.local_backend import LocalContainerBackend
from app.config import settings
from app.models import new_id
from app.sandbox.lease import SandboxLeaseManager
from app.sessions import chat_store as store, execution_state
from app.user_context import bind_user, require_user
from test_sandbox_runtime import group


@pytest.mark.skipif(os.environ.get("OPENWORKBOT_DOCKER_TEST") != "1", reason="需要专用 Docker 验证")
def test_real_quiet_command_uses_aio_timeout(tmp_path):
    backend = LocalContainerBackend(tmp_path / "data", tmp_path / "workspaces", SKILLS_DIR)
    provider = AioSandboxProvider(backend, tmp_path / "data")
    user = require_user()
    sandbox_id = None
    try:
        sandbox_id = provider.acquire("quiet-test", user_id=user)
        sandbox = provider.get_scoped(sandbox_id, thread_id="quiet-test", user_id=user)
        assert "quiet-done" in sandbox.execute_command_in_scope("sleep 2; echo quiet-done", timeout=5, scope_id="quiet")
        sandbox.release_command_scope("quiet")
        provider.release(sandbox_id)
        sandbox_id = None
    finally:
        if sandbox_id:
            provider.release(sandbox_id)
        provider.close()


@pytest.mark.skipif(os.environ.get("OPENWORKBOT_DOCKER_TEST") != "1", reason="需要专用 Docker 验证")
def test_real_sandbox_files_commands_and_network_grants(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "workspaces_dir", tmp_path / "workspaces")
    monkeypatch.setattr(settings, "modelio_dir", tmp_path / "logs")
    monkeypatch.setattr(settings, "bash_timeout", 30)
    db.dispose_engine()
    db.init_db(settings.data_dir)
    backend = LocalContainerBackend(settings.data_dir, settings.workspaces_dir, SKILLS_DIR)
    provider = AioSandboxProvider(backend, settings.data_dir, idle_timeout=0)

    async def run():
        leases = SandboxLeaseManager(provider)
        runtime = Runtime(client=object(), sandbox_provider=provider, sandbox_leases=leases)
        try:
            await runtime.start()
            execution_state.save_execution_state(settings.data_dir, "yolo")
            with db.new_session() as session:
                task = runtime.create_task(session, "验证沙箱", 1)
                task.update(status="running", run_id=new_id(), direct=True)
                store.put(session, f"task:{task['id']}", task)
                session.commit()
            worker = Worker(runtime, task["id"])
            worker.context_id, worker.run_id = task["context_id"], task["run_id"]

            async def call(name, args):
                group_id, identity = await group(runtime, worker, name, args)
                await worker.process_group(group_id)
                with db.new_session() as session:
                    return store.get(session, f"execution:{identity}")["output"]

            assert "已写入" in await call("write_file", {"path": "真实测试.txt", "content": "内容"})
            assert "内容" in await call("read_file", {"path": "真实测试.txt"})
            first_sandbox = runtime.direct_sandboxes[task["id"]]["sandbox_id"]
            with bind_user("b" * 32):
                other = Runtime(client=object(), sandbox_provider=provider, sandbox_leases=leases)
                try:
                    await other.start()
                    with db.new_session() as session:
                        other_task = other.create_task(session, "验证隔离", 1)
                        other_task.update(status="running", run_id=new_id(), direct=True)
                        store.put(session, f"task:{other_task['id']}", other_task)
                        session.commit()
                    other_worker = Worker(other, other_task["id"])
                    other_worker.context_id, other_worker.run_id = other_task["context_id"], other_task["run_id"]
                    other_group, other_identity = await group(other, other_worker, "read_file", {"path": "真实测试.txt"})
                    await other_worker.process_group(other_group)
                    with db.new_session() as session:
                        assert store.get(session, f"execution:{other_identity}")["output"].startswith("Error:")
                    assert other.direct_sandboxes[other_task["id"]]["sandbox_id"] != first_sandbox
                finally:
                    await other.close()
            await call("bash", {"command": "export OWB_TEST_VALUE=ok"})
            assert "ok" in await call("bash", {"command": "printf '%s' \"$OWB_TEST_VALUE\""})
            dns = await call("bash", {"command": "python -c \"import socket; socket.getaddrinfo('example.com',443)\""})
            assert dns.startswith("Error:")
            direct = await call("bash", {"command": "python -c \"import socket; socket.create_connection(('1.1.1.1',443),3)\""})
            assert direct.startswith("Error:")
            output = await call("bash", {"command": "python -c \"print('x'*25000)\""})
            assert len(output) <= 20000 and "中段已截断" in output

            await call("bash", {"command": "curl -fsSI --max-time 15 https://example.com"})
            async with asyncio.timeout(30):
                while True:
                    snapshot = await runtime.snapshot()
                    approval = next((item for item in snapshot["approvals"]
                                     if item.get("kind") == "sandbox_network"
                                     and item["input"]["host"] == "example.com"), None)
                    if approval:
                        break
                    await asyncio.sleep(0.2)
            await runtime.decide(approval["request_id"], "allowSession", assistant_id=snapshot["assistant_id"])
            assert "200" in await call("bash", {"command": "curl -fsSI --max-time 15 https://example.com"})

            background = await call("bash", {"command": "sleep 60 >/dev/null 2>&1 & echo background-pid:$!"})
            match = re.search(r"background-pid:(\d+)", background)
            assert match
            await runtime.release_direct_sandbox(worker.task_id)
            pid = match.group(1)
            check = ("python -c \"import os; p='/proc/" + pid + "/stat'; "
                     "print('alive' if os.path.exists(p) and open(p).read().split()[2] not in ['Z','X'] else 'stopped')\"")
            assert "stopped" in await call("bash", {"command": check})
            await runtime.release_direct_sandbox(worker.task_id)
            await asyncio.to_thread(provider.cleanup_idle)
            assert "200" in await call("bash", {"command": "curl -fsSI --max-time 15 https://example.com"})
            await call("bash", {"command": "exit 0"})
            assert "shell-recreated" in await call("bash", {"command": "echo shell-recreated"})
        finally:
            await runtime.close()
            await asyncio.to_thread(provider.close)

    try:
        asyncio.run(run())
    finally:
        db.dispose_engine()
