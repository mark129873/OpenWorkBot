import asyncio
from types import SimpleNamespace

from app.agent.permission_service import check_permission
from app.sessions import approvals, session_permissions
from app.sessions.approvals import build_options


def test_sandbox_commands_always_ask_in_build_and_edit():
    allow = {"allow": [{"toolName": "bash"}]}
    for mode in ("build", "edit"):
        for command in ("pwd", "rm file"):
            decision = check_permission(mode, allow, "bash", {"command": command}, sandboxed=True)
            assert decision["decision"] == "ask"
    assert [item["option_id"] for item in build_options("bash", {"command": "pwd"})] == ["allowOnce", "deny"]


def test_sandbox_commands_respect_deny_and_yolo_without_enabling_host_commands():
    rules = {"deny": [{"toolName": "bash"}]}
    args = {"command": "pwd"}
    assert check_permission("build", rules, "bash", args, sandboxed=True)["decision"] == "deny"
    assert check_permission("yolo", rules, "bash", args, sandboxed=True)["decision"] == "allow"
    assert check_permission("yolo", {}, "bash", args)["decision"] == "deny"


def test_sandbox_approval_cannot_be_skipped_by_session_grant(monkeypatch):
    async def run():
        args = {"command": "pwd"}
        scope = session_permissions.scope_for("bash", args)
        session_permissions.grant("test", scope)
        calls = []

        async def approver(tool, inputs, reason):
            calls.append((tool, inputs))
            return {"approved": True}

        monkeypatch.setattr(approvals, "evaluate", lambda data, tool, inputs, **kw:
                            check_permission("build", {}, tool, inputs, **kw))
        deps = SimpleNamespace(recorder=SimpleNamespace(session_id="test"), stop_flag=lambda: False,
                               approver=approver)
        try:
            assert await approvals.authorize_tool(deps, "bash", args, sandboxed=True) is None
            assert await approvals.authorize_tool(deps, "bash", args, sandboxed=True) is None
            assert len(calls) == 2
        finally:
            session_permissions.forget_session("test")
    asyncio.run(run())
