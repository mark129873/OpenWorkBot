from types import SimpleNamespace

import httpx
import pytest

from app.community.aio_sandbox.aio_sandbox import AioSandbox
from app.community.aio_sandbox.sandbox_info import SandboxInfo


def response(data=None, success=True):
    return SimpleNamespace(success=success, data=data)


class Shell:
    def __init__(self):
        self.command = ""
        self.calls = 0
        self.timeout = False
        self.cleaned = []
        self.options = {}

    def create_session(self, *, id, **kwargs):
        return response(SimpleNamespace(session_id=id))

    def exec_command(self, *, command, **kwargs):
        self.calls += 1
        self.command = command
        self.options = kwargs
        if self.timeout:
            raise httpx.ReadTimeout("unavailable")
        return response(SimpleNamespace())

    def wait_for_process(self, **kwargs):
        return response(SimpleNamespace(status="completed"))

    def view(self, **kwargs):
        console = [SimpleNamespace(command=self.command, output="head" + "x" * 30000 + "tail")]
        return response(SimpleNamespace(console=console, status="completed", exit_code=0))

    def cleanup_session(self, session_id, **kwargs):
        self.cleaned.append(session_id)
        return response()


def make_sandbox(on_uncertain):
    info = SandboxInfo(sandbox_id="sandbox", sandbox_url="http://127.0.0.1:1", user_id="user",
                       generation="generation", container_id="container", proxy_id="proxy", network_id="network",
                       egress_network_id="egress", relay_token="token", request_headers={"X-DeerFlow-Relay-Token": "token"})
    sandbox = AioSandbox(info, on_uncertain=on_uncertain)
    shell = Shell()
    sandbox._client = SimpleNamespace(shell=shell)
    sandbox._bash_helper = lambda command, input_text: {"ok": True}
    return sandbox, shell


def test_command_output_keeps_head_and_tail_with_deerflow_limit():
    sandbox, shell = make_sandbox(lambda reason: None)
    try:
        output = sandbox.execute_command_in_scope("echo test", scope_id="worker")
        assert len(output) == 20000
        assert output.startswith("head") and output.endswith("tail")
        assert "中段已截断" in output
        assert shell.options["no_change_timeout"] == 600
    finally:
        sandbox.close()
    assert len(shell.cleaned) == 1


def test_idle_timeout_does_not_end_a_quiet_long_command_early():
    sandbox, shell = make_sandbox(lambda reason: None)
    try:
        sandbox.execute_command_in_scope("sleep 900", timeout=1000, scope_id="worker")
        assert shell.options["hard_timeout"] == 1000
        assert shell.options["no_change_timeout"] == 1005
    finally:
        sandbox.close()


def test_unknown_command_result_is_not_retried():
    warnings = []
    sandbox, shell = make_sandbox(warnings.append)
    shell.timeout = True
    try:
        result = sandbox.execute_command_result_in_scope("echo test", scope_id="worker")
        assert result.status == "unknown" and not result.ok
        assert shell.calls == 1 and warnings
    finally:
        sandbox.close()


def test_helper_cleanup_failure_does_not_report_success():
    warnings = []
    sandbox, _ = make_sandbox(warnings.append)
    data = SimpleNamespace(stdout='{"ok": true}', stderr='', offset=12, stderr_offset=0,
                           command=SimpleNamespace(status="completed", exit_code=0))
    sandbox._client.bash = SimpleNamespace(
        create_session=lambda session_id, **kwargs: response(SimpleNamespace(session_id=session_id)),
        exec=lambda **kwargs: response(SimpleNamespace(command_id="command")),
        output=lambda **kwargs: response(data),
        close_session=lambda *args, **kwargs: response(success=False),
    )
    try:
        with pytest.raises(RuntimeError, match="清理失败"):
            AioSandbox._bash_helper(sandbox, "command", None)
        assert "沙箱辅助会话清理失败" in warnings
    finally:
        sandbox.close()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout_does_not_start_a_command(timeout):
    sandbox, shell = make_sandbox(lambda reason: None)
    try:
        with pytest.raises(ValueError, match="超时无效"):
            sandbox.execute_command_in_scope("echo test", timeout=timeout, scope_id="worker")
        assert shell.calls == 0
    finally:
        sandbox.close()
