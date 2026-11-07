import hashlib

from app.agent import tools
from app.sandbox.sandbox_provider import SandboxProvider
from app.user_context import require_user


class FakeSandbox:
    def file_operation(self, name, args):
        if name == "version":
            path = tools.safe_path(args["path"])
            try:
                return hashlib.sha256(path.read_bytes()).hexdigest()
            except FileNotFoundError:
                return "missing"
        return tools.execute_tool(name, args)

    def execute_command_in_scope(self, command, *, scope_id, timeout=None):
        return f"沙箱命令：{command}"

    def request_command_cancel(self, scope_id):
        pass

    def release_command_scope(self, scope_id):
        pass


class FakeProvider(SandboxProvider):
    def __init__(self):
        self.sandbox = FakeSandbox()

    def acquire(self, thread_id=None, *, user_id=None):
        assert user_id == require_user()
        return f"test-{user_id}"

    def get(self, sandbox_id):
        return self.sandbox if sandbox_id == f"test-{require_user()}" else None

    def get_scoped(self, sandbox_id, *, thread_id, user_id):
        return self.get(sandbox_id) if user_id == require_user() and thread_id else None

    def release(self, sandbox_id):
        assert self.get(sandbox_id) is not None
