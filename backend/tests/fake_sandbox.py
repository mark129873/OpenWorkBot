from app.agent import tools
from app.sandbox.sandbox_provider import SandboxProvider
from app.user_context import require_user


class FakeSandbox:
    def _path(self, path):
        root = tools.workspace_root()
        root.mkdir(parents=True, exist_ok=True)
        prefix = "/mnt/user-data/workspace"
        if path == prefix:
            return root
        if not path.startswith(prefix + "/"):
            raise ValueError("测试沙箱仅包含工作区")
        return root / path[len(prefix) + 1:]

    def real_path(self, path):
        root = tools.workspace_root().resolve()
        target = self._path(path).resolve()
        return "/mnt/user-data/workspace/" + target.relative_to(root).as_posix()

    def read_file(self, path, start_line=None, end_line=None):
        try:
            content = self._path(path).read_text(encoding="utf-8")
            if start_line is not None or end_line is not None:
                return "\n".join(content.splitlines()[(start_line or 1) - 1:end_line])
            return content
        except OSError:
            return f"Error: file not found or unreadable: {path}"

    def write_file(self, path, content, append=False):
        target_path = self._path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("a" if append else "w", encoding="utf-8", newline="") as target:
            target.write(content)

    def list_dir(self, path, max_depth=2):
        root = self._path(path)
        if not root.is_dir():
            raise FileNotFoundError(path)
        return [path] + [path.rstrip("/") + "/" + item.relative_to(root).as_posix()
                         for item in root.rglob("*") if len(item.relative_to(root).parts) <= max_depth]

    def glob(self, path, pattern, *, include_dirs=False, max_results=200):
        root = self._path(path)
        matches = [path.rstrip("/") + "/" + item.relative_to(root).as_posix() for item in root.glob(pattern)
                   if include_dirs or item.is_file()]
        return matches[:max_results], len(matches) > max_results

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

    def reset_user(self, user_id):
        assert user_id == require_user()
