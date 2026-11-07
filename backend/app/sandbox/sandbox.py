from abc import ABC, abstractmethod


class Sandbox(ABC):
    def __init__(self, sandbox_id: str) -> None:
        self._id = sandbox_id

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def execute_command(self, command: str, env: dict[str, str] | None = None,
                        timeout: float | None = None) -> str:
        raise NotImplementedError

    def execute_command_in_scope(self, command: str, env: dict[str, str] | None = None,
                                 timeout: float | None = None, *, scope_id: str | None = None) -> str:
        return self.execute_command(command, env=env, timeout=timeout)

    def release_command_scope(self, scope_id: str) -> None:
        pass

    @abstractmethod
    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def write_file(self, path: str, content: str, append: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def glob(self, path: str, pattern: str, *, include_dirs: bool = False,
             max_results: int = 200) -> tuple[list[str], bool]:
        raise NotImplementedError
