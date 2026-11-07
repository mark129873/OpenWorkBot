from contextlib import contextmanager
from contextvars import ContextVar
import re


_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)


def current_user() -> str | None:
    return _user_id.get()


def require_user() -> str:
    user_id = current_user()
    if not user_id:
        raise RuntimeError("缺少用户身份")
    return user_id


def scoped(key: str) -> str:
    return f"{require_user()}/{key}"


def unscoped(key: str) -> str:
    prefix = f"{require_user()}/"
    if not key.startswith(prefix):
        raise ValueError("记录不属于当前用户")
    return key[len(prefix):]


@contextmanager
def bind_user(user_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", user_id):
        raise ValueError("用户身份无效")
    token = _user_id.set(user_id)
    try:
        yield
    finally:
        _user_id.reset(token)
