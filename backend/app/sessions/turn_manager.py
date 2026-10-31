import asyncio

_active: dict[str, str] = {}
_stop_events: dict[str, asyncio.Event] = {}


def register(session_id: str, turn_id: str) -> asyncio.Event:
    _active[session_id] = turn_id
    event = asyncio.Event()
    _stop_events[session_id] = event
    return event


def unregister(session_id: str, turn_id: str) -> None:
    """仅注销匹配的 turn_id，避免旧轮误删新轮。"""
    if _active.get(session_id) == turn_id:
        _active.pop(session_id, None)
        _stop_events.pop(session_id, None)


def request_stop(session_id: str) -> bool:
    event = _stop_events.get(session_id)
    if event is None:
        return False
    event.set()
    return True


def is_running(session_id: str) -> bool:
    return session_id in _active


def running_session_ids() -> set[str]:
    return set(_active.keys())


def running_turn_ids() -> set[str]:
    """回放需要轮次 ID 集合，不能使用会话 ID。"""
    return set(_active.values())
