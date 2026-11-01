import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_registry: dict[str, list[Callable[[dict], Any]]] = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}


def register(event: str, callback: Callable[[dict], Any]) -> None:
    if event not in _registry:
        raise ValueError(f"未知 hook 事件：{event}")
    _registry[event].append(callback)


def trigger(event: str, context: dict) -> Any:
    """首个非 None 返回值短路后续回调。"""
    for callback in _registry.get(event, []):
        result = callback(context)
        if result is not None:
            return result
    return None


def _log_pre_tool(context: dict) -> None:
    logger.info("工具调用开始 tool=%s call_id=%s", context.get("name"), context.get("tool_call_id"))


def _log_post_tool(context: dict) -> None:
    logger.info(
        "工具调用结束 tool=%s call_id=%s status=%s 耗时=%.0fms",
        context.get("name"),
        context.get("tool_call_id"),
        context.get("status"),
        context.get("duration_ms", 0) or 0,
    )


def register_default_hooks() -> None:
    if _log_pre_tool not in _registry["PreToolUse"]:
        register("PreToolUse", _log_pre_tool)
    if _log_post_tool not in _registry["PostToolUse"]:
        register("PostToolUse", _log_post_tool)
