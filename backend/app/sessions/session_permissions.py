"""会话授权仅存内存，不读写项目规则。"""

import json
import os

from app.agent.tools import safe_path

# 文件内容可变，未知参数仍须精确匹配以防扩权。
_FILE_PAYLOADS = {
    "read_file": ("offset", "limit"), "write_file": ("content",),
    "edit_file": ("old_text", "new_text"), "delete_file": (),
}
_grants: dict[str, set[tuple[str, str]]] = {}


def scope_for(tool: str, tool_input: dict) -> tuple[str, str] | None:
    """文件绑定真实路径；其他工具精确匹配完整参数，不支持通配。"""
    scoped = dict(tool_input)
    if tool in _FILE_PAYLOADS:
        path = scoped.get("path")
        if not isinstance(path, str) or not path:
            return None
        try:
            scoped["path"] = os.path.normcase(str(safe_path(path)))
        except (ValueError, OSError):
            return None
        for key in _FILE_PAYLOADS[tool]:
            scoped.pop(key, None)
    try:
        # 保留完整参数及大小写、空白，避免改变命令含义。
        return tool, json.dumps(scoped, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None


def describe(scope: tuple[str, str] | None) -> str:
    if scope is None:
        return "参数无效，无法保存会话授权"
    tool, content = scope
    prefix = "目标" if tool in _FILE_PAYLOADS else "完整命令参数（精确匹配）" if tool == "bash" else "完整参数（精确匹配）"
    return f"{prefix}：{content}"


def grant(session_id: str, scope: tuple[str, str]) -> None:
    _grants.setdefault(session_id, set()).add(scope)


def allows(session_id: str, tool: str, tool_input: dict) -> bool:
    scope = scope_for(tool, tool_input)
    return scope is not None and scope in _grants.get(session_id, set())


def forget_session(session_id: str) -> None:
    _grants.pop(session_id, None)
