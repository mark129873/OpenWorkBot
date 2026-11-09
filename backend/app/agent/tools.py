from pathlib import Path
from typing import Callable

from app.agent.skills import tool_load_skill
from app.agent.todo import TOOL_DESCRIPTION
from app.config import settings
from app.user_context import require_user


def workspace_root() -> Path:
    parent = settings.workspaces_dir.resolve()
    target = parent / require_user()
    if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
        raise ValueError("用户工作区不能是链接")
    if target.resolve().parent != parent:
        raise ValueError("用户工作区路径无效")
    return target


def safe_path(relative: str) -> Path:
    """解析真实路径，拒绝逃出工作区。"""
    root = workspace_root().resolve()
    target = (root / relative).resolve()
    if not (target == root or target.is_relative_to(root)):
        raise ValueError(f"路径越界：{relative} 必须位于工作区内")
    return target


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "load_skill": tool_load_skill,
}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


TOOLS: list[dict] = [
    _schema(
        "bash", "在用户共享的 Docker 沙箱内执行命令；工作目录为用户工作区。",
        {"command": {"type": "string", "description": "完整 shell 命令"},
         "description": {"type": "string"}}, ["command"],
    ),
    _schema(
        "read_file", "按行读取容器内文本文件；/mnt/user-data/workspace/.tool-results/ 的归档可使用 offset 和 limit 按字符分页。",
        {"path": {"type": "string", "description": "容器内绝对路径"}, "description": {"type": "string"},
         "start_line": {"type": "integer", "minimum": 1},
         "end_line": {"type": "integer", "minimum": 1},
         "offset": {"type": "integer", "minimum": 0},
         "limit": {"type": "integer", "minimum": 1, "maximum": 10000}}, ["path"],
    ),
    _schema(
        "write_file", "写入或追加容器内文本文件；已有文件须先读取当前版本。",
        {"path": {"type": "string", "description": "容器内绝对路径"}, "content": {"type": "string"},
         "description": {"type": "string"}, "append": {"type": "boolean"}}, ["path", "content"],
    ),
    _schema(
        "str_replace", "替换文件中的首个或全部匹配；须先读取当前版本。",
        {
            "path": {"type": "string", "description": "容器内绝对路径"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "description": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        ["path", "old_str", "new_str"],
    ),
    _schema(
        "glob", "按 glob 模式列出容器目录内文件。",
        {"path": {"type": "string", "description": "容器内绝对路径"}, "pattern": {"type": "string"}, "description": {"type": "string"},
         "include_dirs": {"type": "boolean"}, "max_results": {"type": "integer", "minimum": 1, "maximum": 1000}},
        ["pattern", "path"],
    ),
    _schema(
        "ls", "列出容器目录内容；/mnt/user-data/workspace/.tool-results 可用 offset 和 limit 分页查看归档。",
        {"path": {"type": "string", "description": "容器内绝对路径"}, "description": {"type": "string"},
         "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, ["path"],
    ),
    _schema(
        "grep", "搜索容器内文件的文本内容。",
        {"path": {"type": "string", "description": "容器内绝对路径"}, "pattern": {"type": "string"}, "description": {"type": "string"},
         "glob": {"type": "string"}, "literal": {"type": "boolean"}, "case_sensitive": {"type": "boolean"},
         "max_results": {"type": "integer", "minimum": 1, "maximum": 500}}, ["pattern", "path"],
    ),
    _schema(
        "write_todos", TOOL_DESCRIPTION,
        {
            "todos": {
                "type": "array",
                "description": "任务数组，每项 {content: str, status: pending|in_progress|completed}",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        ["todos"],
    ),
    _schema(
        "load_skill", "按名称加载完整技能说明；指定相对路径可按需读取技能目录内的文本资源。",
        {"name": {"type": "string", "description": "技能目录名"},
         "path": {"type": "string", "description": "可选，技能目录内的相对资源路径"},
         "offset": {"type": "integer", "minimum": 0, "description": "按资源读取结果提示继续"}}, ["name"],
    ),
]


def execute_tool(name: str, tool_input: dict) -> str:
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: 未知工具 {name}"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: 参数不匹配：{exc}"
    except Exception as exc:
        return f"Error: {exc}"
