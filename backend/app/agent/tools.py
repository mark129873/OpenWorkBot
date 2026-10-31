import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable

from app.agent.skills import tool_load_skill
from app.config import settings


def workspace_root() -> Path:
    return settings.data_dir / "workspace"


def safe_path(relative: str) -> Path:
    """解析真实路径，拒绝逃出工作区。"""
    root = workspace_root().resolve()
    target = (root / relative).resolve()
    if not (target == root or target.is_relative_to(root)):
        raise ValueError(f"路径越界：{relative} 必须位于工作区内")
    return target


def safe_glob_pattern(pattern: str) -> str:
    """仅接受工作区相对模式；固定前缀也须通过沙箱校验。"""
    # 跨平台识别盘符与斜杠，防止异系统路径绕过校验。
    normalized = pattern.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    windows = PureWindowsPath(normalized)
    if not normalized or windows.drive or windows.root or ".." in parts:
        raise ValueError("glob 模式必须是工作区内的相对路径，不能包含父级路径")
    prefix = []
    for part in parts:
        if any(char in part for char in "*?["):
            break
        prefix.append(part)
    safe_path("/".join(prefix))
    return normalized


# 保留输出尾部，避免丢失末尾报错。
_OUTPUT_LIMIT = 30000
_READ_LIMIT = 60000


def _tail(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    return text if len(text) <= limit else f"...(前段已截断)...\n{text[-limit:]}"


def tool_bash(command: str) -> str:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=workspace_root(),
            capture_output=True,
            text=True,
            timeout=settings.bash_timeout,
            encoding="utf-8",
            errors="replace",
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        output = _tail(output.rstrip())
        if completed.returncode != 0:
            # 截断后再加 Error:，保留失败标识。
            return f"Error: 命令退出码 {completed.returncode}\n{output}\n[exit code: {completed.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: 命令超时（>{settings.bash_timeout}s）"
    except Exception as exc:
        return f"Error: 命令执行失败：{exc}"


def tool_read_file(path: str) -> str:
    try:
        target = safe_path(path)
        if not target.is_file():
            return f"Error: 文件不存在：{path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        return _tail(text, _READ_LIMIT)
    except Exception as exc:
        return f"Error: 读取失败：{exc}"


def tool_write_file(path: str, content: str) -> str:
    try:
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {path}（{len(content)} 字符）"
    except Exception as exc:
        return f"Error: 写入失败：{exc}"


def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        target = safe_path(path)
        if not target.is_file():
            return f"Error: 文件不存在：{path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_text)
        if count == 0:
            return "Error: old_text 在文件中未找到"
        if count > 1:
            return f"Error: old_text 出现了 {count} 次，要求恰好 1 次；请提供更长的片段"
        target.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"已编辑 {path}"
    except Exception as exc:
        return f"Error: 编辑失败：{exc}"


def tool_glob(pattern: str) -> str:
    try:
        pattern = safe_glob_pattern(pattern)
        root = workspace_root().resolve()
        matches = []
        for path in root.glob(pattern):
            # 通配结果须复验真实路径，防止符号链接泄漏外部文件名。
            if not path.resolve().is_relative_to(root):
                raise ValueError("glob 匹配越出工作区，请缩小匹配范围")
            if path.is_file():
                matches.append(path.relative_to(root).as_posix())
        matches.sort()
        if not matches:
            return "(无匹配文件)"
        return _tail("\n".join(matches), _READ_LIMIT)
    except Exception as exc:
        return f"Error: glob 失败：{exc}"


def tool_delete_file(path: str) -> str:
    """删除改为移入工作区 .rubbish，可人工恢复。"""
    try:
        target = safe_path(path)
        if not target.is_file():
            return f"Error: 文件不存在：{path}"
        import time

        trash_name = f"{int(time.time() * 1000)}_{target.name}"
        trash_dir = workspace_root() / ".rubbish"
        trash_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(trash_dir / trash_name))
        return f"已删除 {path}（移入 .rubbish/{trash_name}，可找回）"
    except Exception as exc:
        return f"Error: 删除失败：{exc}"


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
    "glob": tool_glob,
    "delete_file": tool_delete_file,
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
        "bash", "在工作区内用 PowerShell 执行命令，返回合并输出。工作目录即工作区根。",
        {"command": {"type": "string", "description": "PowerShell 命令"}}, ["command"],
    ),
    _schema(
        "read_file", "读取工作区内文本文件（超长截断尾部保留）。",
        {"path": {"type": "string", "description": "工作区内相对路径"}}, ["path"],
    ),
    _schema(
        "write_file", "写入/覆盖工作区内文本文件，自动创建父目录。",
        {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"],
    ),
    _schema(
        "edit_file", "精确替换文件片段；old_text 必须在文件中恰好出现一次。",
        {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        ["path", "old_text", "new_text"],
    ),
    _schema(
        "glob", "按 glob 模式列出工作区文件（如 **/*.py）。",
        {"pattern": {"type": "string"}}, ["pattern"],
    ),
    _schema(
        "delete_file", "删除工作区文件（移入 .rubbish/ 可找回）。",
        {"path": {"type": "string"}}, ["path"],
    ),
    _schema(
        "todo_write", "全量更新任务板（≤20 条；进行中最多 1 条）。",
        {
            "items": {
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
        ["items"],
    ),
    _schema(
        "load_skill", "加载指定技能的完整说明文档（按需加载，先看目录再取）。",
        {"name": {"type": "string", "description": "技能目录名"}}, ["name"],
    ),
    _schema(
        "subtask", "派出子助手独立完成一个目标（有自己的工具循环），返回其最终汇报。",
        {"goal": {"type": "string", "description": "子助手要完成的单一目标，要写得具体"}}, ["goal"],
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
