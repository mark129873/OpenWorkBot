import shutil
import subprocess
import threading
import uuid
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
_READ_LIMIT = 100000
_CAPTURE_LIMIT = 1024 * 1024
_CAPTURE_MARKER = "\n...（输出超过 1 MiB，中段已截断）...\n"
_CAPTURE_HALF = _CAPTURE_LIMIT // 2


def _tail(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    return text if len(text) <= limit else f"...(前段已截断)...\n{text[-limit:]}"


def _bounded_capture(text: str) -> str:
    data = text.encode("utf-8")
    if len(data) <= _CAPTURE_LIMIT:
        return text
    keep = (_CAPTURE_LIMIT - len(_CAPTURE_MARKER.encode("utf-8"))) // 2
    return data[:keep].decode("utf-8", errors="ignore") + _CAPTURE_MARKER + data[-keep:].decode("utf-8", errors="ignore")


def capture_bash(command: str) -> tuple[str, bool]:
    """持续排空输出，最多保留首尾各半 MiB。"""
    try:
        process = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command",
             "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); " + command],
            cwd=workspace_root(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        head, tail = bytearray(), bytearray()
        total = 0

        def drain() -> None:
            nonlocal total
            assert process.stdout is not None
            while chunk := process.stdout.read(8192):
                total += len(chunk)
                missing = _CAPTURE_HALF - len(head)
                head.extend(chunk[:missing])
                tail.extend(chunk[missing:])
                if len(tail) > _CAPTURE_HALF:
                    del tail[:-_CAPTURE_HALF]

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        timed_out = False
        try:
            process.wait(timeout=settings.bash_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        reader.join(timeout=1)
        if total > _CAPTURE_LIMIT:
            keep = (_CAPTURE_LIMIT - len(_CAPTURE_MARKER.encode("utf-8"))) // 2
            output = (head[:keep].decode("utf-8", errors="ignore")
                      + _CAPTURE_MARKER
                      + tail[-keep:].decode("utf-8", errors="ignore"))
        else:
            output = (head + tail).decode("utf-8", errors="replace")
        if reader.is_alive():
            output += "\n...（输出管道尚未关闭，采集已停止）..."
        if timed_out:
            return _bounded_capture(f"Error: 命令超时（>{settings.bash_timeout}s）\n{output}"), True
        if process.returncode != 0:
            return _bounded_capture(f"Error: 命令退出码 {process.returncode}\n{output}\n[exit code: {process.returncode}]"), True
        return _bounded_capture(output.rstrip()), False
    except Exception as exc:
        return f"Error: 命令执行失败：{exc}", True


def output_preview(output: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(output) <= limit:
        return output
    marker = "\n...（输出较长，可按输出 ID 分段读取）...\n"
    half = (limit - len(marker)) // 2
    return output[:half] + marker + output[-(limit - half - len(marker)):]


def tool_bash(command: str) -> str:
    output, _ = capture_bash(command)
    return output_preview(output)


def tool_read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
    try:
        if (not isinstance(offset, int) or isinstance(offset, bool) or offset < 1
                or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 2000):
            return "Error: offset 必须从 1 开始，limit 必须为 1 到 2000"
        target = safe_path(path)
        if not target.is_file():
            return f"Error: 文件不存在：{path}"
        rows = []
        size = 0
        line_no = 0
        next_line = None
        with target.open(encoding="utf-8", errors="replace") as source:
            while first := source.readline(2001):
                line_no += 1
                long_line = not first.endswith("\n") and len(first) > 2000
                if long_line:
                    rest = first
                    while rest and not rest.endswith("\n"):
                        rest = source.readline(8192)
                if line_no < offset:
                    continue
                if line_no >= offset + limit:
                    next_line = line_no
                    break
                row = f"{line_no}: {first[:2000].rstrip(chr(10)).rstrip(chr(13))}"
                if long_line:
                    row += " …（该行超过 2000 字符，余下内容已截断）"
                if size + len(row) + 1 > _READ_LIMIT - 100:
                    next_line = line_no
                    break
                rows.append(row)
                size += len(row) + 1
        if next_line is not None:
            rows.append(f"[继续读取：offset={next_line}]")
        return "\n".join(rows) if rows else "(文件为空或已到文件末尾)"
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

        trash_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}_{target.name}"
        trash_dir = safe_path(".rubbish")
        trash_dir.mkdir(parents=True, exist_ok=True)
        candidate = trash_dir / trash_name
        if candidate.exists() or candidate.is_symlink():
            return "Error: 回收站目标已存在"
        destination = safe_path(str(candidate))
        shutil.move(str(target), str(destination))
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
        "read_file", "按行读取工作区内文本文件，超长时提示下一行。",
        {"path": {"type": "string", "description": "工作区内相对路径"},
         "offset": {"type": "integer", "minimum": 1},
         "limit": {"type": "integer", "minimum": 1, "maximum": 2000}}, ["path"],
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
        "load_skill", "按名称加载完整技能说明；指定相对路径可按需读取技能目录内的文本资源。",
        {"name": {"type": "string", "description": "技能目录名"},
         "path": {"type": "string", "description": "可选，技能目录内的相对资源路径"},
         "offset": {"type": "integer", "minimum": 0, "description": "按资源读取结果提示继续"}}, ["name"],
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
