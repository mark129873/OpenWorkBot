VALID_STATUS = ("pending", "in_progress", "completed")
MAX_ITEMS = 20

_STATUS_MARK = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


def parse_items(raw) -> tuple[list[dict] | None, str]:
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except ValueError:
            return None, "Error: items 不是合法 JSON"

    if not isinstance(raw, list):
        return None, "Error: items 必须是数组"
    if len(raw) > MAX_ITEMS:
        return None, f"Error: 任务数超过上限 {MAX_ITEMS}"

    items: list[dict] = []
    in_progress = 0
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            entry = {"content": entry}
        if not isinstance(entry, dict):
            return None, f"Error: 第 {i + 1} 条任务格式不正确"
        content = str(entry.get("content", "")).strip()
        status = entry.get("status", "pending")
        if not content:
            return None, f"Error: 第 {i + 1} 条任务内容为空"
        if status not in VALID_STATUS:
            return None, f"Error: 第 {i + 1} 条任务状态非法：{status}"
        if status == "in_progress":
            in_progress += 1
        items.append({"content": content, "status": status})
    if in_progress > 1:
        return None, "Error: 进行中的任务最多 1 条"
    return items, ""


def render_panel(items: list[dict]) -> str:
    if not items:
        return "(任务板为空)"
    lines = [f"{_STATUS_MARK[it['status']]} {it['content']}" for it in items]
    return "\n".join(lines)


async def handle_todo_write(deps, message_id: str, tool_input: dict) -> tuple[str, list[dict]]:
    items, error = parse_items(tool_input.get("items"))
    if items is None:
        return error, []

    deps.recorder.write_todos(items)
    return render_panel(items), [{"type": "todo_updated", "items": items}]
