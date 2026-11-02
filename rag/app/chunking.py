from collections.abc import Iterator

from app import rag_parameters as params


def _heading_path(parent: str | None, headings: dict[str, dict]) -> list[str]:
    path = []
    seen = set()
    while parent:
        if parent in seen or parent not in headings:
            raise ValueError("文档标题层级无效")
        seen.add(parent)
        heading = headings[parent]
        if heading.get("text", "").strip():
            path.append(heading["text"].strip())
        parent = heading.get("parent")
    return list(reversed(path))


def node_content(node: dict) -> str:
    if node["type"] == "table":
        return "\n".join("\t".join(str(cell) if cell is not None else "" for cell in row)
                         for row in node.get("cells", []))
    if node["type"] == "figure":
        return "\n".join(value for key in ("caption", "ocr_text", "description")
                         if isinstance((value := node.get(key)), str) and value.strip())
    return str(node.get("text") or "")


def _windows(text: str, capacity: int) -> Iterator[tuple[int, int]]:
    if capacity <= 0 or (len(text) > capacity and capacity <= params.CHUNK_OVERLAP):
        raise ValueError("标题过长，无法按分块大小保留原文及重叠")
    start = 0
    while start < len(text):
        end = min(start + capacity, len(text))
        if end < len(text):
            earliest = start + max(params.CHUNK_OVERLAP + 1, capacity // 2)
            boundary = max(text.rfind(mark, earliest, end) for mark in ("\n", "。", "！", "？", "；"))
            if boundary >= earliest:
                end = boundary + 1
        yield start, end
        if end == len(text):
            break
        start = end - params.CHUNK_OVERLAP


def chunks(document: dict) -> Iterator[dict]:
    nodes = document["tree"]
    headings = {node["id"]: node for node in nodes if node["type"] == "heading"}
    groups = []
    parent = None
    items = []
    for node in nodes:
        if node["type"] == "heading":
            if items:
                groups.append((parent, items))
                items = []
            parent = node["id"]
            continue
        value = node_content(node).strip()
        if not value:
            continue
        node_parent = node.get("parent")
        if items and node_parent != parent:
            groups.append((parent, items))
            items = []
        parent = node_parent
        items.append((value, node.get("page")))
    if items:
        groups.append((parent, items))
    if not groups and headings:
        last = next(reversed(headings.values()))
        groups.append((last.get("parent"), [(last.get("text", ""), last.get("page"))]))

    for parent, items in groups:
        heading_path = _heading_path(parent, headings)
        prefix = "\n".join(heading_path)
        capacity = params.CHUNK_SIZE - len(prefix) - bool(prefix)
        text = ""
        positions = []
        for value, page in items:
            if text:
                text += "\n"
            start = len(text)
            text += value
            positions.append((start, len(text), page))
        for start, end in _windows(text, capacity):
            content = text[start:end]
            pages = sorted({page for left, right, page in positions
                            if left < end and right > start and isinstance(page, int)})
            yield {"content": content, "heading_path": heading_path, "pages": pages,
                   "search_text": f"{prefix}\n{content}" if prefix else content}
