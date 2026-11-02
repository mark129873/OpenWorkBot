import re
from collections import Counter
from pathlib import Path

import pdfplumber

from app.documents import _encoding

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAJOR = re.compile(r"^(?:第[零〇一二三四五六七八九十百千万\d]+[章节篇部]|[一二三四五六七八九十百]+[、．.])\s*\S")
_MINOR = re.compile(r"^(?:[（(][一二三四五六七八九十\d]+[）)]|\d+(?:\.\d+)+)\s*\S")
_ARABIC = re.compile(r"^\d+[.、]\s*\S")
_PAGE = re.compile(r"\s*第\s*\d+\s*页\s*$")


def _inside(char: dict, bbox: tuple) -> bool:
    x = (char["x0"] + char["x1"]) / 2
    y = (char["top"] + char["bottom"]) / 2
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _table(page, table) -> list[list] | None:
    cells = table.extract()
    if not cells or not any(cell for row in cells for cell in row):
        return None
    original = Counter(char["text"] for char in page.chars
                       if _inside(char, table.bbox) and not char["text"].isspace())
    extracted = Counter(char for row in cells for cell in row for char in (cell or "")
                        if not char.isspace())
    return cells if not original - extracted else None


def _pdf_events(page) -> tuple[list[dict], float]:
    sizes = Counter(round(char["size"], 1) for char in page.chars if char["text"].strip())
    body_size = sizes.most_common(1)[0][0] if sizes else 0
    tables = [(table, cells) for table in page.find_tables() if (cells := _table(page, table))]
    filtered = page.filter(lambda obj: obj.get("object_type") != "char"
                           or not any(_inside(obj, table.bbox) for table, _ in tables))
    events = []
    for line in filtered.extract_text_lines(return_chars=True):
        chars = line.get("chars", [])
        sizes = [char["size"] for char in chars]
        fonts = [str(char.get("fontname", "")).lower() for char in chars]
        bbox = [line["x0"], line["top"], line["x1"], line["bottom"]]
        events.append({"type": "line", "text": line["text"], "bbox": bbox,
                       "size": max(sizes, default=0),
                       "bold": any("bold" in font or "black" in font for font in fonts),
                       "edge": bbox[1] <= page.height * 0.05 or bbox[3] >= page.height * 0.95})
    for table, cells in tables:
        events.append({"type": "table", "cells": cells, "bbox": list(table.bbox)})
    for item in page.images:
        events.append({"type": "figure", "bbox": [item["x0"], item["top"], item["x1"], item["bottom"]]})
    events.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return events, body_size


def _title(events: list[dict], metadata: str, body_size: float, height: float) -> str:
    lines = [item for item in events if item["type"] == "line" and not item["edge"]]
    if metadata and any(item["text"].strip() == metadata for item in lines):
        return metadata
    for item in lines:
        text = item["text"].strip()
        if (body_size and item["bold"] and item["size"] >= body_size * 1.5
                and item["bbox"][1] < height * 0.3 and 2 < len(text) <= 120
                and not (_MAJOR.match(text) or _MINOR.match(text) or _ARABIC.match(text))):
            return text
    return ""


def _level(event: dict, body_size: float) -> int | None:
    text = event["text"].strip()
    styled = event["bold"] or (body_size and event["size"] >= body_size * 1.15)
    if not styled or len(text) > 120:
        return None
    if _MINOR.match(text):
        return 3
    if _MAJOR.match(text) or _ARABIC.match(text):
        return 2
    if body_size and event["size"] >= body_size * 1.3:
        return 2
    return None


def parse_document(path: Path, suffix: str, filename: str, document_id: str) -> tuple[dict, set[str]]:
    tree = []
    edge_ids = set()
    counts = Counter()
    stack = []

    def add(kind: str, page: int | None, bbox: list | None, parent: str | None, **fields) -> str:
        prefix = {"heading": "h", "paragraph": "p", "table": "t", "figure": "img"}[kind]
        counts[prefix] += 1
        identity = f"{prefix}{counts[prefix]}"
        node = {"id": identity, "type": kind, **fields, "page": page, "bbox": bbox,
                "order": len(tree) + 1}
        if parent:
            node["parent"] = parent
        tree.append(node)
        return identity

    def heading(level: int, text: str, page: int | None, bbox: list | None) -> None:
        while stack and stack[-1][0] >= level:
            stack.pop()
        identity = add("heading", page, bbox, stack[-1][1] if stack else None, level=level, text=text)
        stack.append((level, identity))

    if suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            source = {"filename": filename, "parser": "pdf", "pages": len(pdf.pages)}
            metadata = str(pdf.metadata.get("Title") or "").strip()
            for number, page in enumerate(pdf.pages, 1):
                events, body_size = _pdf_events(page)
                title = _title(events, metadata, body_size, page.height) if number == 1 else ""
                pending = []
                pending_parent = None
                pending_edge = False

                def flush() -> None:
                    if not pending:
                        return
                    bbox = [min(item["bbox"][0] for item in pending), min(item["bbox"][1] for item in pending),
                            max(item["bbox"][2] for item in pending), max(item["bbox"][3] for item in pending)]
                    identity = add("paragraph", number, bbox, pending_parent,
                                   text="\n".join(item["text"] for item in pending))
                    if pending_edge:
                        edge_ids.add(identity)
                    pending.clear()

                for event in events:
                    if event["type"] != "line":
                        flush()
                        parent = stack[-1][1] if stack else None
                        if event["type"] == "table":
                            add("table", number, event["bbox"], parent, cells=event["cells"])
                        else:
                            add("figure", number, event["bbox"], parent,
                                ocr_text=None, description=None, caption=None)
                        continue
                    text = event["text"].strip()
                    level = 1 if title and text == title else (None if event["edge"] else _level(event, body_size))
                    if level:
                        flush()
                        heading(level, text, number, event["bbox"])
                        continue
                    parent = (stack[0][1] if stack and event["edge"] else stack[-1][1] if stack else None)
                    gap = event["bbox"][1] - pending[-1]["bbox"][3] if pending else 0
                    if (pending and (parent != pending_parent or event["edge"] != pending_edge
                                     or gap > max(12, event["size"] * 1.5)
                                     or _ARABIC.match(text))):
                        flush()
                    pending_parent, pending_edge = parent, event["edge"]
                    pending.append(event)
                flush()
                page.close()
    else:
        parser = "markdown" if suffix.lower() == ".md" else "text"
        source = {"filename": filename, "parser": parser, "pages": None}
        with path.open("r", encoding=_encoding(path), newline=None) as file:
            pending = []
            fence = ""

            def flush_text() -> None:
                if pending:
                    add("paragraph", None, None, stack[-1][1] if stack else None,
                        text="\n".join(pending))
                    pending.clear()

            for line in file:
                text = line.rstrip("\n")
                marker = re.match(r"^\s*(`{3,}|~{3,})", text) if parser == "markdown" else None
                if marker:
                    fence = "" if fence and marker.group(1)[0] == fence else (fence or marker.group(1)[0])
                    pending.append(text)
                    continue
                match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*#*\s*$", text) if parser == "markdown" and not fence else None
                if match:
                    flush_text()
                    heading(len(match.group(1)), match.group(2), None, None)
                elif not text.strip() and not fence:
                    flush_text()
                else:
                    pending.append(text)
            flush_text()
    return {"doc_id": document_id, "source": source, "tree": tree}, edge_ids


def _join_pdf_lines(text: str) -> str:
    lines = text.split("\n")
    result = lines[0] if lines else ""
    for line in lines[1:]:
        if (result and line and not _ARABIC.match(line)
                and re.search(r"[\u3400-\u9fff\d，、；：]$", result)
                and re.match(r"[\u3400-\u9fff\d]", line)):
            result += line
        else:
            result += "\n" + line
    return result


def clean_document(document: dict, edge_ids: set[str]) -> None:
    occurrences = {}
    for node in document["tree"]:
        if node["id"] in edge_ids and node["type"] == "paragraph":
            text = _PAGE.sub("", _CONTROL.sub("", node["text"])).strip()
            if text:
                occurrences.setdefault(text, set()).add(node["page"])
    seen = set()
    kept = []
    for node in document["tree"]:
        if node["type"] in {"heading", "paragraph"}:
            text = _CONTROL.sub("", node["text"]).strip()
            if node["id"] in edge_ids:
                text = _PAGE.sub("", text).strip()
                if not text or re.fullmatch(r"[-—]?\s*\d+\s*[-—]?", text):
                    continue
                if len(occurrences.get(text, set())) > 1:
                    if text in seen:
                        continue
                    seen.add(text)
            elif document["source"]["parser"] == "pdf" and node["type"] == "paragraph":
                text = _join_pdf_lines(text)
            node["text"] = text
        elif node["type"] == "table":
            node["cells"] = [[_CONTROL.sub("", cell).strip() if isinstance(cell, str) else cell
                              for cell in row] for row in node["cells"]]
        kept.append(node)
    document["tree"] = kept
