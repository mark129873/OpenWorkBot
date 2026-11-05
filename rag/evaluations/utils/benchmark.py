import json
import math
import os
import re
from pathlib import Path

from mcp import ClientSession

from app import tracing


def require_langfuse() -> None:
    if not tracing.enabled():
        raise SystemExit("评测需要在 rag/.env 中启用 LANGFUSE_ENABLED")
    if not all(os.getenv(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")):
        raise SystemExit("请在 rag/.env 中配置 Langfuse")


def error_name(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} (HTTP {status})" if isinstance(status, int) else type(exc).__name__


def create_output(directory: Path, base_id: str, extension: str, conflicts: Path | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        output_id = base_id if number == 1 else f"{base_id}-{number:02}"
        path = directory / f"{output_id}{extension}"
        if conflicts is None or not (conflicts / f"{output_id}.json").exists():
            try:
                return output_id, path, path.open("x", encoding="utf-8")
            except FileExistsError:
                pass
        number += 1


def load_benchmark(path: Path, limit: int | None = 1) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError("评测题目数量必须大于零")
    cases, seen = [], set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"评测数据第 {line_number} 行 JSON 无效") from exc
            if (not isinstance(case, dict)
                    or any(not isinstance(case.get(key), str) or not case[key].strip()
                           for key in ("id", "question", "reference"))
                    or case["id"] != case["id"].strip()
                    or not isinstance(case.get("behavior", ""), str)
                    or not isinstance(case.get("targets"), list) or not case["targets"]
                    or any(not isinstance(name, str) or document_number(name) is None for name in case["targets"])):
                raise ValueError(f"评测数据第 {line_number} 行内容无效")
            if case["id"] in seen:
                raise ValueError(f"评测数据第 {line_number} 行编号重复")
            seen.add(case["id"])
            case.setdefault("behavior", "")
            cases.append(case)
    if not cases:
        raise ValueError("评测数据为空")
    return cases if limit is None else cases[:limit]


def document_number(filename: str) -> str | None:
    match = re.match(r"^(A\d{2,3})(?=[_\s.]|$)", filename, re.IGNORECASE)
    return match.group(1).upper() if match else None


async def call_tool(session: ClientSession, name: str, arguments: dict) -> dict:
    result = await session.call_tool(name, arguments)
    if result.isError:
        raise RuntimeError(f"MCP {name} 失败")
    if result.structuredContent is not None:
        return result.structuredContent
    for block in result.content:
        if getattr(block, "type", "") == "text":
            return json.loads(block.text)
    raise RuntimeError(f"MCP {name} 没有返回结果")


def retrieval_scores(evidence: list[dict], expected: list[str]) -> tuple[float, float]:
    targets = {document_number(filename) for filename in expected}
    if not targets or None in targets:
        raise ValueError("目标文档编号无效")
    documents = []
    seen = set()
    for chunk in evidence:
        filename = chunk.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("检索结果缺少文档名")
        number = document_number(filename)
        identity = number or filename.casefold()
        if identity not in seen:
            seen.add(identity)
            documents.append(number)
            if len(documents) == 5:
                break
    matches = sum(number in targets for number in documents)
    dcg = sum(1 / math.log2(rank + 2) for rank, number in enumerate(documents) if number in targets)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(targets), 5)))
    return matches / len(targets), dcg / ideal
