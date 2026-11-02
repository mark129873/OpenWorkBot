import asyncio
import logging
import os
import uuid

from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.knowledge import Knowledge
from app import tracing

logging.basicConfig(level=logging.INFO)
for noisy in ("httpx2", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
mcp = FastMCP("OpenWorkBot RAG")
knowledge = Knowledge(settings())
operations: dict[str, asyncio.Task] = {}


async def run_operation(operation_id: str, operation):
    identity = operation_id or uuid.uuid4().hex
    if identity in operations:
        raise ValueError("知识库操作 ID 已在使用")
    operations[identity] = asyncio.current_task()
    try:
        return await operation()
    finally:
        operations.pop(identity, None)


@mcp.tool()
async def rag_search(query: str, operation_id: str = "", include_evidence: bool = False) -> dict:
    """查询知识库并返回答案。"""
    identity = operation_id or uuid.uuid4().hex

    async def search():
        with tracing.query(query, identity) as span:
            result = (await knowledge.search(query, include_evidence=True) if include_evidence
                      else await knowledge.search(query))
            tracing.update(span, output=result)
            return result

    return await run_operation(identity, search)


@mcp.tool()
async def rag_ingest(path: str, operation_id: str = "") -> dict:
    """将工作区内的文件加入知识库；已有文件则更新索引。"""
    return await run_operation(operation_id, lambda: knowledge.ingest(path))


@mcp.tool()
async def rag_list(operation_id: str = "") -> dict:
    """查看知识库文件。"""
    return await run_operation(operation_id, knowledge.list_documents)


@mcp.tool()
async def rag_remove(document_id: str, operation_id: str = "") -> dict:
    """从知识库移出文件，不删除工作区原文件。"""
    return await run_operation(operation_id, lambda: knowledge.remove(document_id))


@mcp.tool()
async def rag_cancel(operation_id: str) -> dict:
    """停止由运行时发起的知识库操作。"""
    task = operations.get(operation_id)
    if not task:
        return {"status": "not_running"}
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return {"status": "cancelled"}


async def rag_flush_tracing() -> dict:
    """发送本进程中待上报的追踪。"""
    if not tracing.enabled():
        return {"status": "disabled"}
    await asyncio.to_thread(tracing.configured_client().flush)
    return {"status": "success"}


if os.getenv("OPENWORKBOT_BENCHMARK_FLUSH") == "1":
    mcp.tool()(rag_flush_tracing)


if __name__ == "__main__":
    mcp.run(transport="stdio")
