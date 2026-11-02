import asyncio
import json
import logging
import os
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3] / "rag"
RAG_NAMES = {"rag_search", "rag_ingest", "rag_list", "rag_remove"}
START_TIMEOUT = 30


class RagClient:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.task = None
        self.session = None

    async def _run(self):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not python.is_file():
                raise RuntimeError("RAG 环境未安装")
            env = {
                "OPENWORKBOT_WORKSPACE": str((settings.data_dir / "workspace").resolve()),
                "LLM_PROVIDER": settings.llm_provider,
                "MODEL_ID": settings.model_id,
            }
            if settings.llm_provider == "openai":
                env.update(OPENAI_API_KEY=settings.openai_api_key, OPENAI_BASE_URL=settings.openai_base_url)
            else:
                env.update(ANTHROPIC_API_KEY=settings.anthropic_api_key, ANTHROPIC_BASE_URL=settings.anthropic_base_url)
            for key in ("RAG_DATA_DIR", "RAG_REASONING_EFFORT", "RAG_TEMPERATURE", "MILVUS_CLOUD_URI",
                        "MILVUS_CLOUD_TOKEN", "MILVUS_COLLECTION_NAME"):
                if key in os.environ:
                    env[key] = os.environ[key]
            params = StdioServerParameters(command=str(python), args=["-m", "app.server"],
                                           cwd=str(ROOT), env=env, encoding="utf-8")
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.ready.set()
                    await self.stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("RAG MCP 服务不可用", exc_info=True)
        finally:
            self.session = None
            self.ready.set()

    async def call(self, name: str, arguments: dict) -> str:
        if name not in RAG_NAMES:
            return "Error: 未知知识库工具"
        async with self.lock:
            if self.task is None or self.task.done():
                self.ready = asyncio.Event()
                self.stop = asyncio.Event()
                self.task = asyncio.create_task(self._run(), name="rag-mcp")
            try:
                await asyncio.wait_for(self.ready.wait(), timeout=START_TIMEOUT)
            except asyncio.TimeoutError:
                self.task.cancel()
                return "Error: 知识库服务启动超时，请检查 RAG 配置"
            session = self.session
        if session is None:
            return "Error: 知识库服务不可用，请检查 RAG 配置"
        try:
            result = await session.call_tool(name, arguments)
            blocks = [block.text for block in result.content if getattr(block, "type", None) == "text"]
            if result.isError:
                suffix = "；操作结果可能未知，请先检查" if name in {"rag_ingest", "rag_remove"} else ""
                return "Error: " + ("\n".join(blocks) or "知识库操作失败") + suffix
            if result.structuredContent is not None:
                return json.dumps(result.structuredContent, ensure_ascii=False)
            return "\n".join(blocks) or "Error: 知识库未返回结果"
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("RAG MCP 调用失败", exc_info=True)
            return "Error: 知识库服务不可用或调用结果未知，请检查后重试"

    async def close(self):
        if self.task:
            self.stop.set()
            try:
                await asyncio.wait_for(self.task, timeout=10)
            except asyncio.TimeoutError:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise

    async def cancel(self, operation_id: str):
        if self.session is not None:
            try:
                await asyncio.wait_for(self.session.call_tool("rag_cancel", {"operation_id": operation_id}), 10)
            except Exception:
                logger.warning("RAG 操作停止结果未知")
