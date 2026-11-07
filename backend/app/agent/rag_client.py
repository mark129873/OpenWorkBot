import asyncio
import json
import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from app.config import settings
from app.models import new_id

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
        self.writes = asyncio.Lock()
        self.workspace = settings.data_dir.resolve() / "rag-inbox" / new_id()

    def _ensure_workspace(self) -> None:
        parent = self.workspace.parent
        if parent.is_symlink() or parent.is_junction() or parent.resolve().parent != parent.parent:
            raise ValueError("知识库临时目录无效")
        parent.mkdir(parents=True, exist_ok=True)
        if self.workspace.is_symlink() or self.workspace.is_junction():
            raise ValueError("知识库临时目录无效")
        self.workspace.mkdir(exist_ok=True)
        if self.workspace.resolve().parent != parent:
            raise ValueError("知识库临时目录无效")

    async def _run(self):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not python.is_file():
                raise RuntimeError("RAG 环境未安装")
            self._ensure_workspace()
            env = {
                "OPENWORKBOT_WORKSPACE": str(self.workspace),
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

    async def call(self, name: str, arguments: dict, *, sent: asyncio.Event | None = None) -> str:
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
            if sent is not None:
                sent.set()
            result = await session.call_tool(name, arguments)
            blocks = [block.text for block in result.content if getattr(block, "type", None) == "text"]
            if result.isError:
                suffix = "；操作结果可能未知，请先检查" if name in {"rag_ingest", "rag_remove"} else ""
                return "Error: " + ("\n".join(blocks) or "知识库操作失败") + suffix
            if result.structuredContent is not None:
                return json.dumps(result.structuredContent, ensure_ascii=False)
            if blocks:
                return "\n".join(blocks)
            if name in {"rag_ingest", "rag_remove"}:
                return "Error: 知识库操作结果未知，请先检查。"
            return "Error: 知识库未返回结果"
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("RAG MCP 调用失败", exc_info=True)
            if name in {"rag_ingest", "rag_remove"}:
                return "Error: 知识库操作结果未知，请先检查。"
            return "Error: 知识库服务不可用，请稍后重试。"

    def _target(self, relative: str) -> Path:
        path = PurePosixPath(relative.replace("\\", "/"))
        windows = PureWindowsPath(relative)
        if (not relative or path.is_absolute() or windows.drive or windows.root or ".." in path.parts
                or not path.name or "\x00" in relative):
            raise ValueError("知识库文档路径无效")
        self._ensure_workspace()
        target = self.workspace.joinpath(*path.parts)
        if not target.resolve().is_relative_to(self.workspace):
            raise ValueError("知识库临时路径越界")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.resolve().is_relative_to(self.workspace) or target.is_symlink():
            raise ValueError("知识库临时路径越界")
        return target

    @staticmethod
    def _copy(source: Path, target: Path, created: list[bool]) -> None:
        before = source.stat()
        if not source.is_file() or source.suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise ValueError("知识库仅支持工作区内的 PDF、TXT、Markdown 文件")
        if before.st_size > 100 * 1024 * 1024:
            raise ValueError("文件超过 100 MiB")
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            created[0] = True
            size = 0
            while chunk := incoming.read(1024 * 1024):
                size += len(chunk)
                if size > 100 * 1024 * 1024:
                    raise ValueError("文件超过 100 MiB")
                outgoing.write(chunk)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("文件读取期间发生变化，请重新入库")

    async def perform(self, runtime, name: str, arguments: dict, operation_id: str, before_send,
                      sent: asyncio.Event) -> str:
        from app.agent import tools
        from app.agent.memory import _io

        if name not in {"rag_ingest", "rag_remove"}:
            error = await before_send()
            return error or await self.call(name, {**arguments, "operation_id": operation_id}, sent=sent)
        async with self.writes:
            if name == "rag_remove":
                error = await before_send()
                if error:
                    return error
                result = await self.call(name, {**arguments, "operation_id": operation_id}, sent=sent)
                if result.startswith("Error:"):
                    return result
                try:
                    parsed = json.loads(result)
                    if (isinstance(parsed, dict) and parsed.get("status") == "success"
                            and parsed.get("document_id") == arguments["document_id"]):
                        return result
                except (KeyError, TypeError, ValueError):
                    pass
                return "Error: 知识库移出结果未知，请先检查。"
            value = arguments.get("path")
            if not isinstance(value, str) or not value:
                return "Error: 文件路径无效"
            try:
                source = tools.safe_path(value)
                if not source.is_file():
                    return "Error: 文件不存在"
                listing = await self.call("rag_list", {})
                if listing.startswith("Error:"):
                    return "Error: 无法读取知识库列表，未开始入库。"
                catalog = json.loads(listing)
                if not isinstance(catalog, dict) or catalog.get("status") != "success" or not isinstance(catalog.get("documents"), list):
                    raise ValueError("知识库列表无效")
                matches = [item for item in catalog["documents"] if isinstance(item, dict)
                           and isinstance(item.get("filename"), str)
                           and item["filename"].casefold() == source.name.casefold()]
                if len(matches) > 1:
                    return "Error: 知识库存在多个同名文档，请先检查。"
                if matches and matches[0].get("status") != "ready":
                    return "Error: 同名文档状态异常，请先检查。"
                relative = matches[0]["path"] if matches else f"new/{source.name}"
                if not isinstance(relative, str) or PurePosixPath(relative.replace("\\", "/")).name.casefold() != source.name.casefold():
                    raise ValueError("知识库文档路径与文件名不一致")
                target = self._target(relative)
                if target.exists():
                    return "Error: 上次入库的临时文件仍在，请先核查操作结果。"
                lock = runtime.file_locks.setdefault(os.path.normcase(str(source)), asyncio.Lock())
                keep, created = False, [False]
                try:
                    async with lock:
                        if tools.safe_path(value) != source:
                            return "Error: 文件路径已变化，请重新入库。"
                        await _io(self._copy, source, target, created)
                        if tools.safe_path(value) != source:
                            return "Error: 文件路径已变化，请重新入库。"
                    error = await before_send()
                    if error:
                        return error
                    result = await self.call("rag_ingest", {"path": relative, "operation_id": operation_id}, sent=sent)
                    keep = sent.is_set()
                    if not result.startswith("Error:"):
                        parsed = json.loads(result)
                        if (isinstance(parsed, dict) and parsed.get("status") == "success"
                                and isinstance(parsed.get("document_id"), str) and parsed["document_id"]
                                and (not matches or parsed["document_id"] == matches[0]["id"])):
                            keep = False
                        else:
                            return "Error: 知识库入库结果未知，请先检查。"
                    return result
                except asyncio.CancelledError:
                    keep = sent.is_set()
                    raise
                finally:
                    if created[0] and not keep and not target.is_symlink() and target.resolve().is_relative_to(self.workspace):
                        target.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                if sent.is_set():
                    return f"Error: 知识库入库结果未知，请先检查：{exc}"
                return f"Error: 知识库入库未开始：{exc}"

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
