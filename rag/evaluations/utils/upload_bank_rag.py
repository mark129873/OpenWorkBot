import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.catalog import Catalog
from app.config import settings
from app.documents import remove_all_structures
from app.models import Models
from app.vector_store import VectorStore
from evaluations.utils.benchmark import call_tool, document_number

def load_documents(workspace: Path) -> list[Path]:
    root = workspace / "benchmarks" / "bank-rag-benchmark"
    documents = sorted((root / "documents").glob("*.pdf"))
    numbers = [document_number(path.name) for path in documents]
    if len(documents) != 100 or None in numbers or len(set(numbers)) != 100:
        raise ValueError("基准文档不完整或文档编号重复")
    return documents


def reset_collection(config) -> None:
    catalog = Catalog(config.data_dir)
    vectors = VectorStore(config, catalog)
    client = vectors._connect()
    try:
        owned = catalog.setting("collection")
        exists = client.has_collection(config.collection)
        identity = vectors._identity()
        if ((owned and owned != identity) or (exists and owned != identity)
                or (not owned and catalog.list())):
            print("集合或本地索引不属于当前知识库，拒绝重建", file=sys.stderr)
            raise SystemExit(1)
        if exists:
            client.drop_collection(config.collection)
        remove_all_structures(config.data_dir)
        with catalog.connect() as db:
            db.execute("DELETE FROM documents")
            db.execute("DELETE FROM staged")
            db.execute("DELETE FROM settings WHERE key IN ('collection', 'embedding_model')")
    finally:
        client.close()


async def ingest_documents(session: ClientSession, documents: list[Path]) -> None:
    for index, path in enumerate(documents, 1):
        result = await call_tool(session, "rag_ingest", {"path": str(path)})
        if result.get("status") != "success":
            raise RuntimeError("基准文档入库失败")
        if index % 10 == 0 or index == len(documents):
            print(f"基准文档已入库 {index}/{len(documents)}", flush=True)


async def main() -> None:
    config = settings()
    if not config.embedding_model or not config.milvus_uri or not config.milvus_token:
        print("请在 rag/.env 中配置 Milvus 和 Embedding 模型", file=sys.stderr)
        raise SystemExit(1)
    if not Path(config.embedding_model).is_file():
        print("Embedding 模型文件不存在", file=sys.stderr)
        raise SystemExit(1)
    try:
        documents = load_documents(config.workspace)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
    try:
        await Models(config).embed(["测试"])
    except Exception:
        print("Embedding 服务不可用，未删除集合", file=sys.stderr)
        raise SystemExit(1)
    print(f"正在重建 Milvus 集合：{config.collection}", flush=True)
    reset_collection(config)
    server_params = StdioServerParameters(command=sys.executable, args=["-m", "app.server"], cwd=ROOT,
                                          env=dict(os.environ))
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await ingest_documents(session, documents)
    print("基准文档入库完成", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"入库未完成：{type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)
