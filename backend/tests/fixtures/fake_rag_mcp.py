import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "rag"))
os.environ["LANGFUSE_ENABLED"] = "false"

from app import server
from app.config import settings
from app.knowledge import Knowledge


class Models:
    async def embed(self, texts):
        return [[0.5, 0.5] for _ in texts]


class Vectors:
    async def add(self, version, filename, start, chunks, vectors):
        if os.environ.get("FAKE_RAG_FAIL_ADD") == "1":
            raise RuntimeError("模拟索引失败")

    async def delete(self, version):
        pass


server.knowledge = Knowledge(settings(), models=Models(), vectors=Vectors())


async def search(query):
    return {"status": "success", "answer": f"知识库回答：{query}", "sources": []}


server.knowledge.search = search
server.mcp.run(transport="stdio")
