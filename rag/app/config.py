import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    workspace: Path
    data_dir: Path
    milvus_uri: str
    milvus_token: str
    collection: str
    embedding_url: str
    embedding_model: str
    rerank_url: str
    rerank_model: str
    api_key: str
    base_url: str
    model: str
    provider: str = "anthropic"
    reasoning_effort: str = "none"
    temperature: float | None = 0


def settings() -> Settings:
    provider = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    prefix = "OPENAI" if provider == "openai" else "ANTHROPIC"
    value = os.getenv("RAG_TEMPERATURE", "0").strip().lower()
    try:
        temperature = None if value in {"", "none"} else float(value)
    except ValueError:
        raise ValueError("RAG_TEMPERATURE 必须为数字或 none") from None
    if temperature is not None and not 0 <= temperature <= 2:
        raise ValueError("RAG_TEMPERATURE 必须为 0 到 2 之间的数字或 none")
    return Settings(
        workspace=Path(os.getenv("OPENWORKBOT_WORKSPACE", ROOT.parent / "backend/data/workspace")).resolve(),
        data_dir=Path(os.getenv("RAG_DATA_DIR", ROOT / "data")).resolve(),
        milvus_uri=os.getenv("MILVUS_CLOUD_URI", ""),
        milvus_token=os.getenv("MILVUS_CLOUD_TOKEN", ""),
        collection=os.getenv("MILVUS_COLLECTION_NAME", "openworkbot_knowledge_chunks"),
        embedding_url=os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11436").rstrip("/"),
        embedding_model=os.getenv("EMBEDDING_MODEL_PATH", ""),
        rerank_url=os.getenv("RERANKER_BASE_URL", "http://127.0.0.1:11435").rstrip("/"),
        rerank_model=os.getenv("RERANKER_MODEL_PATH", ""),
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        base_url=os.getenv(f"{prefix}_BASE_URL", ""),
        model=os.getenv("MODEL_ID", ""),
        provider=provider,
        reasoning_effort=(os.getenv("RAG_REASONING_EFFORT") or "none").strip().lower() or "none",
        temperature=temperature,
    )
