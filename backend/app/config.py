import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[1]

# .env 优先于已有环境变量。
load_dotenv(BACKEND_ROOT / ".env", override=True)


def _env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    llm_provider: str = "anthropic"
    anthropic_api_key: str = field(default="")
    anthropic_base_url: str = field(default="")
    openai_api_key: str = field(default="")
    openai_base_url: str = field(default="")
    model_id: str = field(default="")

    max_tokens: int = 10000
    bash_timeout: int = 120
    context_window: int = 1_000_000
    compact_output_reserve: int = 32000
    compact_buffer: int = 13000
    microcompact_keep: int = 5

    host: str = "127.0.0.1"
    port: int = 8100
    data_dir: Path = field(default=BACKEND_ROOT / "data")

    log_level: str = "INFO"

    modelio_dir: Path = field(default_factory=lambda: BACKEND_ROOT.parent / "log")
    modelio_max_bytes: int = 10 * 1024 * 1024


def load_settings() -> Settings:
    data_dir = Path(_env_str("DATA_DIR", "data"))
    # 相对路径基于 backend/，不依赖启动目录。
    if not data_dir.is_absolute():
        data_dir = BACKEND_ROOT / data_dir

    modelio_dir = Path(_env_str("MODELIO_DIR", str(BACKEND_ROOT.parent / "log")))
    if not modelio_dir.is_absolute():
        modelio_dir = BACKEND_ROOT / modelio_dir

    return Settings(
        llm_provider=_env_str("LLM_PROVIDER", "anthropic").lower(),
        anthropic_api_key=_env_str("ANTHROPIC_API_KEY"),
        anthropic_base_url=_env_str("ANTHROPIC_BASE_URL"),
        openai_api_key=_env_str("OPENAI_API_KEY"),
        openai_base_url=_env_str("OPENAI_BASE_URL"),
        model_id=_env_str("MODEL_ID"),
        max_tokens=_env_int("MAX_TOKENS", 10000),
        bash_timeout=_env_int("BASH_TIMEOUT", 120),
        context_window=_env_int("CONTEXT_WINDOW", 1_000_000),
        compact_output_reserve=_env_int("COMPACT_OUTPUT_RESERVE", 32000),
        compact_buffer=_env_int("COMPACT_BUFFER", 13000),
        microcompact_keep=_env_int("MICROCOMPACT_KEEP", 5),
        host=_env_str("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        data_dir=data_dir,
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        modelio_dir=modelio_dir,
        modelio_max_bytes=_env_int("MODELIO_MAX_MB", 10) * 1024 * 1024,
    )


def validate_llm_config(s: Settings) -> list[str]:
    """发起对话时才校验，健康检查不依赖模型密钥。"""
    missing: list[str] = []
    if s.llm_provider not in {"anthropic", "openai"}:
        missing.append("LLM_PROVIDER")
    elif s.llm_provider == "anthropic" and not s.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    elif s.llm_provider == "openai" and not s.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not s.model_id:
        missing.append("MODEL_ID")
    return missing


settings = load_settings()
