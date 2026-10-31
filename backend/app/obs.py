import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(data_dir: Path, level: str = "INFO") -> None:
    """重置前关闭旧 handler，避免重复日志和文件占用。"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "urllib3", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def shutdown_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:  # 清理失败不能掩盖原异常。
            pass
        root.removeHandler(handler)
