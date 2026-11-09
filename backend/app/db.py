from pathlib import Path

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def postgres_url(*, sqlalchemy: bool = False) -> str:
    try:
        url = make_url(settings.database_url)
        if url.drivername not in {"postgresql", "postgresql+psycopg"} or not url.database:
            raise ValueError
    except (ArgumentError, ValueError, TypeError):
        raise ValueError("PostgreSQL 需要有效的 DATABASE_URL") from None
    driver = "postgresql+psycopg" if sqlalchemy else "postgresql"
    return url.set(drivername=driver).render_as_string(hide_password=False)


def init_db(data_dir: Path) -> Engine:
    """复用已初始化引擎，避免遗留连接池。"""
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    data_dir.mkdir(parents=True, exist_ok=True)
    if settings.database_backend == "sqlite":
        engine = create_engine(f"sqlite:///{data_dir / 'app.db'}", connect_args={"check_same_thread": False})
        event.listen(engine, "connect", _enable_sqlite_pragmas)
    elif settings.database_backend == "postgresql":
        engine = create_engine(postgres_url(sqlalchemy=True), pool_pre_ping=True)
    else:
        raise ValueError("DATABASE_BACKEND 必须是 sqlite 或 postgresql")

    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        required = {
            "session": {"user_id", "assistant_id", "revision", "user_seq", "reset_state", "last_reset",
                        "next_message_sequence", "next_part_sequence", "next_display_order", "runtime_data"},
            "message": {"session_id", "sequence", "display_order", "role", "status", "text", "attachments"},
            "part": {"session_id", "key", "sequence", "type", "context_id", "data"},
        }
        for name, columns in required.items():
            if name in tables and not columns.issubset({column["name"] for column in inspector.get_columns(name)}):
                raise RuntimeError("检测到旧数据库结构；请使用空数据库，旧数据不会自动迁移")
        Base.metadata.create_all(engine)
    except BaseException:
        engine.dispose()
        raise

    _engine = engine
    _session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("数据库未初始化：请先调用 init_db()")
    return _engine


def new_session() -> Session:
    """调用方负责关闭会话。"""
    if _session_factory is None:
        raise RuntimeError("数据库未初始化：请先调用 init_db()")
    return _session_factory()


def get_db() -> Session:
    if _session_factory is None:
        raise RuntimeError("数据库未初始化：请先调用 init_db()")
    db = _session_factory()
    try:
        yield db
    finally:
        db.close()


def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
