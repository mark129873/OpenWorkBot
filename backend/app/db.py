from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _enable_sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def init_db(data_dir: Path) -> Engine:
    """复用已初始化引擎，避免遗留连接池。"""
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "app.db"

    # 请求可能跨线程使用连接，Session 仍须独立。
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    event.listen(engine, "connect", _enable_sqlite_pragmas)

    Base.metadata.create_all(engine)
    # 省略 sequence 时自动追加到所属范围队尾。
    with engine.begin() as connection:
        for table, scope in (("message", "session_id"), ("part", "message_id")):
            connection.exec_driver_sql(f"""
                CREATE TRIGGER IF NOT EXISTS {table}_sequence_autofill
                AFTER INSERT ON {table} WHEN new.sequence IS NULL
                BEGIN
                    UPDATE {table} SET sequence = (
                        SELECT coalesce(max(sequence), -1) + 1 FROM {table}
                        WHERE {scope} = new.{scope}
                    ) WHERE id = new.id;
                END;
            """)

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
