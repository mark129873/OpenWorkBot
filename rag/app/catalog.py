from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Catalog:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "catalog.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, filename TEXT NOT NULL,
                    version TEXT NOT NULL, size INTEGER NOT NULL, status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS staged (version TEXT PRIMARY KEY, path TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def setting(self, key: str) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (key, value))

    def list(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM documents ORDER BY filename, id")]

    def get(self, document_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
            return dict(row) if row else None

    def versions(self) -> list[str]:
        with self.connect() as db:
            return [row["version"] for row in db.execute("SELECT version FROM documents WHERE status = 'ready'")]

    def stage(self, version: str, path: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO staged VALUES (?, ?)", (version, path))

    def unstage(self, version: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM staged WHERE version = ?", (version,))

    def activate(self, document_id: str, path: str, filename: str, version: str, size: int) -> tuple[str, str | None]:
        with self.connect() as db:
            old = db.execute("SELECT id, version FROM documents WHERE path = ?", (path,)).fetchone()
            if old:
                document_id = old["id"]
                db.execute("UPDATE documents SET filename = ?, version = ?, size = ?, status = 'ready' WHERE id = ?",
                           (filename, version, size, document_id))
            else:
                db.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?, 'ready')",
                           (document_id, path, filename, version, size))
            db.execute("DELETE FROM staged WHERE version = ?", (version,))
            return document_id, old["version"] if old else None

    def set_status(self, document_id: str, status: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE documents SET status = ? WHERE id = ?", (status, document_id))

    def remove(self, document_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
