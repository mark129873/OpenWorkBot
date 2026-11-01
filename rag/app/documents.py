import codecs
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

MAX_FILE_SIZE = 100 * 1024 * 1024
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def snapshot(workspace: Path, relative_path: str, staging: Path) -> tuple[Path, Path]:
    root = workspace.resolve()
    source = (root / relative_path).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("文件必须位于工作区内")
    if source.suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise ValueError("知识库仅支持 PDF、TXT、Markdown")
    before = source.stat()
    if before.st_size > MAX_FILE_SIZE:
        raise ValueError("文件超过 100 MiB")
    staging.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="source-", suffix=source.suffix, dir=staging)
    target = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as src:
            shutil.copyfileobj(src, out, length=1024 * 1024)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("文件读取期间发生变化，请重新入库")
        return source, target
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _encoding(path: Path) -> str:
    for encoding in ("utf-8", "gb18030", "big5"):
        decoder = codecs.getincrementaldecoder(encoding)()
        try:
            with path.open("rb") as file:
                while block := file.read(65536):
                    decoder.decode(block)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("文本编码不受支持")


def fragments(path: Path, suffix: str) -> Iterator[str]:
    if suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                yield page.extract_text() or ""
        except Exception as exc:
            raise ValueError("PDF 解析失败") from exc
        return
    with path.open("r", encoding=_encoding(path), newline=None) as file:
        while block := file.read(65536):
            yield block


def chunks(path: Path, suffix: str) -> Iterator[str]:
    buffer = ""
    emitted = False
    for text in fragments(path, suffix):
        buffer += re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        while len(buffer) >= CHUNK_SIZE:
            piece = buffer[:CHUNK_SIZE].strip()
            if piece:
                emitted = True
                yield piece
            buffer = buffer[CHUNK_SIZE - CHUNK_OVERLAP:]
    if buffer.strip() and (not emitted or len(buffer) > CHUNK_OVERLAP):
        yield buffer.strip()
