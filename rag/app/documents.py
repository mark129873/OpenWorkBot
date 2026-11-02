import codecs
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from app.chunking import node_content

MAX_FILE_SIZE = 100 * 1024 * 1024


def _structures_root(data_dir: Path) -> Path:
    data_root = data_dir.resolve()
    if (data_root / "documents").is_symlink():
        raise ValueError("文档目录无效")
    root = (data_root / "documents").resolve()
    if not root.is_relative_to(data_root):
        raise ValueError("文档目录无效")
    return root


def _document_directory(data_dir: Path, document_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", document_id):
        raise ValueError("文档 ID 无效")
    root = _structures_root(data_dir)
    if (root / document_id).is_symlink():
        raise ValueError("文档目录无效")
    directory = (root / document_id).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("文档目录无效")
    return directory


def save_structure(data_dir: Path, document_id: str, version: str, document: dict) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", version):
        raise ValueError("文档版本无效")
    directory = _document_directory(data_dir, document_id)
    directory.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="structure-", suffix=".tmp", dir=directory)
    temporary = Path(name)
    target = directory / f"{version}.json"
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(document, file, ensure_ascii=False, indent=2)
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def read_document(data_dir: Path, document_id: str, version: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", version):
        raise ValueError("文档版本无效")
    path = _document_directory(data_dir, document_id) / f"{version}.json"
    if path.is_symlink():
        raise ValueError("文档结构文件无效")
    with path.open(encoding="utf-8") as file:
        document = json.load(file)
    if (not isinstance(document, dict) or document.get("doc_id") != document_id
            or not isinstance(document.get("tree"), list)):
        raise ValueError("文档结构文件无效")
    parts, last_page = [], None
    for node in document["tree"]:
        text = node_content(node).strip()
        if text:
            page = node.get("page")
            if isinstance(page, int) and page != last_page:
                parts.append(f"[第 {page} 页]")
                last_page = page
            parts.append(text)
    if not parts:
        raise ValueError("文档正文为空")
    return "\n".join(parts)


def remove_structure(data_dir: Path, document_id: str, version: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", version):
        raise ValueError("文档版本无效")
    (_document_directory(data_dir, document_id) / f"{version}.json").unlink(missing_ok=True)


def remove_structures(data_dir: Path, document_id: str) -> None:
    directory = _document_directory(data_dir, document_id)
    if directory.exists():
        for pattern in ("*.json", "structure-*.tmp"):
            for path in directory.glob(pattern):
                path.unlink()
        directory.rmdir()


def remove_all_structures(data_dir: Path) -> None:
    root = _structures_root(data_dir)
    if root.exists():
        for path in root.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", path.name) and (path.is_dir() or path.is_symlink()):
                remove_structures(data_dir, path.name)


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
