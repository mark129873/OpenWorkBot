"""容器内固定文件操作；所有路径由内核限制在用户工作区内。"""

import ctypes
import errno
import hashlib
import io
import json
import os
import posixpath
import re
import stat
import sys
import time
import uuid
from pathlib import Path, PurePosixPath


ROOT = "/mnt/user-data/workspace"
READ_LIMIT = 100000


class OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64), ("resolve", ctypes.c_uint64)]


LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long


def relative_path(path):
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("路径无效")
    value = path.replace("\\", "/")
    if value.startswith("/") or re.match(r"^[a-zA-Z]:", value) or ".." in PurePosixPath(value).parts:
        raise ValueError("路径必须位于工作区内")
    return posixpath.normpath(value)


def safe_open(root_fd, path, flags, mode=0):
    how = OpenHow(flags | os.O_CLOEXEC, mode, 0x08 | 0x02)
    result = LIBC.syscall(437, root_fd, ctypes.c_char_p(os.fsencode(path)), ctypes.byref(how), ctypes.sizeof(how))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), path)
    return result


def open_file(root_fd, path, flags=os.O_RDONLY, mode=0):
    fd = safe_open(root_fd, path, flags | os.O_NONBLOCK, mode)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("目标不是普通文件")
    return fd


def make_parents(root_fd, path):
    parts = PurePosixPath(path).parts[:-1]
    prefix = []
    for part in parts:
        parent = "/".join(prefix) or "."
        parent_fd = safe_open(root_fd, parent, os.O_PATH | os.O_DIRECTORY)
        try:
            try:
                os.mkdir(part, dir_fd=parent_fd)
            except FileExistsError:
                pass
        finally:
            os.close(parent_fd)
        prefix.append(part)
        directory = safe_open(root_fd, "/".join(prefix), os.O_PATH | os.O_DIRECTORY)
        os.close(directory)


def read_file(root_fd, path, offset, limit):
    if type(offset) is not int or offset < 1 or type(limit) is not int or not 1 <= limit <= 2000:
        raise ValueError("offset 必须从 1 开始，limit 必须为 1 到 2000")
    fd = open_file(root_fd, path)
    rows, size, line_no, next_line = [], 0, 0, None
    with io.TextIOWrapper(os.fdopen(fd, "rb"), encoding="utf-8", errors="replace", newline=None) as source:
        while first := source.readline(2001):
            line_no += 1
            long_line = not first.endswith("\n") and len(first) > 2000
            if long_line:
                rest = first
                while rest and not rest.endswith("\n"):
                    rest = source.readline(8192)
            if line_no < offset:
                continue
            if line_no >= offset + limit:
                next_line = line_no
                break
            row = f"{line_no}: {first[:2000].rstrip(chr(10)).rstrip(chr(13))}"
            if long_line:
                row += " …（该行超过 2000 字符，余下内容已截断）"
            if size + len(row) + 1 > READ_LIMIT - 100:
                next_line = line_no
                break
            rows.append(row)
            size += len(row) + 1
    if next_line is not None:
        rows.append(f"[继续读取：offset={next_line}]")
    return "\n".join(rows) if rows else "(文件为空或已到文件末尾)"


def file_version(root_fd, path):
    try:
        fd = open_file(root_fd, path)
    except FileNotFoundError:
        return "missing"
    digest = hashlib.sha256()
    with os.fdopen(fd, "rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_file(root_fd, path, content, windows_newline, append=False):
    if not isinstance(content, str):
        raise ValueError("文件内容必须是文本")
    if type(append) is not bool:
        raise ValueError("追加参数无效")
    make_parents(root_fd, path)
    fd = open_file(root_fd, path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else 0), 0o666)
    with os.fdopen(fd, "wb") as target:
        if not append:
            target.truncate(0)
        target.write((content.replace("\n", "\r\n") if windows_newline else content).encode("utf-8"))
    return f"已写入 {path}（{len(content)} 字符）"


def edit_file(root_fd, path, old_text, new_text, windows_newline):
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        raise ValueError("替换内容必须是文本")
    fd = open_file(root_fd, path, os.O_RDWR)
    with os.fdopen(fd, "r+b") as target:
        text = target.read().decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        count = text.count(old_text)
        if count == 0:
            raise ValueError("old_text 在文件中未找到")
        if count > 1:
            raise ValueError(f"old_text 出现了 {count} 次，要求恰好 1 次；请提供更长的片段")
        changed = text.replace(old_text, new_text, 1)
        if windows_newline:
            changed = changed.replace("\n", "\r\n")
        target.seek(0)
        target.truncate()
        target.write(changed.encode("utf-8"))
    return f"已编辑 {path}"


def glob_files(root_fd, pattern):
    matches = []
    for value in Path(ROOT).glob(pattern):
        relative = value.relative_to(ROOT).as_posix()
        try:
            fd = open_file(root_fd, relative)
        except IsADirectoryError:
            continue
        except ValueError as exc:
            if str(exc) == "目标不是普通文件":
                continue
            raise
        except OSError as exc:
            if exc.errno == 2:
                continue
            raise ValueError("glob 匹配越出工作区，请缩小匹配范围") from exc
        os.close(fd)
        matches.append(relative)
    matches.sort()
    if not matches:
        return "(无匹配文件)"
    text = "\n".join(matches)
    return text if len(text) <= READ_LIMIT else f"...(前段已截断)...\n{text[-READ_LIMIT:]}"


def delete_file(root_fd, path):
    fd = open_file(root_fd, path)
    try:
        actual = os.readlink(f"/proc/self/fd/{fd}")
        real_root = os.readlink(f"/proc/self/fd/{root_fd}")
        if not actual.startswith(real_root.rstrip("/") + "/"):
            raise ValueError("路径越出工作区")
        target = posixpath.relpath(actual, real_root)
        parent, name = posixpath.split(target)
        parent_fd = safe_open(root_fd, parent or ".", os.O_PATH | os.O_DIRECTORY)
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(fd)
            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("文件已变化，请重新读取")
            make_parents(root_fd, ".rubbish/entry")
            trash_fd = safe_open(root_fd, ".rubbish", os.O_PATH | os.O_DIRECTORY)
            try:
                trash_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex}_{name}"
                rename = getattr(LIBC, "renameat2", None)
                code = errno.ENOSYS
                if rename is not None:
                    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
                    code = 0 if rename(parent_fd, os.fsencode(name), trash_fd, os.fsencode(trash_name), 1) == 0 else ctypes.get_errno()
                if code in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
                    os.link(name, trash_name, src_dir_fd=parent_fd, dst_dir_fd=trash_fd, follow_symlinks=False)
                    try:
                        source = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                        saved = os.stat(trash_name, dir_fd=trash_fd, follow_symlinks=False)
                        identity = (opened.st_dev, opened.st_ino)
                        if (source.st_dev, source.st_ino) != identity or (saved.st_dev, saved.st_ino) != identity:
                            raise RuntimeError("文件已变化，回收站可能已有副本；请核查")
                        os.unlink(name, dir_fd=parent_fd)
                    except Exception as exc:
                        raise RuntimeError("文件删除结果未知，回收站可能已有副本；请核查") from exc
                elif code:
                    raise OSError(code, os.strerror(code))
            finally:
                os.close(trash_fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(fd)
    return f"已删除 {path}（移入 .rubbish/{trash_name}，可找回）"


def execute(request):
    operation = request.get("operation")
    args = request.get("args")
    if not isinstance(args, dict):
        raise ValueError("文件工具参数无效")
    root_fd = os.open(ROOT, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        if operation == "glob":
            return glob_files(root_fd, relative_path(args.get("pattern")))
        path = relative_path(args.get("path"))
        if operation == "version":
            return file_version(root_fd, path)
        if operation == "read_file":
            return read_file(root_fd, path, args.get("offset", 1), args.get("limit", 2000))
        if operation == "write_file":
            return write_file(root_fd, path, args.get("content"), request.get("windows_newline") is True,
                              args.get("append", False))
        if operation == "edit_file":
            return edit_file(root_fd, path, args.get("old_text"), args.get("new_text"),
                             request.get("windows_newline") is True)
        if operation == "delete_file":
            return delete_file(root_fd, path)
        raise ValueError("未知文件操作")
    finally:
        os.close(root_fd)


if __name__ == "__main__":
    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict):
            raise ValueError("文件操作请求无效")
        result = {"ok": True, "output": execute(request)}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(json.dumps(result, ensure_ascii=True), flush=True)
