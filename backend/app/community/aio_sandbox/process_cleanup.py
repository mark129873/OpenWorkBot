"""按运行标记清理同一沙箱内的协作进程。"""

import json
import os
import re
import select
import signal
import sys
import time


OWNER_ENV = b"OPENWORKBOT_SCOPE_TOKEN="


def matching_processes(token):
    expected = OWNER_ENV + token.encode("ascii")
    handles = []
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit() or int(name) == os.getpid():
                continue
            pid = int(name)
            try:
                with open(f"/proc/{pid}/status", "r", encoding="utf-8") as source:
                    uid_line = next(line for line in source if line.startswith("Uid:"))
                if int(uid_line.split()[1]) != os.getuid():
                    continue
                pidfd = os.pidfd_open(pid)
                try:
                    with open(f"/proc/{pid}/environ", "rb") as source:
                        matched = expected in source.read().split(b"\0")
                except PermissionError:
                    exited = not alive(pidfd)
                    os.close(pidfd)
                    if exited:
                        continue
                    raise
                except BaseException:
                    os.close(pidfd)
                    raise
                if matched:
                    handles.append(pidfd)
                else:
                    os.close(pidfd)
            except (FileNotFoundError, ProcessLookupError, StopIteration):
                continue
    except BaseException:
        for handle in handles:
            os.close(handle)
        raise
    return handles


def alive(handle):
    poller = select.poll()
    poller.register(handle, select.POLLIN)
    return not poller.poll(0)


def cleanup(token):
    if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
        raise ValueError("进程标记无效")
    deadline = time.monotonic() + 5
    empty = 0
    while time.monotonic() < deadline:
        handles = matching_processes(token)
        if not handles:
            empty += 1
            if empty >= 2:
                return
            time.sleep(0.05)
            continue
        empty = 0
        try:
            for handle in handles:
                try:
                    signal.pidfd_send_signal(handle, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            time.sleep(0.2)
            for handle in handles:
                if alive(handle):
                    try:
                        signal.pidfd_send_signal(handle, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        finally:
            for handle in handles:
                os.close(handle)
        time.sleep(0.1)
    remaining = matching_processes(token)
    for handle in remaining:
        os.close(handle)
    if remaining:
        raise RuntimeError("不能确认残留进程已结束")


if __name__ == "__main__":
    try:
        cleanup(sys.argv[1])
        print(json.dumps({"ok": True}), flush=True)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True), flush=True)
        raise SystemExit(1)
