import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def command(model: str, url: str, mode: str) -> list[str]:
    if not model or not Path(model).expanduser().is_file():
        raise ValueError(f"{mode} 模型文件不存在或未配置")
    address = urlsplit(url)
    if address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"} or not address.port:
        raise ValueError("模型服务必须使用带端口的本机 HTTP 地址")
    result = ["llama", "serve", "--model", model, mode, "--host", address.hostname,
              "--port", str(address.port), "--no-ui"]
    context = int(os.getenv("LLAMA_CONTEXT_SIZE", "4096"))
    if context < 0:
        raise ValueError("模型上下文长度无效")
    if context:
        result += ["--ctx-size", str(context)]
    if device := os.getenv("LLAMA_DEVICE", "").strip():
        result += ["--device", device]
    return result


def main() -> int:
    commands = [
        command(os.getenv("EMBEDDING_MODEL_PATH", ""),
                os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11436"), "--embedding"),
        command(os.getenv("RERANKER_MODEL_PATH", ""),
                os.getenv("RERANKER_BASE_URL", "http://127.0.0.1:11435"), "--rerank"),
    ]
    processes = []
    try:
        for args in commands:
            processes.append(subprocess.Popen(args, cwd=ROOT))
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        print("本地 RAG 模型服务已停止；普通聊天继续可用。", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"本地 RAG 模型启动失败：{error}", file=sys.stderr)
        raise SystemExit(1)
