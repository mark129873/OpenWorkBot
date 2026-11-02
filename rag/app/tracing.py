import hashlib
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from app import prompts

logger = logging.getLogger(__name__)
_client = ContextVar("rag_trace_client", default=None)


def enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "true").strip().lower() in {"true", "1", "yes", "on"}


def prompt_version():
    values = [f"{key}={value}" for key, value in sorted(vars(prompts).items())
              if key.isupper() and isinstance(value, str)]
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def chunk_references(items):
    return [{key: value for key, value in item.items() if key not in {"content", "search_text"}}
            if isinstance(item, dict) else item for item in items]


def compact(value, fields=("candidates", "rerank_candidates", "evidence")):
    if not isinstance(value, dict):
        return value
    return {key: chunk_references(field) if key in fields and isinstance(field, list) else field
            for key, field in value.items()}


class GraphCallback(CallbackHandler):
    run_inline = True

    def on_chain_start(self, serialized, inputs, *, parent_run_id=None, **kwargs):
        return super().on_chain_start(serialized, compact(inputs) if parent_run_id else inputs,
                                      parent_run_id=parent_run_id, **kwargs)

    def on_chain_end(self, outputs, *, parent_run_id=None, **kwargs):
        if parent_run_id:
            outputs = compact(outputs)
            if "inputs" in kwargs:
                kwargs["inputs"] = compact(kwargs["inputs"])
        else:
            outputs = compact(outputs, ("candidates", "rerank_candidates"))
        return super().on_chain_end(outputs, parent_run_id=parent_run_id, **kwargs)


def configured_client():
    return Langfuse()


def _warning(exc: Exception) -> None:
    logger.warning("RAG 追踪不可用：%s", type(exc).__name__)


def update(span, **fields) -> None:
    if span is not None:
        try:
            span.update(**fields)
        except Exception as exc:
            _warning(exc)


@contextmanager
def observation(name: str, **fields):
    client = _client.get()
    if client is None:
        yield None
        return
    try:
        manager = client.start_as_current_observation(name=name, **fields)
        span = manager.__enter__()
    except Exception as exc:
        _warning(exc)
        yield None
        return
    try:
        yield span
    except BaseException as exc:
        update(span, level="ERROR", status_message=type(exc).__name__)
        raise
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            _warning(exc)


@contextmanager
def query(query: str, operation_id: str):
    if not enabled():
        yield None
        return
    configured = [bool(os.getenv(name)) for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")]
    if not all(configured):
        if any(configured):
            logger.warning("RAG 追踪未启用：Langfuse 配置不完整")
        yield None
        return
    try:
        client = configured_client()
        trace_id = Langfuse.create_trace_id(seed=operation_id)
    except Exception as exc:
        _warning(exc)
        yield None
        return
    token = _client.set(client)
    try:
        with observation("rag_search", as_type="tool", input={"query": query},
                         trace_context={"trace_id": trace_id}, metadata={"operation_id": operation_id}) as span:
            if span is None:
                _client.reset(token)
                token = None
                yield None
            else:
                propagation = None
                try:
                    propagation = propagate_attributes(metadata={"operation_id": operation_id})
                    propagation.__enter__()
                except Exception as exc:
                    propagation = None
                    _warning(exc)
                try:
                    yield span
                finally:
                    if propagation:
                        try:
                            propagation.__exit__(None, None, None)
                        except Exception as exc:
                            _warning(exc)
    finally:
        if token is not None:
            _client.reset(token)


def graph_config() -> dict | None:
    if _client.get() is None:
        return None
    try:
        return {"callbacks": [GraphCallback(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"))], "run_name": "rag-langgraph"}
    except Exception as exc:
        _warning(exc)
        return None
