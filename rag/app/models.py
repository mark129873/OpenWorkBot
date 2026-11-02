import json

import httpx
from anthropic import AsyncAnthropic
from openai import APIError, AsyncOpenAI

from app.config import Settings
from app import prompts, rag_parameters as params, tracing


class Models:
    def __init__(self, config: Settings):
        self.config = config

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.config.embedding_model:
            raise RuntimeError("Embedding 模型未配置")
        with tracing.observation("embedding", as_type="embedding", model=self.config.embedding_model,
                                 input=texts) as span:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{self.config.embedding_url}/v1/embeddings",
                                             json={"model": self.config.embedding_model, "input": texts})
                response.raise_for_status()
                data = response.json().get("data")
            if not isinstance(data, list) or len(data) != len(texts):
                raise ValueError("Embedding 响应数量不匹配")
            try:
                data = sorted(data, key=lambda item: int(item["index"]))
                vectors = [item["embedding"] for item in data]
                if ([int(item["index"]) for item in data] != list(range(len(texts)))
                        or any(not isinstance(vector, list) or not vector for vector in vectors)
                        or len({len(vector) for vector in vectors}) != 1):
                    raise ValueError
                tracing.update(span, output={"count": len(vectors), "dimension": len(vectors[0]) if vectors else 0})
                return vectors
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Embedding 响应无效") from exc

    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        if not items:
            return []
        if not self.config.rerank_model:
            raise RuntimeError("重排模型未配置")
        with tracing.observation("rerank", as_type="span",
                                 metadata={"model": self.config.rerank_model, "top_n": len(items)},
                                 input={"query": query, "candidates": tracing.chunk_references(items)}) as span:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(f"{self.config.rerank_url}/v1/rerank", json={
                    "model": self.config.rerank_model, "query": query,
                    "top_n": len(items), "documents": [item["search_text"] for item in items],
                })
                response.raise_for_status()
                results = response.json().get("results")
            if not isinstance(results, list) or len(results) != len(items):
                raise ValueError("重排响应数量不匹配")
            try:
                scores = {int(row["index"]): float(row["relevance_score"]) for row in results}
                if set(scores) != set(range(len(items))):
                    raise ValueError
                ranked = sorted(items, key=lambda item: scores[item["_rank_index"]], reverse=True)
                ranked = [{key: value for key, value in item.items() if key != "search_text"} for item in ranked]
                tracing.update(span, output={"scores": scores, "ranked_candidates": tracing.chunk_references(ranked)})
                return ranked
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("重排响应无效") from exc

    async def text(self, system: str, prompt: str) -> str:
        if self.config.provider not in {"anthropic", "openai"} or not self.config.api_key or not self.config.model:
            raise RuntimeError("Bot 模型未配置")
        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        parameters = ({"max_output_tokens": params.MAX_OUTPUT_TOKENS,
                       "reasoning_effort": self.config.reasoning_effort, "store": False}
                      if self.config.provider == "openai" else {"max_tokens": params.MAX_OUTPUT_TOKENS, "thinking": "disabled"})
        temperature = {} if self.config.temperature is None else {"temperature": self.config.temperature}
        parameters.update(temperature)
        with tracing.observation(
                "llm", as_type="generation", model=self.config.model, model_parameters=parameters,
                metadata={"provider": self.config.provider,
                          "api": "responses" if self.config.provider == "openai" else "messages"},
                input={"system": system, "prompt": prompt}) as span:
            if self.config.provider == "openai":
                kwargs["base_url"] = self.config.base_url or "https://api.openai.com/v1"
                try:
                    async with AsyncOpenAI(**kwargs, timeout=120.0, max_retries=params.MODEL_REQUEST_RETRIES) as client:
                        response = await client.responses.create(
                            model=self.config.model, max_output_tokens=params.MAX_OUTPUT_TOKENS,
                            reasoning={"effort": self.config.reasoning_effort}, store=False,
                            input=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                            **temperature)
                except APIError as exc:
                    tracing.update(span, metadata={"error_type": type(exc).__name__,
                                                   "http_status": getattr(exc, "status_code", None) or "connection"})
                    raise RuntimeError(f"OpenAI 请求失败：{getattr(exc, 'status_code', None) or '连接错误'}") from None
                tracing.update(span, metadata={"finish_reason": response.status,
                                               "incomplete_reason": getattr(response.incomplete_details, "reason", None)})
                if response.status != "completed":
                    raise RuntimeError("模型响应未完成")
                text = response.output_text
                if not text.strip():
                    raise RuntimeError("模型没有返回内容")
                usage = getattr(response, "usage", None)
                tracing.update(span, output=text.strip(), usage_details={"input": getattr(usage, "input_tokens", 0) or 0,
                                                                         "output": getattr(usage, "output_tokens", 0) or 0}
                               if usage else None)
                return text.strip()
            async with AsyncAnthropic(**kwargs, timeout=120.0, max_retries=params.MODEL_REQUEST_RETRIES) as client:
                response = await client.messages.create(model=self.config.model, max_tokens=params.MAX_OUTPUT_TOKENS,
                                                        thinking={"type": "disabled"}, system=system,
                                                        messages=[{"role": "user", "content": prompt}], **temperature)
            tracing.update(span, metadata={"finish_reason": response.stop_reason})
            if response.stop_reason == "max_tokens":
                raise RuntimeError("模型输出超过上限")
            text = "".join(block.text for block in response.content if block.type == "text").strip()
            if not text:
                raise RuntimeError("模型没有返回内容")
            usage = getattr(response, "usage", None)
            tracing.update(span, output=text, usage_details={"input": getattr(usage, "input_tokens", 0) or 0,
                                                              "output": getattr(usage, "output_tokens", 0) or 0}
                           if usage else None)
            return text

    async def structured(self, system: str, prompt: str, default: dict) -> dict:
        for attempt in range(params.JSON_RETRIES + 1):
            raw = await self.text(system, prompt + (prompts.JSON_RETRY_SUFFIX if attempt else ""))
            try:
                start, end = raw.index("{"), raw.rindex("}") + 1
                value = json.loads(raw[start:end])
                if isinstance(value, dict):
                    return value
            except (ValueError, json.JSONDecodeError):
                continue
        return default
