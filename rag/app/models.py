import json

import httpx
from anthropic import AsyncAnthropic
from openai import APIError, AsyncOpenAI

from app.config import Settings


class Models:
    def __init__(self, config: Settings):
        self.config = config

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.config.embedding_model:
            raise RuntimeError("Embedding 模型未配置")
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
            return vectors
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Embedding 响应无效") from exc

    async def rerank(self, query: str, items: list[dict]) -> list[dict]:
        if not items:
            return []
        if not self.config.rerank_model:
            raise RuntimeError("重排模型未配置")
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.config.rerank_url}/v1/rerank", json={
                "model": self.config.rerank_model, "query": query,
                "top_n": len(items), "documents": [item["content"] for item in items],
            })
            response.raise_for_status()
            results = response.json().get("results")
        if not isinstance(results, list) or len(results) != len(items):
            raise ValueError("重排响应数量不匹配")
        try:
            scores = {int(row["index"]): float(row["relevance_score"]) for row in results}
            if set(scores) != set(range(len(items))):
                raise ValueError
            return sorted(items, key=lambda item: scores[item["_rank_index"]], reverse=True)[:10]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("重排响应无效") from exc

    async def text(self, system: str, prompt: str) -> str:
        if self.config.provider not in {"anthropic", "openai"} or not self.config.api_key or not self.config.model:
            raise RuntimeError("Bot 模型未配置")
        kwargs = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.provider == "openai":
            kwargs["base_url"] = self.config.base_url or "https://api.openai.com/v1"
            try:
                async with AsyncOpenAI(**kwargs, timeout=120.0, max_retries=1) as client:
                    response = await client.chat.completions.create(
                        model=self.config.model, max_completion_tokens=4096, reasoning_effort="none", store=False,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            except APIError as exc:
                raise RuntimeError(f"OpenAI 请求失败：{getattr(exc, 'status_code', None) or '连接错误'}") from None
            if not response.choices or response.choices[0].finish_reason != "stop":
                raise RuntimeError("模型响应未完成")
            text = response.choices[0].message.content or ""
            if not text.strip():
                raise RuntimeError("模型没有返回内容")
            return text.strip()
        async with AsyncAnthropic(**kwargs, timeout=120.0, max_retries=1) as client:
            response = await client.messages.create(model=self.config.model, max_tokens=4096,
                                                    thinking={"type": "disabled"}, system=system,
                                                    messages=[{"role": "user", "content": prompt}])
        if response.stop_reason == "max_tokens":
            raise RuntimeError("模型输出超过上限")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise RuntimeError("模型没有返回内容")
        return text

    async def structured(self, system: str, prompt: str, default: dict) -> dict:
        for attempt in range(2):
            raw = await self.text(system, prompt + ("\n只输出 JSON 对象。" if attempt else ""))
            try:
                start, end = raw.index("{"), raw.rindex("}") + 1
                value = json.loads(raw[start:end])
                if isinstance(value, dict):
                    return value
            except (ValueError, json.JSONDecodeError):
                continue
        return default
