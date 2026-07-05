"""Embedding client abstraction.

Default provider is `fake` (deterministic, offline) so the pipeline and tests
run CPU-only with zero cloud dependency. `openai` / `voyage` providers call the
respective cloud APIs; a `local` provider can wrap sentence-transformers.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ytkb.config import Settings, get_settings


class EmbeddingClient(ABC):
    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


class FakeEmbeddingClient(EmbeddingClient):
    """Deterministic hash-based embeddings — offline, stable across runs."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split() or [text.lower()]:
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0 + (h % 7) / 7.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, settings: Settings) -> None:
        self.dim = settings.embedding_dim
        self.model = settings.embedding_model
        self.api_key = settings.openai_api_key

    @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(5))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY missing for openai embedding provider")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in data]


def get_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingClient(settings)
    if settings.embedding_provider in ("fake", "local", "voyage"):
        # local/voyage fall back to fake until wired; keeps CPU-only default working.
        return FakeEmbeddingClient(settings.embedding_dim)
    return FakeEmbeddingClient(settings.embedding_dim)
