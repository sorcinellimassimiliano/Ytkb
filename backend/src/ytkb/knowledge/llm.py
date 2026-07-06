"""LLM transport abstraction (Anthropic). Isolated so the knowledge and rag
layers depend on an interface, not a vendor. When no API key is configured the
factory returns None and callers fall back to the offline heuristic extractor,
keeping the pipeline runnable CPU-only with zero cloud dependency."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ytkb.config import Settings, get_settings

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class LLMClient(ABC):
    model: str

    @abstractmethod
    async def complete(
        self, *, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.0
    ) -> str: ...


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(4))
    async def complete(
        self, *, system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.0
    ) -> str:
        import httpx

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                ANTHROPIC_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )


def get_llm_client(
    settings: Settings | None = None, *, model: str | None = None
) -> LLMClient | None:
    settings = settings or get_settings()
    if not settings.anthropic_api_key:
        return None
    return AnthropicClient(settings.anthropic_api_key, model or settings.llm_extraction_model)
