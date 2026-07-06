"""Chat responders: stream tokens from an LLM given a grounded prompt.

AnthropicChatResponder streams from the Anthropic Messages API. When no API key
is configured `get_responder` returns None and the chat service falls back to a
deterministic, offline responder — so the feature degrades gracefully and tests
run without a cloud LLM.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ytkb.config import Settings, get_settings
from ytkb.knowledge.llm import ANTHROPIC_URL, ANTHROPIC_VERSION
from ytkb.rag.retrieval import RetrievedContext


class ChatResponder(ABC):
    @abstractmethod
    def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]: ...


class AnthropicChatResponder(ChatResponder):
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        import httpx

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", ANTHROPIC_URL, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")


class OfflineResponder(ChatResponder):
    """Deterministic fallback: no LLM. Grounds a short answer in the retrieved
    sources so the chat endpoint stays functional (and testable) offline."""

    def __init__(self, context: RetrievedContext) -> None:
        self.context = context

    async def stream(self, *, system: str, prompt: str) -> AsyncIterator[str]:
        if self.context.is_empty:
            yield "Non ho abbastanza contesto nella knowledge base per rispondere."
            return
        cited = ", ".join(f"[{s.label}]" for s in self.context.sources[:3])
        yield "In base alla knowledge base "
        yield f"({len(self.context.sources)} fonti pertinenti): "
        yield f"vedi {cited}." if cited else "nessuna citazione disponibile."


def get_responder(settings: Settings | None = None) -> ChatResponder | None:
    settings = settings or get_settings()
    if not settings.anthropic_api_key:
        return None
    return AnthropicChatResponder(settings.anthropic_api_key, settings.llm_chat_model)
