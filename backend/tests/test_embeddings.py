import math

import pytest

from ytkb.indexing.embeddings import FakeEmbeddingClient, get_embedding_client


@pytest.mark.asyncio
async def test_fake_embeddings_deterministic_and_normalized():
    client = FakeEmbeddingClient(dim=1536)
    a = await client.embed_one("context management in agents")
    b = await client.embed_one("context management in agents")
    assert a == b
    assert len(a) == 1536
    norm = math.sqrt(sum(x * x for x in a))
    assert pytest.approx(norm, abs=1e-6) == 1.0


@pytest.mark.asyncio
async def test_different_texts_differ():
    client = FakeEmbeddingClient(dim=384)
    a = await client.embed_one("prompt caching")
    b = await client.embed_one("vector database")
    assert a != b


def test_default_provider_is_offline_fake():
    client = get_embedding_client()
    assert isinstance(client, FakeEmbeddingClient)
    assert client.dim == 1536
