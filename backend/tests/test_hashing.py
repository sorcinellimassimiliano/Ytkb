from ytkb.knowledge.hashing import (
    chunk_deterministic_key,
    normalize_text,
    unit_content_hash,
)


def test_normalize_collapses_whitespace_and_case():
    assert normalize_text("  Hello   WORLD\n") == "hello world"


def test_unit_hash_is_stable_and_type_sensitive():
    h1 = unit_content_hash("Use prompt caching", "technique")
    h2 = unit_content_hash("use   prompt caching", "technique")
    h3 = unit_content_hash("Use prompt caching", "claim")
    assert h1 == h2  # normalization
    assert h1 != h3  # type matters
    assert len(h1) == 64


def test_chunk_key_is_deterministic():
    assert chunk_deterministic_key("abc123", 12.0) == "abc123:12.00"
    assert chunk_deterministic_key("v", 3.1) == "v:3.10"
    # Deterministic: same input always yields the same key.
    assert chunk_deterministic_key("abc123", 12.007) == chunk_deterministic_key("abc123", 12.007)
