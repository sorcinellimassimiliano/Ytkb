"""Unit tests for taxonomy helpers (no DB)."""

from __future__ import annotations

import math

from ytkb.knowledge.taxonomy import slugify, title_from_unit, update_centroid


def test_slugify_ascii_folds_accents_and_symbols():
    assert slugify("Context Management!") == "context-management"
    assert slugify("Città & Verità") == "citta-verita"
    assert slugify("") == "topic"


def test_title_from_unit_drops_stopwords():
    title = title_from_unit("Il context management degli agenti riduce i token")
    assert "context" in title.lower()
    assert not title.lower().startswith("il ")


def test_update_centroid_from_empty_returns_vector():
    assert update_centroid(None, 0, [1.0, 2.0]) == [1.0, 2.0]


def test_update_centroid_incremental_mean():
    # mean of [0,0] (count 1) and [2,4] → [1,2]
    assert update_centroid([0.0, 0.0], 1, [2.0, 4.0]) == [1.0, 2.0]
    # folding an identical vector keeps the centroid stable
    assert update_centroid([1.0, 1.0], 5, [1.0, 1.0]) == [1.0, 1.0]


def test_update_centroid_matches_running_average():
    vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    centroid: list[float] | None = None
    for i, v in enumerate(vecs):
        centroid = update_centroid(centroid, i, v)
    expected = [sum(c) / len(vecs) for c in zip(*vecs, strict=True)]
    assert all(math.isclose(a, b) for a, b in zip(centroid, expected, strict=True))
