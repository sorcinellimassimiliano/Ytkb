"""Taxonomy utilities: slugs, incremental centroid math, title heuristics.

Pure functions (no DB) so they're unit-tested directly; the assignment service
and the future topics-review command both build on them.
"""

from __future__ import annotations

import re
import unicodedata

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {
    "il",
    "lo",
    "la",
    "i",
    "gli",
    "le",
    "un",
    "uno",
    "una",
    "di",
    "a",
    "da",
    "in",
    "con",
    "su",
    "per",
    "tra",
    "fra",
    "e",
    "che",
    "come",
    "the",
    "of",
    "to",
    "and",
    "for",
    "with",
    "is",
    "are",
    "del",
    "della",
    "dei",
    "delle",
    "un'",
}


def slugify(text: str, *, max_len: int = 60) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_text).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "topic"


def title_from_unit(text: str, *, max_words: int = 5) -> str:
    """Cheap topic title: the first few significant words of a unit. A better
    title can be set later by the LLM arbiter or `rebuild-topic`."""
    words = [w for w in re.findall(r"\w+", text) if w.lower() not in _STOPWORDS]
    picked = words[:max_words] if words else re.findall(r"\w+", text)[:max_words]
    title = " ".join(picked).strip()
    return (title[:1].upper() + title[1:]) if title else "Nuovo argomento"


def update_centroid(centroid: list[float] | None, count: int, new_vec: list[float]) -> list[float]:
    """Incremental mean: fold new_vec into the running centroid of `count` items."""
    if centroid is None or count <= 0:
        return list(new_vec)
    return [(c * count + v) / (count + 1) for c, v in zip(centroid, new_vec, strict=True)]
