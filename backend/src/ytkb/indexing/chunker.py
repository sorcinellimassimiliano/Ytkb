"""Timestamp-aware chunker.

Merges transcript segments into chunks of ~target tokens with a configurable
overlap, preferring to cut on sentence boundaries. Every chunk keeps the exact
start/end timestamp of its source segments so citations resolve to mm:ss.

Token counting is approximate (word-based) to stay CPU-only and dependency
free; it only needs to be *consistent*, not exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ytkb.ingestion.providers.base import TranscriptSegment

_SENTENCE_END = re.compile(r"[.!?…](?:\s|$)")


@dataclass(slots=True)
class Chunk:
    start_s: float
    end_s: float
    text: str
    token_count: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per whitespace word, min 1 per word."""
    words = text.split()
    return max(len(words), int(len(words) * 1.3))


def _ends_on_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text[-2:])) or text.endswith((".", "!", "?", "…"))


def chunk_segments(
    segments: list[TranscriptSegment],
    *,
    target_tokens: int = 500,
    overlap_ratio: float = 0.15,
    max_tokens: int | None = None,
) -> list[Chunk]:
    """Group segments into overlapping, sentence-aware chunks.

    Invariants (see tests):
    - chunks are ordered and non-empty when input is non-empty;
    - each chunk's start_s <= end_s and matches segment boundaries;
    - coverage: the first chunk starts at the first segment, the last chunk
      ends at the last segment (no content dropped);
    - overlap only ever repeats earlier content, never invents timestamps.
    """
    if not segments:
        return []

    if max_tokens is None:
        max_tokens = int(target_tokens * 1.4)
    overlap_tokens = int(target_tokens * overlap_ratio)

    chunks: list[Chunk] = []
    cur: list[TranscriptSegment] = []
    cur_tokens = 0

    def flush(carry_overlap: bool) -> None:
        nonlocal cur, cur_tokens
        if not cur:
            return
        text = " ".join(s.text for s in cur).strip()
        chunks.append(
            Chunk(
                start_s=cur[0].start_s,
                end_s=cur[-1].end_s,
                text=text,
                token_count=estimate_tokens(text),
            )
        )
        if not carry_overlap or overlap_tokens <= 0:
            cur, cur_tokens = [], 0
            return
        # Build overlap tail: keep trailing segments up to overlap_tokens.
        tail: list[TranscriptSegment] = []
        acc = 0
        for seg in reversed(cur):
            seg_tok = estimate_tokens(seg.text)
            if acc + seg_tok > overlap_tokens and tail:
                break
            tail.insert(0, seg)
            acc += seg_tok
        cur = tail
        cur_tokens = acc

    for seg in segments:
        seg_tokens = estimate_tokens(seg.text)
        cur.append(seg)
        cur_tokens += seg_tokens

        if cur_tokens >= target_tokens:
            joined = " ".join(s.text for s in cur)
            # Cut here if we hit a sentence boundary or exceeded the hard max.
            if _ends_on_sentence(joined) or cur_tokens >= max_tokens:
                flush(carry_overlap=True)

    flush(carry_overlap=False)
    return chunks
