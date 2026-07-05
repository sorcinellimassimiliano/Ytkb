"""Deterministic IDs / content hashing — the backbone of pipeline idempotency.

Rerunning any stage must never duplicate rows, so IDs derive from content:
- chunk identity: (yt_video_id, start_s)
- knowledge unit identity: normalized text hash
"""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def unit_content_hash(text: str, unit_type: str) -> str:
    payload = f"{unit_type}:{normalize_text(text)}".encode()
    return hashlib.sha256(payload).hexdigest()


def chunk_deterministic_key(yt_video_id: str, start_s: float) -> str:
    return f"{yt_video_id}:{start_s:.2f}"
