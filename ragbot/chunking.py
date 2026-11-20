"""Step 2: split text into overlapping fixed-size chunks (sliding window)."""
from __future__ import annotations

from . import config


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.OVERLAP) -> list[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks
