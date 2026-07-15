"""Cross-platform canonicalization and deterministic text chunking."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass


def canonical_text(text: str) -> str:
    """Normalize Unicode and line endings without changing semantic spacing."""

    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def sha256_text(text: str) -> str:
    """Return a stable SHA-256 for canonical UTF-8 text."""

    return hashlib.sha256(canonical_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One deterministic source segment with line-level provenance."""

    index: int
    text: str
    sha256: str
    start_line: int
    end_line: int


def _preferred_boundary(text: str, start: int, hard_end: int, minimum: int) -> int:
    """Prefer paragraph and line boundaries without creating tiny chunks."""

    if hard_end >= len(text):
        return len(text)
    paragraph = text.rfind("\n\n", minimum, hard_end)
    if paragraph >= minimum:
        return paragraph + 2
    line = text.rfind("\n", minimum, hard_end)
    if line >= minimum:
        return line + 1
    return hard_end


def chunk_text(
    text: str, *, max_chars: int, overlap_chars: int
) -> tuple[TextChunk, ...]:
    """Split text deterministically while retaining stable source line ranges."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")

    normalized = canonical_text(text)
    if not normalized:
        return ()

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(normalized):
        hard_end = min(start + max_chars, len(normalized))
        minimum = min(hard_end, start + max(max_chars // 2, 1))
        end = _preferred_boundary(normalized, start, hard_end, minimum)
        segment = normalized[start:end]
        if not segment:
            raise RuntimeError("chunker made no progress")
        start_line = normalized.count("\n", 0, start) + 1
        end_line = start_line + segment.count("\n")
        chunks.append(
            TextChunk(
                index=index,
                text=segment,
                sha256=sha256_text(segment),
                start_line=start_line,
                end_line=end_line,
            )
        )
        if end >= len(normalized):
            break
        next_start = end - overlap_chars
        if overlap_chars:
            line_boundary = normalized.find("\n", next_start, end)
            if line_boundary != -1:
                next_start = line_boundary + 1
        start = max(next_start, start + 1)
        index += 1
    return tuple(chunks)


def chunk_id(
    *,
    project: str,
    model: str,
    dimension: int,
    chunker_version: str,
    path: str,
    chunk: TextChunk,
) -> str:
    """Build an immutable ID from provenance rather than process state."""

    payload = "\x1f".join(
        (
            project,
            model,
            str(dimension),
            chunker_version,
            path,
            str(chunk.index),
            chunk.sha256,
        )
    )
    return f"kn_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
