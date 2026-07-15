"""Determinism tests for project-knowledge source evidence."""

from packages.knowledge.chunking import (
    canonical_text,
    chunk_id,
    chunk_text,
    sha256_text,
)


def test_crlf_and_lf_have_identical_hashes_chunks_and_ids() -> None:
    lf = "# Heading\n\nalpha\nbeta\ngamma\n" * 8
    crlf = lf.replace("\n", "\r\n")

    assert canonical_text(crlf) == lf
    assert sha256_text(crlf) == sha256_text(lf)

    lf_chunks = chunk_text(lf, max_chars=80, overlap_chars=12)
    crlf_chunks = chunk_text(crlf, max_chars=80, overlap_chars=12)
    assert lf_chunks == crlf_chunks
    assert len(lf_chunks) > 1
    assert [
        chunk_id(
            project="quantex",
            model="nomic-embed-text",
            dimension=768,
            chunker_version="v1",
            path="wiki/content/example.md",
            chunk=chunk,
        )
        for chunk in lf_chunks
    ] == [
        chunk_id(
            project="quantex",
            model="nomic-embed-text",
            dimension=768,
            chunker_version="v1",
            path="wiki/content/example.md",
            chunk=chunk,
        )
        for chunk in crlf_chunks
    ]


def test_chunk_boundaries_and_ids_are_repeatable_and_provenance_bound() -> None:
    text = "first paragraph\n\n" + ("line of deterministic content\n" * 12)

    first = chunk_text(text, max_chars=96, overlap_chars=16)
    second = chunk_text(text, max_chars=96, overlap_chars=16)

    assert first == second
    assert [chunk.index for chunk in first] == list(range(len(first)))
    assert all(chunk.text for chunk in first)
    assert all(chunk.start_line <= chunk.end_line for chunk in first)

    base_id = chunk_id(
        project="quantex",
        model="nomic-embed-text",
        dimension=768,
        chunker_version="v1",
        path="src/a.py",
        chunk=first[0],
    )
    assert base_id.startswith("kn_")
    assert len(base_id) == 67
    assert base_id != chunk_id(
        project="quantex",
        model="nomic-embed-text",
        dimension=768,
        chunker_version="v1",
        path="src/b.py",
        chunk=first[0],
    )
