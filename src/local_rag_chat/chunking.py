"""Deterministic overlapping character windows, without model dependencies."""

from collections.abc import Iterable

from .models import Chunk, Document


def chunk_documents(
    documents: Iterable[Document], chunk_size: int = 600, overlap: int = 100
) -> list[Chunk]:
    """Preserve original text slices and never join text across documents.

    Sizes are characters, not tokens. The embedding adapter separately rejects
    inputs beyond its model's token limit instead of silently truncating them.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be non-negative and smaller than chunk size.")

    chunks = []
    for document in documents:
        chunk_id = 0
        for start in range(0, len(document.text), chunk_size - overlap):
            end = min(start + chunk_size, len(document.text))
            text = document.text[start:end]
            if text.strip():
                chunks.append(Chunk(document.source, chunk_id, text))
                chunk_id += 1
            if end == len(document.text):
                break  # Do not emit a tail already covered by the last window.
    return chunks
