"""Exact cosine search with an in-memory FAISS index and aligned metadata."""

from collections.abc import Sequence

import faiss
import numpy as np
from numpy.typing import NDArray

from .models import Chunk, SearchResult


def _unit_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Own the float32 buffer so normalization cannot mutate a caller's data."""
    matrix = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if matrix.ndim != 2 or not all(matrix.shape):
        raise ValueError("Embeddings must be a non-empty two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must contain only finite values.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Embeddings must have finite, non-zero lengths.")
    matrix /= norms
    return matrix


class FaissIndex:
    def __init__(self, chunks: Sequence[Chunk], vectors: NDArray[np.float32]) -> None:
        matrix = _unit_rows(vectors)
        if len(chunks) != len(matrix):
            raise ValueError("Each chunk must have exactly one embedding row.")
        self._chunks = tuple(chunks)  # FAISS row i always maps to metadata row i.
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("Top-k must be positive.")
        if min_score is not None and not -1 <= min_score <= 1:
            raise ValueError("Minimum score must be between -1 and 1.")
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 1 or query_vector.size != self._index.d:
            raise ValueError("Query dimension must match the indexed embeddings.")
        query = _unit_rows(query_vector[None, :])
        scores, rows = self._index.search(query, min(top_k, len(self._chunks)))
        return [
            SearchResult(chunk=self._chunks[int(row)], score=float(score))
            for score, row in zip(scores[0], rows[0])
            if min_score is None or score >= min_score
        ]
