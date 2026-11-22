"""Convert a text question into ranked chunks without knowing about a UI or LLM."""

from .embeddings import Embedder
from .index import FaissIndex
from .models import SearchResult


class Retriever:
    def __init__(self, embedder: Embedder, index: FaissIndex) -> None:
        self._embedder = embedder
        self._index = index

    @property
    def chunk_count(self) -> int:
        return self._index.chunk_count

    def search(
        self, question: str, top_k: int = 5, min_score: float | None = None
    ) -> list[SearchResult]:
        if not question.strip():
            raise ValueError("Question must not be blank.")
        query_vector = self._embedder.embed_query(question)
        return self._index.search(query_vector, top_k=top_k, min_score=min_score)
