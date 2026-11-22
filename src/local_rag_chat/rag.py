"""Query orchestration over injected retrieval, prompt, and generation components."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .generation import CONTEXT_WINDOW, LLMClient
from .models import SearchResult
from .prompting import INSUFFICIENT_CONTEXT_ANSWER, PreparedPrompt, build_prompt
from .tokenization import PromptTokenizer


# the threshold is heuristic, dataset-dependent, model-dependent, and query-dependent;
# callers can override it per query
DEFAULT_MIN_SCORE = 0.3


class NoDocumentsError(ValueError):
    """A query was attempted before any non-blank document chunks were indexed."""


class ChunkRetriever(Protocol):
    @property
    def chunk_count(self) -> int:
        ...

    def search(
        self, question: str, top_k: int = 5, min_score: float | None = None
    ) -> list[SearchResult]:
        """Return ranked matches retained by the optional similarity cutoff."""
        ...


RAGStatus = Literal["generated", "no_retrieval_matches", "no_context_fits"]


@dataclass(frozen=True, slots=True)
class RAGResult:
    question: str
    answer: str
    status: RAGStatus
    retrieved: tuple[SearchResult, ...]
    prompt: PreparedPrompt
    indexed_chunks: int
    top_k: int
    min_score: float | None

    @property
    def sources(self) -> tuple[SearchResult, ...]:
        """Exact chunks supplied as context, independent of generated citations."""
        return self.prompt.selected

    @property
    def source_files(self) -> tuple[Path, ...]:
        """Unique context files in selection order; equal basenames stay distinct."""
        return tuple(dict.fromkeys(hit.chunk.source for hit in self.sources))

    def debug_info(self) -> dict[str, object]:
        return {
            "question": self.question,
            "status": self.status,
            "source_files": [str(path) for path in self.source_files],
            "retrieval": {
                "indexed_chunks": self.indexed_chunks,
                "top_k": self.top_k,
                "metric": "cosine_similarity",
                "min_score": self.min_score,
                "threshold_inclusive": True,
                "matches": [
                    {
                        "rank": i, "source": hit.chunk.source.name,
                        "path": str(hit.chunk.source), "chunk_id": hit.chunk.chunk_id,
                        "score": hit.score, "text": hit.chunk.text,
                        "selected": hit in self.sources,
                    }
                    for i, hit in enumerate(self.retrieved, start=1)
                ],
            },
            "prompt": self.prompt.debug_info(),
        }


class RAGService:
    """Coordinate one stateless question; ingestion and runtime setup live elsewhere.

    The injected LLM must accept the tokenizer's fully serialized prompt without
    adding its own template. For Ollama, configure the same prompt_tokenizer on
    the client to enable its bounded raw-prompt path.
    """

    def __init__(
        self, retriever: ChunkRetriever, llm: LLMClient, tokenizer: PromptTokenizer
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._tokenizer = tokenizer

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float | None = DEFAULT_MIN_SCORE,
        max_context_tokens: int = CONTEXT_WINDOW,
        reserved_output_tokens: int = 512,
        system_instructions: str | None = None,
    ) -> RAGResult:
        if self._retriever.chunk_count <= 0:
            raise NoDocumentsError(
                "No documents are indexed. Ingest non-blank .md or .txt files before querying."
            )

        # Reject an oversized fixed prompt before the query embedding or search.
        prompt = build_prompt(
            question, [], self._tokenizer, max_context_tokens,
            reserved_output_tokens, system_instructions,
        )
        retrieved = tuple(self._retriever.search(question, top_k=top_k, min_score=min_score))
        status: RAGStatus
        if not retrieved:
            status = "no_retrieval_matches"
            answer = INSUFFICIENT_CONTEXT_ANSWER
        else:
            prompt = build_prompt(
                question, retrieved, self._tokenizer, max_context_tokens,
                reserved_output_tokens, system_instructions,
            )
            if not prompt.selected:
                status = "no_context_fits"
                answer = INSUFFICIENT_CONTEXT_ANSWER
            else:
                status = "generated"
                # LLMError and PromptBudgetError propagate; never turn a failed
                # request into a fabricated answer or an unrelated model fallback.
                answer = self._llm.generate(prompt.text, max_tokens=prompt.reserved_output_tokens)

        return RAGResult(
            question=question, answer=answer, status=status, retrieved=retrieved,
            prompt=prompt, indexed_chunks=self._retriever.chunk_count,
            top_k=top_k, min_score=min_score,
        )
