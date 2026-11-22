"""Shared component setup for CLI and UI; no presentation or query logic."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_documents
from .embeddings import SentenceTransformerEmbedder
from .index import FaissIndex
from .models import Document
from .ollama import OllamaClient
from .rag import NoDocumentsError, RAGService
from .retrieval import Retriever
from .tokenization import load_prompt_tokenizer


@dataclass(frozen=True, slots=True)
class IndexedCorpus:
    service: RAGService
    file_count: int
    chunk_count: int


def index_documents(
    documents: Sequence[Document],
    model_path: Path,
    tokenizer_path: Path,
    model: str = "qwen2.5:1.5b",
    host: str = "http://127.0.0.1:11434",
    timeout: float = 120,
    chunk_size: int = 600,
    overlap: int = 100,
) -> IndexedCorpus:
    """Build one in-memory corpus using the same local components in every UI."""
    chunks = chunk_documents(documents, chunk_size, overlap)
    if not chunks:
        raise NoDocumentsError("No documents are indexed: the supplied files contain no non-blank text.")
    tokenizer = load_prompt_tokenizer(model, tokenizer_path)
    llm = OllamaClient(model=model, host=host, timeout=timeout, prompt_tokenizer=tokenizer)
    embedder = SentenceTransformerEmbedder(model_path)
    index = FaissIndex(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return IndexedCorpus(
        service=RAGService(Retriever(embedder, index), llm, tokenizer),
        file_count=len({chunk.source for chunk in chunks}),
        chunk_count=index.chunk_count,
    )
