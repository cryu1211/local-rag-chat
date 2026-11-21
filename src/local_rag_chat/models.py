"""Values passed between pipeline stages; no framework or runtime objects."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    source: Path
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    source: Path
    chunk_id: int  # Zero-based within a document; (source, chunk_id) is the identity.
    text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float  # Cosine similarity, not a probability or confidence level.
