"""Load explicitly selected UTF-8 Markdown and plain-text files."""

from collections.abc import Iterable
from pathlib import Path

from .models import Document


def document_from_bytes(source: Path, content: bytes) -> Document:
    """Decode file or upload bytes with the same extension and UTF-8 rules."""
    if source.suffix.lower() not in {".md", ".txt"}:
        raise ValueError(f"Unsupported file type: {source}. Use .md or .txt.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"File must be UTF-8 encoded: {source}") from error
    return Document(source=source, text=text)


def load_documents(paths: Iterable[Path]) -> list[Document]:
    """Keep input order, resolve paths, and load each source only once."""
    documents = []
    seen: set[Path] = set()
    for path in paths:
        source = path.expanduser().resolve()
        if source in seen:
            continue
        if source.suffix.lower() not in {".md", ".txt"}:
            raise ValueError(f"Unsupported file type: {source}. Use .md or .txt.")
        if not source.is_file():
            raise ValueError(f"Not a readable file: {source}")
        documents.append(document_from_bytes(source, source.read_bytes()))
        seen.add(source)

    if not documents:
        raise ValueError("Provide at least one .md or .txt file.")
    return documents
