from pathlib import Path

import pytest

from local_rag_chat.chunking import chunk_documents
from local_rag_chat.loading import document_from_bytes, load_documents
from local_rag_chat.models import Document


@pytest.mark.architecture
def test_loads_utf8_files_preserves_metadata_and_deduplicates(tmp_path):
    markdown = tmp_path / "notes.MD"
    plain = tmp_path / "notes.txt"
    markdown.write_text("\ufeff# Notes\nCafé", encoding="utf-8")
    plain.write_text("Plain text", encoding="utf-8")

    documents = load_documents([markdown, plain, markdown])

    assert documents == [
        Document(markdown.resolve(), "# Notes\nCafé"),
        Document(plain.resolve(), "Plain text"),
    ]


@pytest.mark.architecture
def test_rejects_unsupported_missing_and_non_utf8_files(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        load_documents([tmp_path / "file.pdf"])
    with pytest.raises(ValueError, match="Not a readable file"):
        load_documents([tmp_path / "missing.md"])
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        load_documents([invalid])
    with pytest.raises(ValueError, match="at least one"):
        load_documents([])


def test_upload_bytes_use_the_same_text_loading_rules():
    assert document_from_bytes(Path("notes.MD"), b"\xef\xbb\xbf# Notes") == Document(
        Path("notes.MD"), "# Notes",
    )
    with pytest.raises(ValueError, match="Unsupported file type"):
        document_from_bytes(Path("notes.pdf"), b"text")
    with pytest.raises(ValueError, match="UTF-8"):
        document_from_bytes(Path("notes.txt"), b"\xff")


def test_chunk_overlap_covers_text_without_redundant_tail():
    source = Path("notes.md")
    chunks = chunk_documents([Document(source, "abcdefghij")], chunk_size=6, overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert [chunk.chunk_id for chunk in chunks] == [0, 1]
    assert all(chunk.source == source for chunk in chunks)
    assert chunks[0].text + chunks[1].text[2:] == "abcdefghij"


@pytest.mark.architecture
def test_chunks_preserve_tail_whitespace_and_document_boundaries():
    documents = [
        Document(Path("a/notes.txt"), " abcdefghi "),
        Document(Path("b/notes.txt"), "second"),
        Document(Path("empty.txt"), " \n\t"),
    ]
    chunks = chunk_documents(documents, chunk_size=6, overlap=1)

    assert [chunk.text for chunk in chunks] == [" abcde", "efghi ", "second"]
    assert [chunk.chunk_id for chunk in chunks] == [0, 1, 0]
    assert chunks[0].source != chunks[-1].source


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)])
def test_invalid_chunk_sizes(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=size, overlap=overlap)
