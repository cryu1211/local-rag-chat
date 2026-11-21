from pathlib import Path

import numpy as np
import pytest

from local_rag_chat.index import FaissIndex
from local_rag_chat.models import Chunk
from local_rag_chat.retrieval import Retriever


def make_index():
    chunks = [
        Chunk(Path("a/notes.md"), 0, "first"),
        Chunk(Path("b/notes.md"), 0, "second"),
        Chunk(Path("c.txt"), 1, "third"),
    ]
    vectors = np.array([[0, 3], [2, 0], [-4, 0]], dtype=np.float32)
    return FaissIndex(chunks, vectors), chunks, vectors


def test_real_faiss_cosine_ranking_metadata_and_bounded_top_k():
    index, chunks, vectors = make_index()
    query = np.array([8, 0], dtype=np.float32)

    results = index.search(query, top_k=20)

    assert [result.chunk for result in results] == [chunks[1], chunks[0], chunks[2]]
    assert [result.score for result in results] == pytest.approx([1, 0, -1])
    np.testing.assert_array_equal(vectors, [[0, 3], [2, 0], [-4, 0]])
    np.testing.assert_array_equal(query, [8, 0])
    assert len(index.search(query, top_k=1)) == 1


def test_threshold_can_return_no_evidence_and_is_inclusive():
    index, chunks, _ = make_index()
    results = index.search(np.array([1, 1]), min_score=0.8)
    assert results == []
    results = index.search(np.array([1, 0]), min_score=1)
    assert [result.chunk for result in results] == [chunks[1]]


@pytest.mark.parametrize(
    "vectors",
    [np.empty((0, 2)), np.array([1, 2]), np.array([[0, 0]]),
     np.array([[np.nan, 1]]), np.array([[np.inf, 1]])],
)
def test_rejects_invalid_embeddings(vectors):
    with pytest.raises(ValueError):
        FaissIndex([Chunk(Path("a.md"), 0, "text")], vectors)


def test_rejects_metadata_mismatch_and_invalid_search_parameters():
    with pytest.raises(ValueError, match="exactly one"):
        FaissIndex([], np.array([[1, 0]]))
    index, _, _ = make_index()
    for query in [np.array([1, 0, 0]), np.array([[1, 0]]), np.array([0, 0])]:
        with pytest.raises(ValueError):
            index.search(query)
    with pytest.raises(ValueError, match="Top-k"):
        index.search(np.array([1, 0]), top_k=0)
    for cutoff in [1.1, -1.1, float("nan")]:
        with pytest.raises(ValueError, match="Minimum score"):
            index.search(np.array([1, 0]), min_score=cutoff)


def test_retriever_uses_injected_embedder_and_rejects_blank_questions():
    class FakeEmbedder:
        def __init__(self):
            self.inputs = []

        def embed_documents(self, texts):
            raise AssertionError("A query must use query encoding.")

        def embed_query(self, text):
            self.inputs.append(text)
            return np.array([1, 0], dtype=np.float32)

    index, chunks, _ = make_index()
    embedder = FakeEmbedder()
    retriever = Retriever(embedder, index)

    assert retriever.search("question", top_k=1)[0].chunk == chunks[1]
    assert embedder.inputs == ["question"]
    with pytest.raises(ValueError, match="blank"):
        retriever.search(" \n")
    assert embedder.inputs == ["question"]
