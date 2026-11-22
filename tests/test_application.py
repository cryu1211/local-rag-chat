"""Model-free tests across the real loader, application setup, and RAG service."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from local_rag_chat import application
from local_rag_chat.loading import load_documents
from local_rag_chat.prompting import INSUFFICIENT_CONTEXT_ANSWER


pytestmark = pytest.mark.architecture
FIXTURES = Path(__file__).parent / "fixtures/rag"
LIMIT_QUESTION = "What is the context window?"
RUNTIME_QUESTION = "Which runtime generates answers?"
UNRELATED_QUESTION = "How do I bake bread?"


class FixtureEmbedder:
    """Explicit vectors control relevance; this is not a semantic quality test."""

    def __init__(self, documents):
        self.document_vectors = {
            documents[0].text: [1, 0, 0],
            documents[1].text: [0.6, 0.8, 0],
        }
        self.document_batches = []
        self.questions = []

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return np.array([self.document_vectors[text] for text in texts], dtype=np.float32)

    def embed_query(self, text):
        self.questions.append(text)
        return np.array({
            LIMIT_QUESTION: [1, 0, 0],
            RUNTIME_QUESTION: [0.6, 0.8, 0],
            UNRELATED_QUESTION: [0, 0, 1],
        }[text], dtype=np.float32)


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens):
        self.calls.append((prompt, max_tokens))
        return "Deterministic test response."


@pytest.fixture
def indexed_corpus(monkeypatch, tmp_path, qwen_tokenizer):
    documents = load_documents([FIXTURES / "limits.md", FIXTURES / "runtime.txt"])
    embedder = FixtureEmbedder(documents)
    llm = RecordingLLM()
    embedder_factory = Mock(return_value=embedder)
    llm_factory = Mock(return_value=llm)
    tokenizer_factory = Mock(return_value=qwen_tokenizer)
    monkeypatch.setattr(application, "SentenceTransformerEmbedder", embedder_factory)
    monkeypatch.setattr(application, "OllamaClient", llm_factory)
    monkeypatch.setattr(application, "load_prompt_tokenizer", tokenizer_factory)
    model_path, tokenizer_path = tmp_path / "embedding", tmp_path / "tokenizer"
    corpus = application.index_documents(documents, model_path, tokenizer_path)
    embedder_factory.assert_called_once_with(model_path)
    tokenizer_factory.assert_called_once_with("qwen2.5:1.5b", tokenizer_path)
    llm_factory.assert_called_once_with(
        model="qwen2.5:1.5b", host="http://127.0.0.1:11434", timeout=120,
        prompt_tokenizer=qwen_tokenizer,
    )
    return SimpleNamespace(
        corpus=corpus, documents=documents, embedder=embedder, llm=llm,
        embedder_factory=embedder_factory,
    )


def test_application_reuses_embedding_space_and_preserves_ranked_prompt_evidence(indexed_corpus):
    state = indexed_corpus
    assert (state.corpus.file_count, state.corpus.chunk_count) == (2, 2)
    documents = {document.source: document.text for document in state.documents}

    for question, expected_first in [(LIMIT_QUESTION, "limits.md"), (RUNTIME_QUESTION, "runtime.txt")]:
        result = state.corpus.service.query(question, reserved_output_tokens=64)

        assert result.status == "generated"
        assert result.answer == "Deterministic test response."
        assert [hit.score for hit in result.retrieved] == pytest.approx([1.0, 0.6])
        assert result.sources == result.prompt.selected == result.retrieved
        assert result.source_files[0].name == expected_first
        for hit in result.sources:
            assert hit.chunk.chunk_id == 0
            assert hit.chunk.text == documents[hit.chunk.source]
        assert state.llm.calls[-1] == (result.prompt.text, 64)
        assert question in result.prompt.text
        assert result.prompt.usage.upper_bound + 64 <= 6000
        prompt_metadata = [
            json.loads(line.split(" ", 1)[1])
            for line in result.prompt.text.splitlines() if line.startswith("[S")
        ]
        assert prompt_metadata == [
            {"source": hit.chunk.source.name, "path": str(hit.chunk.source),
             "chunk_id": hit.chunk.chunk_id, "score": hit.score}
            for hit in result.sources
        ]

    # Ingestion embeds once; questions use the same instance's query method.
    assert state.embedder.document_batches == [[document.text for document in state.documents]]
    assert state.embedder.questions == [LIMIT_QUESTION, RUNTIME_QUESTION]
    state.embedder_factory.assert_called_once()
    assert len(state.llm.calls) == 2


def test_application_abstains_on_unrelated_question_without_generation(indexed_corpus):
    state = indexed_corpus
    result = state.corpus.service.query(UNRELATED_QUESTION)

    assert result.status == "no_retrieval_matches"
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.retrieved == result.sources == result.source_files == ()
    assert result.prompt.selected == ()
    assert "No source context is available." in result.prompt.text
    assert state.embedder.questions == [UNRELATED_QUESTION]
    assert state.llm.calls == []
