import json
import subprocess
import sys
from math import inf, nextafter
from pathlib import Path

import pytest

from local_rag_chat.generation import LLMError
from local_rag_chat.models import Chunk, SearchResult
from local_rag_chat.prompting import INSUFFICIENT_CONTEXT_ANSWER, PromptBudgetError
from local_rag_chat.rag import NoDocumentsError, RAGService


class FakeRetriever:
    def __init__(self, hits=(), chunk_count=10, events=None):
        self.chunk_count = chunk_count
        self.hits = list(hits)
        self.calls = []
        self.events = events if events is not None else []

    def search(self, question, top_k=5, min_score=None):
        self.calls.append((question, top_k, min_score))
        self.events.append("retrieve")
        return self.hits


class FakeLLM:
    def __init__(self, events=None, error=None, answer="The limit is 6,000 tokens."):
        self.calls = []
        self.events = events if events is not None else []
        self.error = error
        self.answer = answer

    def generate(self, prompt, max_tokens):
        self.events.append("generate")
        self.calls.append((prompt, max_tokens))
        if self.error:
            raise self.error
        return self.answer


def hit(text="The limit is 6,000 tokens.", score=0.9, chunk_id=0):
    return SearchResult(Chunk(Path("limits.md"), chunk_id, text), score)


def test_query_coordinates_components_and_preserves_structured_evidence(qwen_tokenizer):
    events = []
    matches = [hit(), hit("Additional evidence.", 0.8, 1)]
    retriever = FakeRetriever(matches, events=events)
    llm = FakeLLM(events=events)

    result = RAGService(retriever, llm, qwen_tokenizer).query(
        "What is the limit?", top_k=7, min_score=0.4, reserved_output_tokens=64,
        system_instructions="Answer concisely.",
    )

    assert events == ["retrieve", "generate"]
    assert retriever.calls == [("What is the limit?", 7, 0.4)]
    assert llm.calls == [(result.prompt.text, 64)]
    assert result.status == "generated"
    assert result.answer == "The limit is 6,000 tokens."
    assert result.retrieved == tuple(matches)
    assert result.sources == tuple(matches)
    assert [source.chunk.chunk_id for source in result.sources] == [0, 1]
    assert [source.score for source in result.sources] == [0.9, 0.8]
    assert "Answer concisely." in result.prompt.text
    assert "Answer the question using only the provided source context" in result.prompt.text
    assert result.prompt.usage.upper_bound + 64 <= 6000
    debug = result.debug_info()
    assert debug["retrieval"]["matches"][0]["selected"] is True
    assert debug["prompt"]["selected"][0]["label"] == "S1"
    json.dumps(debug)


@pytest.mark.architecture
def test_sources_include_only_context_that_fits_not_every_retrieval_match(qwen_tokenizer):
    huge = SearchResult(Chunk(Path("excluded.md"), 1, "x" * 7000), 0.99)
    small = hit(chunk_id=2)
    llm = FakeLLM()
    result = RAGService(FakeRetriever([huge, small]), llm, qwen_tokenizer).query("What is the limit?")

    assert result.retrieved == (huge, small)
    assert result.sources == (small,)
    assert result.source_files == (Path("limits.md"),)
    assert result.prompt.skipped[0].result == huge
    assert "x" * 100 not in llm.calls[0][0]
    assert result.debug_info()["retrieval"]["matches"][0]["selected"] is False


@pytest.mark.architecture
def test_attribution_matches_prompt_metadata_and_deduplicates_files(qwen_tokenizer):
    first = SearchResult(Chunk(Path("a/notes.md"), 7, "First evidence."), 0.95)
    repeated_file = SearchResult(Chunk(Path("a/notes.md"), 8, "More evidence."), 0.8)
    same_basename = SearchResult(Chunk(Path("b/notes.md"), 2, "Other evidence."), 0.7)
    # A filename mentioned by the model must never become application attribution.
    llm = FakeLLM(answer="An answer mentioning invented.md.")
    result = RAGService(
        FakeRetriever([same_basename, repeated_file, first]), llm, qwen_tokenizer,
    ).query("What does the evidence say?")

    assert result.sources == result.prompt.selected == (first, repeated_file, same_basename)
    assert result.source_files == (Path("a/notes.md"), Path("b/notes.md"))
    assert result.answer == llm.answer
    prompt_metadata = [
        json.loads(line.split(" ", 1)[1])
        for line in llm.calls[0][0].splitlines() if line.startswith("[S")
    ]
    expected = [
        {"source": hit.chunk.source.name, "path": str(hit.chunk.source),
         "chunk_id": hit.chunk.chunk_id, "score": hit.score}
        for hit in result.sources
    ]
    assert prompt_metadata == expected
    debug = result.debug_info()
    assert debug["source_files"] == ["a/notes.md", "b/notes.md"]
    assert debug["prompt"]["selected"] == [
        {"label": f"S{i}", **metadata} for i, metadata in enumerate(expected, start=1)
    ]
    assert {entry["text"] for entry in debug["retrieval"]["matches"]} == {
        hit.chunk.text for hit in result.sources
    }


@pytest.mark.architecture
def test_no_indexed_documents_fails_without_retrieval_or_generation(qwen_tokenizer):
    retriever = FakeRetriever(chunk_count=0)
    llm = FakeLLM()
    with pytest.raises(NoDocumentsError, match="No documents are indexed"):
        RAGService(retriever, llm, qwen_tokenizer).query("Question?")
    assert retriever.calls == []
    assert llm.calls == []


def test_no_retrieval_matches_abstains_without_calling_llm(qwen_tokenizer):
    retriever = FakeRetriever()
    llm = FakeLLM()
    result = RAGService(retriever, llm, qwen_tokenizer).query("Question?", min_score=0.8)

    assert result.status == "no_retrieval_matches"
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.retrieved == ()
    assert result.sources == ()
    assert result.source_files == ()
    assert llm.calls == []
    assert retriever.calls == [("Question?", 5, 0.8)]
    assert result.prompt.usage.upper_bound <= result.prompt.input_budget


def test_all_retrieved_context_over_budget_abstains_without_calling_llm(qwen_tokenizer):
    matches = [hit("x" * 7000)]
    llm = FakeLLM()
    result = RAGService(FakeRetriever(matches), llm, qwen_tokenizer).query("Question?")

    assert result.status == "no_context_fits"
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.retrieved == tuple(matches)
    assert result.sources == ()
    assert result.source_files == ()
    assert result.prompt.skipped[0].reason == "input budget exceeded"
    assert llm.calls == []


@pytest.mark.architecture
def test_llm_failure_propagates_instead_of_producing_a_fallback_answer(qwen_tokenizer):
    failure = LLMError("Cannot reach Ollama; start ollama serve.")
    llm = FakeLLM(error=failure)
    with pytest.raises(LLMError, match="Cannot reach Ollama") as caught:
        RAGService(FakeRetriever([hit()]), llm, qwen_tokenizer).query("Question?")
    assert caught.value is failure
    assert len(llm.calls) == 1


@pytest.mark.architecture
@pytest.mark.parametrize("question", ["", " \n\t"])
def test_empty_question_fails_before_retrieval_or_generation(qwen_tokenizer, question):
    retriever = FakeRetriever([hit()])
    llm = FakeLLM()
    with pytest.raises(ValueError, match="Question must not be blank"):
        RAGService(retriever, llm, qwen_tokenizer).query(question)
    assert retriever.calls == []
    assert llm.calls == []


def test_large_question_fails_before_query_embedding_or_generation(qwen_tokenizer):
    retriever = FakeRetriever([hit()])
    llm = FakeLLM()
    with pytest.raises(PromptBudgetError, match="Instructions and question exceed"):
        RAGService(retriever, llm, qwen_tokenizer).query("q" * 7000)
    assert retriever.calls == []
    assert llm.calls == []


@pytest.mark.architecture
def test_service_imports_no_ui_runtime_adapter_or_native_libraries():
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import local_rag_chat.rag; "
         "assert not {'faiss', 'torch', 'sentence_transformers', 'streamlit', "
         "'local_rag_chat.ollama', 'local_rag_chat.application'} & sys.modules.keys()"],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture
def scored_retriever():
    """Known vectors test policy mechanics, not an embedding model's semantics."""
    import numpy as np
    from local_rag_chat.index import FaissIndex
    from local_rag_chat.retrieval import Retriever

    class QueryEmbedder:
        def embed_documents(self, texts):
            raise AssertionError("The fixture already supplies indexed vectors.")

        def embed_query(self, text):
            return np.array({
                "What is the context limit?": [1, 0],
                "Who won the football match?": [0, 1],
                "Borderline question": [1, 1],
            }[text], dtype=np.float32)

    chunks = [
        Chunk(Path("limits.md"), 0, "The context limit is 6,000 tokens."),
        Chunk(Path("setup.txt"), 0, "Install the local runtime before starting the application."),
    ]
    return Retriever(QueryEmbedder(), FaissIndex(chunks, np.array([[1, 0], [-1, 0]])))


def test_relevant_query_keeps_only_chunks_above_default_cutoff(scored_retriever, qwen_tokenizer):
    question = "What is the context limit?"
    raw = scored_retriever.search(question, min_score=None)
    assert [match.score for match in raw] == [1.0, -1.0]
    llm = FakeLLM()

    result = RAGService(scored_retriever, llm, qwen_tokenizer).query(question)

    assert result.min_score == 0.3
    assert result.status == "generated"
    assert len(result.sources) == 1
    assert result.sources[0].chunk.source == Path("limits.md")
    assert result.sources[0].score == 1.0
    assert "6,000 tokens" in llm.calls[0][0]
    assert "Install the local runtime" not in llm.calls[0][0]
    debug = result.debug_info()["retrieval"]
    assert debug["metric"] == "cosine_similarity"
    assert debug["threshold_inclusive"] is True


def test_unrelated_query_abstains_by_default_without_llm(scored_retriever, qwen_tokenizer):
    llm = FakeLLM()
    result = RAGService(scored_retriever, llm, qwen_tokenizer).query("Who won the football match?")

    assert result.status == "no_retrieval_matches"
    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.sources == ()
    assert result.retrieved == ()
    assert llm.calls == []


def test_borderline_score_is_inclusive_and_a_stricter_cutoff_abstains(scored_retriever, qwen_tokenizer):
    question = "Borderline question"
    boundary = scored_retriever.search(question, top_k=1)[0].score
    equal_llm = FakeLLM()
    equal = RAGService(scored_retriever, equal_llm, qwen_tokenizer).query(
        question, min_score=boundary,
    )
    assert equal.status == "generated"
    assert equal.sources[0].score == boundary
    assert len(equal_llm.calls) == 1

    stricter_llm = FakeLLM()
    stricter = RAGService(scored_retriever, stricter_llm, qwen_tokenizer).query(
        question, min_score=nextafter(boundary, inf),
    )
    assert stricter.status == "no_retrieval_matches"
    assert stricter.sources == ()
    assert stricter_llm.calls == []
