import json
import subprocess
import sys
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
    def __init__(self, events=None, error=None):
        self.calls = []
        self.events = events if events is not None else []
        self.error = error

    def generate(self, prompt, max_tokens):
        self.events.append("generate")
        self.calls.append((prompt, max_tokens))
        if self.error:
            raise self.error
        return "The limit is 6,000 tokens [S1]."


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
    assert result.answer == "The limit is 6,000 tokens [S1]."
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


def test_sources_include_only_context_that_fits_not_every_retrieval_match(qwen_tokenizer):
    huge = hit("x" * 7000, 0.99, 1)
    small = hit(chunk_id=2)
    llm = FakeLLM()
    result = RAGService(FakeRetriever([huge, small]), llm, qwen_tokenizer).query("What is the limit?")

    assert result.retrieved == (huge, small)
    assert result.sources == (small,)
    assert result.prompt.skipped[0].result == huge
    assert "x" * 100 not in llm.calls[0][0]
    assert result.debug_info()["retrieval"]["matches"][0]["selected"] is False


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
    assert result.prompt.skipped[0].reason == "input budget exceeded"
    assert llm.calls == []


def test_llm_failure_propagates_instead_of_producing_a_fallback_answer(qwen_tokenizer):
    failure = LLMError("Cannot reach Ollama; start ollama serve.")
    llm = FakeLLM(error=failure)
    with pytest.raises(LLMError, match="Cannot reach Ollama") as caught:
        RAGService(FakeRetriever([hit()]), llm, qwen_tokenizer).query("Question?")
    assert caught.value is failure
    assert len(llm.calls) == 1


def test_large_question_fails_before_query_embedding_or_generation(qwen_tokenizer):
    retriever = FakeRetriever([hit()])
    llm = FakeLLM()
    with pytest.raises(PromptBudgetError, match="Instructions and question exceed"):
        RAGService(retriever, llm, qwen_tokenizer).query("q" * 7000)
    assert retriever.calls == []
    assert llm.calls == []


def test_service_imports_no_native_retrieval_or_model_libraries():
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import local_rag_chat.rag; "
         "assert not {'faiss', 'torch', 'sentence_transformers'} & sys.modules.keys()"],
        capture_output=True, text=True, timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
