import json
import os
import subprocess
import sys

import numpy as np
import pytest

from local_rag_chat import rag_cli
from local_rag_chat.generation import LLMError


@pytest.fixture
def cli_components(monkeypatch):
    class FakeEmbedder:
        def __init__(self, model_path):
            pass

        def embed_documents(self, texts):
            return np.array([[1, 0] if "Saturn" in text else [0, 1] for text in texts],
                            dtype=np.float32)

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    class FakeLLM:
        instances = []

        def __init__(self, model, host, timeout, prompt_tokenizer):
            self.prompt_tokenizer = prompt_tokenizer
            self.calls = []
            self.error = None
            self.instances.append(self)

        def generate(self, prompt, max_tokens):
            self.calls.append((prompt, max_tokens))
            if self.error:
                raise self.error
            return "Saturn has rings [S1]."

    monkeypatch.setattr(rag_cli, "SentenceTransformerEmbedder", FakeEmbedder)
    monkeypatch.setattr(rag_cli, "OllamaClient", FakeLLM)
    return FakeLLM


def arguments(tmp_path, min_score=0.5):
    args = ["--model-path", str(tmp_path / "embedding"), "--tokenizer-path", str(tmp_path / "tokenizer"),
            "--query", "What surrounds Saturn?", "--top-k", "2", "--debug"]
    if min_score is not None:
        args.extend(["--min-score", str(min_score)])
    return args


def debug_json(stderr):
    return json.loads(stderr.split("RAG debug:\n", 1)[1])


def test_cli_ingests_files_searches_real_faiss_and_prints_answer_sources_and_debug(
    tmp_path, qwen_tokenizer, cli_components, capsys,
):
    first, second = tmp_path / "a.md", tmp_path / "b.txt"
    first.write_text("Cats purr.", encoding="utf-8")
    second.write_text("Saturn has rings.", encoding="utf-8")

    rag_cli.main([str(first), str(second), *arguments(tmp_path), "--max-tokens", "32"])

    output = capsys.readouterr()
    assert "Saturn has rings [S1]." in output.out
    assert f"[S1] {second.resolve()} chunk_id=0 cosine=1.0000" in output.out
    assert "a.md" not in output.out
    debug = debug_json(output.err)
    assert debug["status"] == "generated"
    assert debug["retrieval"]["indexed_chunks"] == 2
    assert debug["retrieval"]["min_score"] == 0.5
    assert debug["retrieval"]["matches"][0]["source"] == "b.txt"
    assert debug["prompt"]["input_token_upper_bound"] + 32 <= 6000
    client = cli_components.instances[0]
    assert client.prompt_tokenizer.model == "qwen2.5:1.5b"
    assert client.calls[0][1] == 32
    assert "Saturn has rings." in client.calls[0][0]
    assert "Cats purr." not in client.calls[0][0]


def test_cli_no_useful_matches_does_not_generate(tmp_path, qwen_tokenizer, cli_components, capsys):
    source = tmp_path / "a.md"
    source.write_text("Cats purr.", encoding="utf-8")
    rag_cli.main([str(source), *arguments(tmp_path)])
    output = capsys.readouterr()
    assert "information is not available" in output.out
    assert debug_json(output.err)["status"] == "no_retrieval_matches"
    assert cli_components.instances[0].calls == []


def test_cli_default_cutoff_rejects_unrelated_context(tmp_path, qwen_tokenizer, cli_components, capsys):
    source = tmp_path / "a.md"
    source.write_text("Cats purr.", encoding="utf-8")
    rag_cli.main([str(source), *arguments(tmp_path, min_score=None)])

    output = capsys.readouterr()
    assert "information is not available" in output.out
    debug = debug_json(output.err)
    assert debug["retrieval"]["min_score"] == 0.3
    assert debug["status"] == "no_retrieval_matches"
    assert cli_components.instances[0].calls == []


def test_cli_blank_corpus_reports_no_documents_before_loading_models(tmp_path, capsys):
    source = tmp_path / "empty.txt"
    source.write_text(" \n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        rag_cli.main([str(source), *arguments(tmp_path)])
    assert caught.value.code == 1
    assert "No documents are indexed" in capsys.readouterr().err


def test_cli_llm_unavailable_reports_error_without_an_answer(
    tmp_path, qwen_tokenizer, cli_components, monkeypatch, capsys,
):
    source = tmp_path / "a.txt"
    source.write_text("Saturn has rings.", encoding="utf-8")

    def unavailable(self, prompt, max_tokens):
        raise LLMError("Cannot reach Ollama; start ollama serve.")

    monkeypatch.setattr(cli_components, "generate", unavailable)
    with pytest.raises(SystemExit) as caught:
        rag_cli.main([str(source), *arguments(tmp_path)])
    assert caught.value.code == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "error: Cannot reach Ollama" in output.err
    assert "Traceback" not in output.err


def test_real_end_to_end_rag_query():
    if os.environ.get("RUN_RAG_SMOKE") != "1":
        pytest.skip("Set RUN_RAG_SMOKE=1 to run the fully local RAG subprocess test.")
    embedding = os.environ.get("LOCAL_RAG_MODEL_PATH")
    tokenizer = os.environ.get("LOCAL_QWEN_TOKENIZER_PATH")
    if not embedding or not tokenizer:
        pytest.skip("Set LOCAL_RAG_MODEL_PATH and LOCAL_QWEN_TOKENIZER_PATH to provisioned directories.")
    environment = os.environ.copy()
    environment.pop("OMP_NUM_THREADS", None)  # Exercise this CLI's own startup default.
    command = [sys.executable, "-m", "local_rag_chat", "examples/architecture.md", "examples/context.txt",
         "--model-path", embedding, "--tokenizer-path", tokenizer,
         "--model", os.environ.get("OLLAMA_TEST_MODEL", "qwen2.5:1.5b"),
         "--host", os.environ.get("OLLAMA_TEST_HOST", "http://127.0.0.1:11434"),
         "--query", "How many tokens is the total context limit?",
         "--top-k", "3", "--max-tokens", "128", "--debug"]
    completed = subprocess.run(
        command,
        env=environment, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Context sources:" in completed.stdout
    debug = debug_json(completed.stderr)
    assert debug["status"] == "generated"
    assert debug["retrieval"]["min_score"] == 0.3
    assert debug["prompt"]["selected"]
    assert debug["prompt"]["input_token_upper_bound"] + 128 <= 6000

    # An unrelated question must abstain even when inference is unreachable.
    unrelated = command.copy()
    unrelated[unrelated.index("--query") + 1] = "What is the recipe for chocolate cake?"
    unrelated[unrelated.index("--host") + 1] = "http://127.0.0.1:1"
    completed = subprocess.run(
        unrelated, env=environment, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "information is not available in the provided documents" in completed.stdout
    debug = debug_json(completed.stderr)
    assert debug["status"] == "no_retrieval_matches"
    assert debug["retrieval"]["matches"] == []
    assert debug["prompt"]["selected"] == []
