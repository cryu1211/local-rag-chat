import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from local_rag_chat import retrieval_cli


def test_cli_ingestion_to_real_faiss_and_source_output(tmp_path, monkeypatch, capsys):
    first = tmp_path / "a.md"
    second = tmp_path / "b.txt"
    first.write_text("Cats purr.", encoding="utf-8")
    second.write_text("Saturn has rings.", encoding="utf-8")

    class FakeEmbedder:
        def __init__(self, model_path):
            pass

        def embed_documents(self, texts):
            return np.array([
                [1, 0] if "Saturn" in text else [0, 1] for text in texts
            ], dtype=np.float32)

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    monkeypatch.setattr(retrieval_cli, "SentenceTransformerEmbedder", FakeEmbedder)
    arguments = [str(first), str(second), "--model-path", str(tmp_path),
                 "--query", "What surrounds Saturn?", "--top-k", "1"]

    retrieval_cli.main(arguments)

    output = capsys.readouterr().out
    assert "Indexed 2 chunks from 2 document(s)." in output
    assert "cosine=1.0000 source=b.txt chunk_id=0" in output
    assert f"path={second.resolve()}" in output
    assert "Saturn has rings." in output
    assert "Cats purr." not in output

    retrieval_cli.main([str(first), "--model-path", str(tmp_path), "--query", "Saturn",
                        "--min-score", "0.5"])
    assert "No chunks met" in capsys.readouterr().out


def test_empty_corpus_exits_before_loading_model(tmp_path, capsys):
    source = tmp_path / "empty.txt"
    source.write_text(" \n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        retrieval_cli.main([str(source), "--model-path", str(tmp_path / "missing"), "--query", "q"])
    assert error.value.code == 1
    assert "no non-blank text" in capsys.readouterr().err


def test_cli_with_real_local_model_in_fresh_process(tmp_path):
    """Exercise PyTorch and FAISS together, including native startup settings."""
    model_path = os.environ.get("LOCAL_RAG_MODEL_PATH")
    if not model_path:
        pytest.skip("Set LOCAL_RAG_MODEL_PATH to run the offline model smoke test.")
    model_path = str(Path(model_path).expanduser().resolve())
    first = tmp_path / "a.md"
    second = tmp_path / "b.txt"
    first.write_text("Cats purr.", encoding="utf-8")
    second.write_text("Saturn has rings.", encoding="utf-8")

    environment = os.environ.copy()
    # Verify the CLI's default in a clean process, not pytest's imported runtimes.
    environment.pop("OMP_NUM_THREADS", None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-m", "local_rag_chat.retrieval_cli",
         str(first), str(second), "--model-path", model_path,
         "--query", "What surrounds Saturn?", "--top-k", "2"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Indexed 2 chunks from 2 document(s)." in completed.stdout
    assert "source=a.md chunk_id=0" in completed.stdout
    assert "source=b.txt chunk_id=0" in completed.stdout
    assert "cosine=" in completed.stdout
