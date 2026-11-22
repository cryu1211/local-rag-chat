import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from local_rag_chat.embeddings import SentenceTransformerEmbedder


@pytest.mark.architecture
def test_adapter_loads_only_local_files_and_refuses_truncation(tmp_path, monkeypatch):
    calls = []

    class FakeModel:
        max_seq_length = 4

        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

        def tokenizer(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            assert kwargs["add_special_tokens"] is True
            return {"length": [len(text.split()) + 2 for text in texts]}

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 3), dtype=np.float64)

    for flag in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"]:
        monkeypatch.setenv(flag, "0")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=FakeModel
    ))
    embedder = SentenceTransformerEmbedder(tmp_path)

    vectors = embedder.embed_documents(["two words"])

    assert calls[0] == (str(tmp_path.resolve()), {
        "device": "cpu", "local_files_only": True, "trust_remote_code": False
    })
    assert vectors.shape == (1, 3)
    assert vectors.dtype == np.float32
    assert calls[1][1]["prompt"] == ""
    with pytest.raises(ValueError, match="No input was truncated"):
        embedder.embed_documents(["three words here"])
    assert len(calls) == 2  # Encoding never runs after a failed length check.
    for texts in [[], [" \n"]]:
        with pytest.raises(ValueError, match="non-blank"):
            embedder.embed_documents(texts)
    assert embedder.embed_query("question").shape == (3,)
    with pytest.raises(ValueError, match="No input was truncated"):
        embedder.embed_query("three words here")


def test_missing_model_directory_fails_without_loading_a_model(tmp_path):
    with pytest.raises(ValueError, match="directory does not exist"):
        SentenceTransformerEmbedder(tmp_path / "missing")
