"""A small embedding interface and its offline sentence-transformers adapter."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    """Keep the document and query sperate to allow models with distinct instructions for each."""
    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return a finite (number of texts, embedding dimension) matrix.

        Documents and queries must use compatible encodings from the same model.
        """
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Return one finite vector of the same dimension as document vectors."""
        ...


class SentenceTransformerEmbedder:
    """Load a trusted local model directory and perform inference on the CPU.

    This adapter uses symmetric encoding, suitable for all-MiniLM-L6-v2.
    Models requiring distinct query/document instructions need another adapter.
    """

    def __init__(self, model_path: Path) -> None:
        model_path = model_path.expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError(f"Embedding model directory does not exist: {model_path}")

        # Set before importing the libraries that read these flags at import time.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            str(model_path),
            device="cpu",
            local_files_only=True,
            trust_remote_code=False,
        )

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embedds a batch of documents and returns a matrix of shape (len(texts), embedding dimension)."""
        return self._embed(texts)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embedds one questions and returns one vector of the same dimension as document vectors."""
        return self._embed([text])[0]

    def _embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs must contain non-blank text.")
        texts = list(texts)
        lengths = self._model.tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            return_length=True,
            return_attention_mask=False,
        )["length"]
        for position, length in enumerate(lengths):
            if length > self._model.max_seq_length:
                raise ValueError(
                    f"Embedding input {position} has {length} tokens; the model limit "
                    f"is {self._model.max_seq_length}. Reduce the document chunk size "
                    "or shorten the query. No input was truncated."
                )

        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
            prompt="",  # Do not add a model default prompt after the length check.
        )
        return np.asarray(vectors, dtype=np.float32)
