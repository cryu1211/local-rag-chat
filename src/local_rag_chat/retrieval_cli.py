"""Command-line entry point for document ingestion and retrieval-only queries."""

import argparse
import os
import sys
from pathlib import Path

# macOS PyTorch/FAISS wheels can conflict in OpenMP worker threads.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from .chunking import chunk_documents
from .embeddings import SentenceTransformerEmbedder
from .index import FaissIndex
from .loading import load_documents
from .retrieval import Retriever


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search local .md and .txt files.")
    parser.add_argument("files", type=Path, nargs="+", help="UTF-8 documents to ingest")
    parser.add_argument("--model-path", type=Path, required=True, help="Local model directory")
    parser.add_argument("--query", required=True, help="Question to retrieve context for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=600, help="Characters per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Overlapping characters")
    parser.add_argument("--min-score", type=float, help="Optional cosine cutoff in [-1, 1]")
    args = parser.parse_args(argv)

    try:
        # Validate inexpensive inputs before loading the model.
        if not args.query.strip():
            raise ValueError("Question must not be blank.")
        if args.top_k <= 0:
            raise ValueError("Top-k must be positive.")
        if args.min_score is not None and not -1 <= args.min_score <= 1:
            raise ValueError("Minimum score must be between -1 and 1.")
        documents = load_documents(args.files)
        chunks = chunk_documents(documents, args.chunk_size, args.overlap)
        if not chunks:
            raise ValueError("The supplied documents contain no non-blank text.")

        # Load the local embedding model
        embedder = SentenceTransformerEmbedder(args.model_path)
        # Embed chunk text and build a FAISS index
        index = FaissIndex(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
        # Retrieve top-k matches using the embedded query
        results = Retriever(embedder, index).search(args.query, args.top_k, args.min_score)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    print(f"Indexed {len(chunks)} chunks from {len(documents)} document(s).")
    if not results:
        print("No chunks met the minimum similarity score.")
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(
            f"\n[{rank}] cosine={result.score:.4f} "
            f"source={chunk.source.name} chunk_id={chunk.chunk_id}"
        )
        print(f"path={chunk.source}")
        print(chunk.text)


if __name__ == "__main__":
    main()
