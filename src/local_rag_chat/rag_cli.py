"""Ingest local files and run one complete, budgeted RAG query."""

import argparse
import json
import os
import sys
from pathlib import Path

# Apply the same native-library compatibility default as the retrieval CLI.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

from .chunking import chunk_documents
from .embeddings import SentenceTransformerEmbedder
from .generation import CONTEXT_WINDOW, LLMError
from .index import FaissIndex
from .loading import load_documents
from .ollama import OllamaClient
from .rag import NoDocumentsError, RAGService
from .retrieval import Retriever
from .tokenization import load_prompt_tokenizer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Answer a question using local .md and .txt files.")
    parser.add_argument("files", type=Path, nargs="+", help="UTF-8 documents to ingest")
    parser.add_argument("--model-path", type=Path, required=True, help="Local embedding model directory")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Local generation tokenizer directory")
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Installed local Ollama model")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=128, help="Reserved maximum output tokens")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, help="Optional calibrated cosine cutoff in [-1, 1]")
    parser.add_argument("--chunk-size", type=int, default=600, help="Characters per chunk")
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--system-instructions", help="Additional instructions; grounding rules remain required")
    parser.add_argument("--debug", action="store_true", help="Print retrieval and token-budget JSON to stderr")
    args = parser.parse_args(argv)

    try:
        if not args.query.strip():
            raise ValueError("Question must not be blank.")
        if args.top_k <= 0:
            raise ValueError("Top-k must be positive.")
        if args.min_score is not None and not -1 <= args.min_score <= 1:
            raise ValueError("Minimum score must be between -1 and 1.")
        if not 0 < args.max_tokens < CONTEXT_WINDOW:
            raise ValueError("Maximum output tokens must be between 1 and 5999.")

        # Ingestion remains separate from query orchestration.
        documents = load_documents(args.files)
        chunks = chunk_documents(documents, args.chunk_size, args.overlap)
        if not chunks:
            raise NoDocumentsError("No documents are indexed: the supplied files contain no non-blank text.")
        tokenizer = load_prompt_tokenizer(args.model, args.tokenizer_path)
        llm = OllamaClient(
            model=args.model, host=args.host, timeout=args.timeout, prompt_tokenizer=tokenizer,
        )
        embedder = SentenceTransformerEmbedder(args.model_path)
        index = FaissIndex(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
        service = RAGService(Retriever(embedder, index), llm, tokenizer)
        result = service.query(
            args.query, top_k=args.top_k, min_score=args.min_score,
            reserved_output_tokens=args.max_tokens, system_instructions=args.system_instructions,
        )
    except (OSError, ValueError, LLMError) as error:
        parser.exit(1, f"error: {error}\n")

    print(result.answer)
    if result.sources:
        print("\nContext sources:")
        for i, hit in enumerate(result.sources, start=1):
            print(f"[S{i}] {hit.chunk.source} chunk_id={hit.chunk.chunk_id} cosine={hit.score:.4f}")
    else:
        print(f"\nNo context sources supplied ({result.status}).")
    if args.debug:
        print("\nRAG debug:", file=sys.stderr)
        print(json.dumps(result.debug_info(), indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
