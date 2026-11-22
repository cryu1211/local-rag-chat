# Test scope

Run the focused architecture checks from the project root:

```sh
python -m pytest -q -m architecture
```

This marker selects existing contract tests plus the application-level tests in
[test_application.py](test_application.py). It requires no model weights, model
downloads, internet access, or running Ollama. FAISS runs on explicit vectors,
the embedding and generation boundaries use fakes/mocks, and the tokenizer
fixture builds a tiny real byte-BPE vocabulary locally. One HTTP contract test
uses its own loopback server rather than Ollama.

## What the focused checks cover

| Behavior | Evidence |
| --- | --- |
| `.md` and `.txt` loading; unsupported types and invalid UTF-8 | [test_ingestion.py](test_ingestion.py) checks both file formats, text, resolved paths, duplicate handling, and explicit rejection. |
| Chunk metadata | [test_ingestion.py](test_ingestion.py) checks original text slices, source boundaries, and per-document chunk IDs. |
| Embedding/retrieval interfaces | [test_embeddings.py](test_embeddings.py) checks offline loading and rejection rather than silent truncation. [test_application.py](test_application.py) checks one document-embedding batch and reuse of the same embedder for successive query embeddings. |
| Module boundaries | [test_rag.py](test_rag.py) checks that the service imports no UI, concrete runtime adapter, or native model/search libraries. [test_ollama.py](test_ollama.py) checks that standalone generation imports no UI or retrieval libraries. |
| Known relevance and ranking | [test_retrieval.py](test_retrieval.py) checks cosine normalization, descending scores, and metadata alignment. [test_application.py](test_application.py) checks two known questions against the small document fixture through real FAISS. |
| Prompt construction and the 6K cap | [test_prompting.py](test_prompting.py) checks grounding, serialized metadata, output reserve, cumulative overflow, exact boundaries, and long questions. [test_ollama.py](test_ollama.py) checks that the exact serialized prompt and reserve reach the HTTP request without extra templating. |
| Source attribution | [test_rag.py](test_rag.py) checks file deduplication, filename collisions, exclusion of budget-skipped chunks, and independence from generated filenames. [test_application.py](test_application.py) compares returned chunk metadata with serialized prompt metadata. |
| Abstention and invalid input | [test_application.py](test_application.py) checks unrelated-query abstention without generation. [test_rag.py](test_rag.py) checks empty-index and blank-question failures before retrieval/generation. |
| LLM failures | [test_ollama.py](test_ollama.py) checks actionable connection failures; [test_rag.py](test_rag.py) checks that generation errors propagate rather than producing a fabricated answer. |

The [fixture documents](fixtures/rag/) describe the context limit and local
runtime. Their known questions and explicit vectors live next to the application
tests. The vectors intentionally control the scores: these tests verify pipeline
wiring and policy, not whether a real embedding model understands those texts.

## What these checks intentionally do not establish

- Real-model semantic retrieval quality, a universally correct relevance cutoff,
  or factual correctness of generated answers. A fake LLM's output cannot prove
  grounding, and source attribution means supplied context rather than verified
  support for every claim.
- Exact generated wording, pixel-level UI appearance, performance benchmarks,
  or compatibility with every model, tokenizer, operating system, and native
  library build.
- A complete defense against document prompt injection. Prompt structure and
  role-marker escaping have regression tests; attack resistance needs a broader
  adversarial evaluation.

Run `python -m pytest -q` for the full default regression suite, including CLI/UI
behavior when the optional UI dependency is installed. Real-model and Ollama
checks remain opt-in; see the [project validation commands](../README.md#validation).
Those smoke tests establish representative integration, not a quality benchmark.

## Most valuable production additions

1. A held-out document/question dataset with answerable questions, unrelated
   questions, and related topics with missing facts. Measure retrieval quality
   and false acceptance/abstention before calibrating thresholds or reranking.
2. Answer-grounding evaluation, including unsupported claims, conflicting
   sources, misleading documents, and prompt-injection attempts.
3. Pinned-model/runtime integration tests at the context boundary, including
   Unicode, long metadata, output reserve, and actual runtime token counts.
4. Representative latency, memory, large-corpus, and runtime-failure tests on
   supported hardware and verified native dependency combinations.
