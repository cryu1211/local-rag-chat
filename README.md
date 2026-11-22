# Local RAG chat

A small Python application that answers questions using local Markdown and text
files. It provides a CLI and a thin Streamlit demo, with source filenames and
inspectable retrieval/token-budget decisions.

**Constraints:** embeddings, vector search, and LLM inference run locally; no
OpenAI, Anthropic, or cloud model APIs are used. RAG prompts and reserved output
share a **6,000-token application context limit**. Dependencies and model
artifacts must be provisioned separately before offline use.

## Architecture

```text
Ingestion
.md/.txt files
  → UTF-8 loader → overlapping character chunks → local sentence-transformers
  → FAISS index + aligned chunk metadata (source, chunk ID, original text)

Query
question → local query embedding → FAISS cosine top-k search
  → relevance filtering → context-budget selection → grounded prompt builder
  → local Ollama LLM → answer + application-generated source list
```

Context selection and prompt construction share one module so each candidate is
budgeted as part of the complete serialized prompt. CLI and UI use the same
application setup and query service.

| Responsibility | Code |
| --- | --- |
| Shared component setup and indexing | [application.py](src/local_rag_chat/application.py) |
| File/upload decoding and chunking | [loading.py](src/local_rag_chat/loading.py), [chunking.py](src/local_rag_chat/chunking.py) |
| Embedding interface, FAISS, retrieval | [embeddings.py](src/local_rag_chat/embeddings.py), [index.py](src/local_rag_chat/index.py), [retrieval.py](src/local_rag_chat/retrieval.py) |
| Context selection, prompt serialization, counting | [prompting.py](src/local_rag_chat/prompting.py), [tokenization.py](src/local_rag_chat/tokenization.py) |
| Generation interface and local HTTP adapter | [generation.py](src/local_rag_chat/generation.py), [ollama.py](src/local_rag_chat/ollama.py) |
| Query orchestration and structured results | [rag.py](src/local_rag_chat/rag.py) |
| CLI and Streamlit view | [rag_cli.py](src/local_rag_chat/rag_cli.py), [app.py](src/local_rag_chat/app.py) |


## Setup

Commands below run from the repository root on macOS/Linux.

### 1. Python environment and dependencies

Use Python **3.11+**; the development environment used Python 3.13.7.
[.python-version](.python-version) names the author's pyenv environment,
`rag-chat`. If using pyenv, select your own installed interpreter with
`pyenv shell <installed-version>` before creating the environment.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ui]'
```

Core dependencies are `sentence-transformers`, `faiss-cpu`, `numpy`, and
`tokenizers`. Extras add `pytest` (`dev`) and `streamlit` (`ui`).
Use `.[dev]` if only the CLI and tests are needed. PyTorch is installed through
sentence-transformers; Ollama is a separate process. No orchestration framework
or Ollama Python SDK is required.

### 2. Provision the embedding model and generation tokenizer

Run this once with network access, or transfer the same directories from
another machine. The first download is a complete sentence-transformers model;
the second contains only the generation tokenizer's JSON assets.

```sh
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/all-MiniLM-L6-v2",
)
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir="models/qwen2.5-1.5b-tokenizer",
    allow_patterns=["tokenizer.json", "tokenizer_config.json"],
)
PY
```

These [artifact downloads](https://huggingface.co/docs/huggingface_hub/guides/download)
send no documents or questions and use no hosted inference API. Application
execution loads local files only and never automatically downloads missing
models. Model directories are excluded from Git.

### 3. Install Ollama and provision the local LLM

Install [Ollama](https://docs.ollama.com/quickstart); on macOS with Homebrew,
`brew install ollama` is an alternative. Start a foreground server in its own
terminal:

```sh
OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

If an Ollama desktop app or service already owns port 11434, stop it first or
configure and restart that server with cloud features disabled. Setting a
variable in a client terminal does not change an already-running server.
See [Ollama's local-only configuration](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features).

In another terminal, provision the selected local model:

```sh
ollama pull qwen2.5:1.5b
ollama list
```

The model pull requires network access. Subsequent inference uses the installed
weights and needs no API key. Ollama manages LLM weights separately from the
embedding model and tokenizer directories above.

**macOS note:** the CLI and UI default `OMP_NUM_THREADS=1` before importing native
libraries to work around an observed PyTorch/FAISS crash. This reduces CPU
parallelism; explicitly supplied values are preserved. See
[compatibility evidence](DESIGN.md#macos-native-library-compatibility).

## Run

Keep Ollama running and activate the project Python environment.

### CLI

```sh
python -m local_rag_chat examples/architecture.md examples/context.txt \
  --model-path ./models/all-MiniLM-L6-v2 \
  --tokenizer-path ./models/qwen2.5-1.5b-tokenizer \
  --model qwen2.5:1.5b \
  --query "How many tokens is the total context limit?" \
  --top-k 3 --min-score 0.3 --max-tokens 128 --debug
```

The command ingests the specified files and builds an in-memory index on each
invocation. Answer wording varies; normal output includes:

```text
The total context limit is 6,000 tokens.

Source files:
- context.txt
```

`local-rag-query` is the equivalent installed command. Defaults are top-k
`5`, minimum cosine score `0.3`, and output reserve `128` tokens.
Use `python -m local_rag_chat --help` for all options, including chunk size,
overlap, host, timeout, and additional system instructions.

`--debug` writes JSON after `RAG debug:` on stderr: retained scores and chunk
text/IDs, selected/skipped context, source files, tokenizer counts, input budget,
and reserved output. Loading progress may precede it; append
`2> rag-debug.log` to capture the log.

To inspect scores below the RAG cutoff, use retrieval-only search, which has no
default threshold:

```sh
local-rag examples/architecture.md examples/context.txt \
  --model-path ./models/all-MiniLM-L6-v2 \
  --query "How many tokens is the total context limit?" --top-k 3
```

For a standalone inference check without retrieval:

```sh
local-llm --model qwen2.5:1.5b --max-tokens 128 \
  --prompt "Explain what RAG is in one sentence."
```

### Streamlit UI

```sh
python -m streamlit run src/local_rag_chat/app.py
```

Open [127.0.0.1:8501](http://127.0.0.1:8501), upload UTF-8 `.md`/`.txt` files,
and click **Index documents**. A success message shows file/chunk counts. Ask a
question in the chat input; answers show source filenames. Expand
**Retrieval and token budget** for diagnostics. Local model paths, model name,
and Ollama host are configurable in the sidebar.

The project [Streamlit configuration](.streamlit/config.toml) binds to loopback
and disables usage telemetry. Run from the repository root so it is loaded.
Unsupported types, invalid UTF-8, duplicate upload filenames, and entirely blank
uploads produce errors. A new indexing attempt clears the previous corpus and
chat, even if replacement fails. Each question is independent; displayed history
is not sent to the LLM. Stop Streamlit with **Ctrl+C** in its terminal.

## Validation

Model-free checks require no downloads or running Ollama:

```sh
python -m pytest -q -m architecture
python -m pytest -q
```

The focused subset covers loading, metadata, embedding reuse, cosine ranking,
prompt/budget contracts, source attribution, abstention, input guards, and
failure propagation. Tests use deterministic vectors, fake model clients, a
tiny locally built tokenizer, and a loopback HTTP stub. UI tests run when the
`ui` extra is installed. See the [test coverage guide](tests/README.md) for limits
and production evaluation priorities.

To additionally run the opt-in local-model and Ollama smoke tests after setup:

```sh
LOCAL_RAG_MODEL_PATH=./models/all-MiniLM-L6-v2 \
LOCAL_QWEN_TOKENIZER_PATH=./models/qwen2.5-1.5b-tokenizer \
RUN_OLLAMA_SMOKE=1 RUN_RAG_SMOKE=1 python -m pytest -q
```

Actual upload-widget smoke testing requires Streamlit 1.67+; the app uses APIs
available in 1.40+. Real native-model tests use fresh processes. Smoke tests check
integration, not answer quality or retrieval accuracy.

## Key design decisions

| Decision | Rationale and current behavior |
| --- | --- |
| FAISS | Exact `IndexFlatIP` search is sufficient for a small corpus and needs no database service or index training. Metadata remains aligned with vector rows. |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) is a compact English-oriented, 384-dimensional baseline, run on CPU. Document/query methods share one model instance but remain separate substitution points. |
| Ollama | Convenient local model provisioning and HTTP inference. The [Qwen2.5 1.5B model](https://ollama.com/library/qwen2.5:1.5b) keeps demo resource needs modest; quality is not guaranteed. The adapter implements a small `LLMClient` protocol. |
| Chunking | Fixed windows of 600 characters with 100-character overlap are simple to inspect. They are starting values, not tuned optima. Embedding inputs exceeding the model's token limit are rejected rather than truncated. |
| Similarity and top-k | Normalize document and query vectors, then use inner product as cosine similarity. Retrieve at most `min(top_k, chunk_count)` candidates, highest first. Filtering and budgeting can reduce that count; top-k is not a token budget. |
| 6K budget | Reserve output first, then enforce a conservative bound on the entire serialized input, including instructions, question, metadata, context, and role markers. Skip oversized chunks and continue in relevance order. |
| Relevance | Keep scores `>= min_score` (RAG default 0.3). If none pass, skip generation and return an explicit not-found answer. Similarity is a heuristic, not answer confidence. |
| Sources | Derive attribution from chunks actually supplied to the prompt, not generated citations. Deduplicate files by path and retain chunk IDs, scores, and text for debug inspection. |
| Conversation history | The current implementation treats each question independently. Previous turns are displayed in the UI but are not included in subsequent prompts. This preserves the limited context budget for retrieved document context. A production multi-turn version could allocate a bounded portion of the 6K context window to recent conversation history or summarized older turns. |

## Trade-offs and limitations

- Small, in-memory corpus; ingestion materializes documents and embeddings.
  There is no persisted index, incremental update, or durable chunk identity.
- Dense vector retrieval only: no lexical retrieval, reranking, or calibrated
  answerability model. The threshold is model-, corpus-, and query-dependent.
- Only UTF-8 Markdown and plain text; Markdown structure is not parsed. Character
  windows can split sentences or code blocks.
- The token bound is conservative and model-specific. Only the registered Qwen
  profile supports bounded RAG generation; swapping a model name alone is unsafe.
- Grounding instructions and source lists do not prove factual support or prevent
  all hallucinations or document prompt injection.
- Local single-user demo: no authentication, supported multi-user isolation,
  streaming, or conversational memory. UI state is lost on reload/restart;
  indexing again replaces it. Dependencies/model revisions are not locked.

## Future improvements

Prioritize held-out retrieval/abstention and answer-grounding evaluation, then
consider a local reranker or hybrid lexical/vector retrieval. Add persistence
only when needed: a saved FAISS/metadata pair, or a self-hosted store such as
pgvector or Qdrant, would require versioned model/chunk metadata and update rules.
Alternative local runtimes such as llama.cpp or vLLM can implement the generation
interface with separately validated prompt/tokenizer behavior.

### Optional multi-turn conversation

Pass an optional history of user/assistant messages through the RAG service into
the prompt builder, preserving message roles. The UI/session layer would own
history; the service would continue to receive it explicitly rather than storing
hidden conversation state.

History must share the same 6,000-token window with instructions, the current
question, retrieved context, framing, and reserved output. A simple initial
policy would drop the oldest complete conversation turns when needed to retain
useful document evidence, then validate the complete serialized prompt again.

For context-dependent follow-ups such as "What about its limitations?", retrieval
also needs conversation context. A local query-rewriting step could use recent
history to produce a standalone search question, with its own bounded prompt.
Tests should cover history eviction, preserved roles, follow-up retrieval, and
source attribution after history and context selection.
