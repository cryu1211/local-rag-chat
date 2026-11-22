"""Exercise the Streamlit view with real indexing/prompting and fake local models."""

from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

st = pytest.importorskip("streamlit", reason="Install the ui extra to test the demo.")
from streamlit.testing.v1 import AppTest

# Import the UI entry point first, preserving its macOS native-library startup order.
from local_rag_chat import app as app_module
from local_rag_chat import application
from local_rag_chat.generation import LLMError


def upload(name, content):
    file = BytesIO(content)
    file.name = name
    return file


@pytest.fixture
def ui(monkeypatch, qwen_tokenizer):
    uploads = []
    # Mock the browser upload boundary; decoding, indexing, and RAG stay real.
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: uploads)
    monkeypatch.setattr(application, "load_prompt_tokenizer", lambda *args: qwen_tokenizer)

    class Embedder:
        instances = 0

        def __init__(self, model_path):
            Embedder.instances += 1

        def embed_documents(self, texts):
            return np.array([[1, 0] if "Saturn" in text else [0, 1] for text in texts], dtype=np.float32)

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    class LLM:
        calls = []

        def __init__(self, **kwargs):
            pass

        def generate(self, prompt, max_tokens):
            LLM.calls.append((prompt, max_tokens))
            return "Saturn has rings."

    monkeypatch.setattr(application, "SentenceTransformerEmbedder", Embedder)
    monkeypatch.setattr(application, "OllamaClient", LLM)
    app = AppTest.from_file(app_module.__file__).run()
    assert not app.exception
    return app, uploads, Embedder, LLM


def test_chat_disabled_until_documents_are_indexed(ui):
    app, _, embedder, llm = ui
    assert app.chat_input[0].disabled
    assert "Upload and index" in app.info[0].value
    assert embedder.instances == 0
    assert llm.calls == []


def test_ingest_ask_and_rerun_preserve_corpus_answer_sources_and_debug(ui):
    app, uploads, embedder, llm = ui
    uploads.extend([upload("notes.md", b"Saturn has rings."), upload("empty.txt", b" \n")])
    app.button[0].click().run()
    assert not app.exception
    assert app.success[0].value == "Indexed 1 file(s) into 1 chunk(s)."
    assert not app.chat_input[0].disabled

    app.chat_input[0].set_value("What surrounds Saturn?").run()
    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    assert app.chat_message[1].markdown[0].value == "Saturn has rings."
    assert app.chat_message[1].text[0].value == "notes.md"
    assert app.chat_message[1].expander[0].label == "Retrieval and token budget"
    result = app.session_state["answers"][0]
    assert result.sources == result.prompt.selected
    assert result.source_files == (Path("notes.md"),)
    assert result.debug_info()["retrieval"]["matches"][0]["text"] == "Saturn has rings."
    assert llm.calls == [(result.prompt.text, 128)]

    app.run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert embedder.instances == 1
    assert len(llm.calls) == 1


@pytest.mark.parametrize("files,error", [
    ([], "Upload at least one"),
    ([("bad.pdf", b"text")], "Unsupported file type"),
    ([("bad.txt", b"\xff")], "UTF-8"),
    ([("blank.md", b" \n")], "No documents are indexed"),
    ([("notes.md", b"first"), ("notes.md", b"second")], "Duplicate upload filename"),
])
def test_invalid_uploads_report_error_without_loading_models(ui, files, error):
    app, uploads, embedder, llm = ui
    uploads.extend(upload(name, content) for name, content in files)
    app.button[0].click().run()
    assert not app.exception
    assert error in app.error[0].value
    assert app.chat_input[0].disabled
    assert app.session_state["corpus"] is None
    assert embedder.instances == 0
    assert llm.calls == []


def test_failed_reindex_clears_previous_corpus_and_chat(ui):
    app, uploads, _, llm = ui
    uploads.append(upload("notes.md", b"Saturn has rings."))
    app.button[0].click().run()
    app.chat_input[0].set_value("Tell me about Saturn.").run()
    assert len(app.session_state["answers"]) == 1

    uploads[:] = [upload("replacement.txt", b"")]
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["corpus"] is None
    assert app.session_state["answers"] == []
    assert not app.chat_message
    assert app.chat_input[0].disabled
    assert len(llm.calls) == 1


def test_blank_question_does_not_call_generation(ui):
    app, uploads, _, llm = ui
    uploads.append(upload("notes.txt", b"Saturn has rings."))
    app.button[0].click().run()
    app.chat_input[0].set_value("   ").run()
    assert not app.exception
    assert "non-empty question" in app.warning[0].value
    assert app.session_state["answers"] == []
    assert llm.calls == []


def test_unrelated_question_shows_abstention_without_sources(ui):
    app, uploads, _, llm = ui
    uploads.append(upload("notes.txt", b"Saturn has rings."))
    app.button[0].click().run()
    app.chat_input[0].set_value("How do I bake a cake?").run()
    assert not app.exception
    assert "information is not available" in app.chat_message[1].markdown[0].value
    assert app.chat_message[1].caption[0].value == "No context sources supplied."
    assert llm.calls == []


def test_runtime_failure_is_visible_without_a_fabricated_answer(ui, monkeypatch):
    app, uploads, _, llm = ui
    uploads.append(upload("notes.txt", b"Saturn has rings."))
    app.button[0].click().run()

    def unavailable(self, prompt, max_tokens):
        raise LLMError("Cannot reach Ollama; start ollama serve.")

    monkeypatch.setattr(llm, "generate", unavailable)
    app.chat_input[0].set_value("Tell me about Saturn.").run()
    assert not app.exception
    assert "Cannot reach Ollama" in app.error[0].value
    assert app.session_state["answers"] == []


def test_real_streamlit_upload_and_local_rag():
    if os.environ.get("RUN_RAG_SMOKE") != "1":
        pytest.skip("Set RUN_RAG_SMOKE=1 to exercise the UI with provisioned local models.")
    embedding = os.environ.get("LOCAL_RAG_MODEL_PATH")
    tokenizer = os.environ.get("LOCAL_QWEN_TOKENIZER_PATH")
    if not embedding or not tokenizer:
        pytest.skip("Set LOCAL_RAG_MODEL_PATH and LOCAL_QWEN_TOKENIZER_PATH.")
    pytest.importorskip("streamlit", minversion="1.67")
    # Like the CLI smoke test, exercise native initialization in a fresh process,
    # without FAISS/OpenMP state left by deterministic-vector unit tests.
    environment = os.environ.copy()
    environment.pop("OMP_NUM_THREADS", None)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ui_smoke.py"))],
        env=environment, capture_output=True, text=True, timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "UI smoke passed" in completed.stdout
