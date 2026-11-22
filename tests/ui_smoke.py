"""Opt-in real-model UI assertions, isolated from pytest's native-library state."""

import os
from pathlib import Path

from local_rag_chat import app as app_module  # Apply macOS startup settings first.
from streamlit.testing.v1 import AppTest


def main():
    root = Path(__file__).parents[1]
    app = AppTest.from_file(app_module.__file__, default_timeout=180).run()
    app.file_uploader[0].set_value([
        ("architecture.md", (root / "examples/architecture.md").read_bytes(), "text/markdown"),
        ("context.txt", (root / "examples/context.txt").read_bytes(), "text/plain"),
    ])
    app.text_input[0].set_value(os.environ["LOCAL_RAG_MODEL_PATH"])
    app.text_input[1].set_value(os.environ["LOCAL_QWEN_TOKENIZER_PATH"])
    app.text_input[2].set_value(os.environ.get("OLLAMA_TEST_MODEL", "qwen2.5:1.5b"))
    app.text_input[3].set_value(os.environ.get("OLLAMA_TEST_HOST", "http://127.0.0.1:11434"))
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert app.success[0].value == "Indexed 2 file(s) into 2 chunk(s)."

    app.chat_input[0].set_value("How many tokens is the total context limit?").run()
    assert not app.exception
    assert not app.error
    result = app.session_state["answers"][0]
    assert result.status == "generated"
    assert result.answer.strip()
    assert result.source_files == (Path("context.txt"),)
    assert app.chat_message[1].text[0].value == "context.txt"
    assert result.prompt.usage.upper_bound + 128 <= 6000

    # Reuse the corpus while keeping previous chat out of the next prompt.
    app.chat_input[0].set_value("What is the recipe for chocolate cake?").run()
    assert not app.exception
    assert not app.error
    result = app.session_state["answers"][1]
    assert result.status == "no_retrieval_matches"
    assert result.source_files == ()
    assert "How many tokens is the total context limit?" not in result.prompt.text
    print("UI smoke passed")


if __name__ == "__main__":
    main()
