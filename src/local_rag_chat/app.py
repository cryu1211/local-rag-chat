"""Thin Streamlit demo: uploads, session state, and presentation only."""

import os
import sys
from pathlib import Path

# Set before importing the shared factory's native FAISS/PyTorch dependencies.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st

from local_rag_chat.application import index_documents
from local_rag_chat.generation import LLMError
from local_rag_chat.loading import document_from_bytes
from local_rag_chat.rag import RAGResult


def display_answer(result: RAGResult) -> None:
    with st.chat_message("assistant"):
        st.markdown(result.answer)
        if result.source_files:
            st.caption("Source files")
            filenames = [path.name for path in result.source_files]
            for path in result.source_files:
                st.text(str(path) if filenames.count(path.name) > 1 else path.name)
        else:
            st.caption("No context sources supplied.")
        with st.expander("Retrieval and token budget"):
            st.json(result.debug_info())


def main() -> None:
    st.set_page_config(page_title="Local RAG chat", layout="centered")
    st.title("Local RAG chat")
    st.caption("Upload documents, build an index, and ask questions using local models.")
    st.session_state.setdefault("corpus", None)
    st.session_state.setdefault("answers", [])

    with st.sidebar:
        st.header("Documents")
        with st.form("ingestion"):
            uploads = st.file_uploader(
                "Upload .md or .txt files", type=["md", "txt"], accept_multiple_files=True,
            )
            with st.expander("Local model settings"):
                model_path = st.text_input("Embedding model path", "./models/all-MiniLM-L6-v2")
                tokenizer_path = st.text_input("Tokenizer path", "./models/qwen2.5-1.5b-tokenizer")
                model = st.text_input("Ollama model", "qwen2.5:1.5b")
                host = st.text_input("Ollama host", "http://127.0.0.1:11434")
            with st.expander("Chunking settings"):
                chunk_size = st.number_input("Chunk size (characters)", min_value=1, value=600, step=1)
                overlap = st.number_input("Overlap (characters)", min_value=0, value=100, step=1)
                st.caption("For token-dense text, reduce chunk size. Overlap must be smaller than chunk size.")
            ingest = st.form_submit_button("Index documents", type="primary")
        st.caption("Indexing replaces the current documents and clears the chat.")

    if ingest:
        # Clear first so a failed replacement cannot query a previous corpus.
        st.session_state["corpus"] = None
        st.session_state["answers"] = []
        try:
            if not uploads:
                raise ValueError("Upload at least one .md or .txt file before indexing.")
            documents = []
            names = set()
            for upload in uploads:
                source = Path(upload.name).name
                if source in names:
                    raise ValueError(f"Duplicate upload filename: {source}. Rename one file and retry.")
                names.add(source)
                documents.append(document_from_bytes(Path(source), upload.getvalue()))
            with st.spinner("Indexing documents locally…"):
                st.session_state["corpus"] = index_documents(
                    documents, Path(model_path), Path(tokenizer_path), model=model, host=host,
                    chunk_size=chunk_size, overlap=overlap,
                )
        except (OSError, ValueError, LLMError) as error:
            st.error(str(error))

    corpus = st.session_state["corpus"]
    if corpus is not None:
        st.success(f"Indexed {corpus.file_count} file(s) into {corpus.chunk_count} chunk(s).")
    else:
        st.info("Upload and index documents to start asking questions.")

    st.caption("Each question is independent; earlier messages are shown for reference only.")
    for result in st.session_state["answers"]:
        with st.chat_message("user"):
            st.markdown(result.question)
        display_answer(result)

    question = st.chat_input("Ask about your documents", disabled=corpus is None)
    if question is not None:
        if not question.strip():
            st.warning("Enter a non-empty question.")
            return
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Finding an answer…"):
                result = corpus.service.query(question.strip(), reserved_output_tokens=128)
        except (OSError, ValueError, LLMError) as error:
            st.error(str(error))
        else:
            st.session_state["answers"].append(result)
            display_answer(result)


if __name__ == "__main__":
    main()
