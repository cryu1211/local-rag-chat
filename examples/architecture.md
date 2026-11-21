# Local retrieval

Documents are loaded from Markdown and plain-text files on disk.
Sentence-transformers runs an embedding model locally on the CPU.
FAISS searches the document embeddings using cosine similarity.
Each retrieved chunk retains its source path and chunk ID.
