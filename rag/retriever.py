"""
rag/retriever.py
Builds and loads a FAISS vector store from document chunks.
"""

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever


FAISS_INDEX_PATH = "faiss_index"


def get_embeddings() -> FastEmbedEmbeddings:
    """Load the embedding model (384-dim, matches your Pinecone index)."""
    return FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_faiss_index(chunks: List[Document]) -> FAISS:
    """Build a FAISS index from document chunks and save to disk."""
    print("[retriever] Building FAISS index...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[retriever] Index saved to '{FAISS_INDEX_PATH}/'")
    return vectorstore


def load_faiss_index() -> FAISS:
    """Load an existing FAISS index from disk."""
    index_path = Path(FAISS_INDEX_PATH)
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
            "Run ingest.py first."
        )
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[retriever] Loaded FAISS index from '{FAISS_INDEX_PATH}/'")
    return vectorstore


def get_retriever(k: int = 3) -> VectorStoreRetriever:
    """Return a retriever from the saved FAISS index."""
    vectorstore = load_faiss_index()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )