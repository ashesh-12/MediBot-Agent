"""
rag/loader.py
Loads a PDF file, cleans metadata, and splits into chunks.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_pdf(path: str) -> List[Document]:
    """Load all pages from a PDF file."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path.resolve()}")
    
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"[loader] Loaded {len(documents)} pages from '{pdf_path.name}'")
    return documents


def clean_metadata(documents: List[Document]) -> List[Document]:
    """Keep only source and page number in metadata."""
    cleaned = []
    for doc in documents:
        cleaned.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "page":   doc.metadata.get("page", 0),
                }
            )
        )
    return cleaned


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[loader] Split into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def load_and_split(
    path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Full pipeline: load → clean → split."""
    docs   = load_pdf(path)
    docs   = clean_metadata(docs)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    return chunks