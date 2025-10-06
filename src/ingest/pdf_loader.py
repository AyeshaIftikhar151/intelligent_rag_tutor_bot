"""
pdf_loader.py
-------------
Handles loading and text extraction from PDF files.
Supports both text-based and scanned PDFs.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Loads a PDF file, extracts text, and splits it into chunks.
    Returns a list of Document objects ready for embedding.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load {pdf_path}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", " ", ".", "!", "?", ","],
    )

    docs = splitter.split_documents(documents)
    print(f"✅ Loaded {len(docs)} text chunks from {pdf_path}")
    return docs
