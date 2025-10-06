"""
ingest_manager.py
-----------------
Controls the ingestion process for all subjects.
Embeds PDF text and stores them in Chroma vector DBs.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from src.config import SUBJECT_PATHS, VECTOR_DB_DIRS, EMBEDDING_MODEL, PERSIST_CHROMA
from src.ingest.pdf_loader import load_and_split_pdf

def ingest_subject(subject_name: str, pdf_path: str, db_path: str):
    """Ingests a single subject’s PDF into a Chroma collection."""
    print(f"\n📘 Ingesting data for subject: {subject_name}")
    docs = load_and_split_pdf(str(pdf_path))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path,
    )

    if PERSIST_CHROMA:
        db.persist()

    print(f"✅ Successfully created ChromaDB for '{subject_name}' → {db_path}")


def ingest_all_subjects():
    """Iterates through all subjects and builds their vector DBs."""
    for subject, pdf_path in SUBJECT_PATHS.items():
        db_dir = VECTOR_DB_DIRS[subject]
        os.makedirs(db_dir, exist_ok=True)
        ingest_subject(subject, pdf_path, str(db_dir))

    print("\n🎉 All subjects successfully ingested!")


if __name__ == "__main__":
    ingest_all_subjects()
