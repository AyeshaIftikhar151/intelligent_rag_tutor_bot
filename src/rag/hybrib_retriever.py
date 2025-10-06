# src/rag/hybrid_retriever.py
"""
Hybrid document retriever and re-ranker for EduTutor RAG.
Combines embedding similarity and lexical overlap for improved context selection.
"""

from typing import List, Optional
from rapidfuzz import fuzz
from langchain.schema import Document

# ---------------------------
# Lexical scoring
# ---------------------------
def lexical_score(query: str, text: str) -> float:
    """
    Compute a lexical overlap score using rapidfuzz's partial_ratio.
    Returns a score between 0 and 100.
    """
    try:
        return fuzz.partial_ratio(query.lower(), text.lower())
    except Exception:
        return 0.0

# ---------------------------
# Hybrid ranking
# ---------------------------
def hybrid_rank(
    docs: List[Document],
    query: str,
    embedding_scores: Optional[List[float]] = None,
    alpha: float = 0.6,
    top_k: int = 5
) -> List[Document]:
    """
    Re-rank retrieved documents by combining embedding similarity and lexical overlap.
    
    Parameters:
    - docs: List of Document objects retrieved from vector store.
    - query: User query string.
    - embedding_scores: Optional list of floats (0..1) corresponding to each doc.
      If None, defaults to 50 for all.
    - alpha: Weight for embedding similarity (0..1), (1-alpha) is lexical weight.
    - top_k: Number of top documents to return.
    
    Returns:
    - List of top_k Document objects, sorted by combined score.
    """
    # Normalize embedding scores to 0..100
    emb_scores = [s * 100 for s in embedding_scores] if embedding_scores else [50.0] * len(docs)
    ranked = []

    for doc, emb in zip(docs, emb_scores):
        lex = lexical_score(query, doc.page_content[:2000])  # Only first 2000 chars for efficiency
        score = alpha * emb + (1.0 - alpha) * lex
        ranked.append((score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked][:top_k]

# ---------------------------
# Build context string
# ---------------------------
def build_context_string(docs: List[Document]) -> str:
    """
    Concatenate top documents into a single context string for LLM.
    Each document is labeled with its source.
    """
    pieces = []
    for doc in docs:
        src = doc.metadata.get("source") or doc.metadata.get("source_file") or "unknown"
        txt = doc.page_content.strip()
        pieces.append(f"Source: {src}\n{txt}\n---")
    return "\n\n".join(pieces)
