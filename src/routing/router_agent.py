# src/routing/router_agent.py
"""
Router Agent: Automatically routes user queries to the correct subject QA agent
based on keyword matching and fuzzy similarity.
"""

from difflib import SequenceMatcher
from typing import Optional

SUBJECT_KEYWORDS = {
    "english": ["noun", "verb", "grammar", "sentence", "tense", "english"],
    "physics": ["force", "motion", "energy", "light", "speed", "physics"],
    "biology": ["cell", "organism", "protein", "photosynthesis", "biology"],
    "pakistan_studies": ["pakistan", "independence", "1947", "quaid", "lahore", "pakistan studies"],
}

def detect_subject(query: str, threshold: float = 0.45) -> Optional[str]:
    """
    Detect the most likely subject for a given query.

    Args:
        query: User's input string.
        threshold: Minimum similarity score to accept a subject.

    Returns:
        The subject key string if detected, else None.
    """
    query_lower = query.lower()
    scores = {}

    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = max(SequenceMatcher(None, query_lower, kw).ratio() for kw in keywords)
        scores[subject] = score

    # Select best subject if above confidence threshold
    best_subject = max(scores, key=scores.get)
    if scores[best_subject] >= threshold:
        return best_subject

    return None
