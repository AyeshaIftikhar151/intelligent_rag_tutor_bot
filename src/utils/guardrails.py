# src/utils/guardrails.py
import re
from typing import Optional

# Basic small talk responses, with placeholders
SMALL_TALK_RESPONSES = {
    "hello": "Hello, {name}! How can I help you with English, Physics, Biology or Pakistan Studies today?",
    "hi": "Hi {name}! What topic are you studying now?",
    "hey": "Hey {name}! Ready to learn something new?",
    "how are you": "I’m a study assistant — I don’t sleep! I’m ready to help you. 😊",
    "thanks": "You’re welcome, {name}! Happy to help.",
    "thank you": "My pleasure, {name}!",
    "good morning": "Good morning, {name}! Let’s make today productive.",
    "good night": "Good night — study dreams!",
}

def extract_name(query: str) -> Optional[str]:
    """Try to detect when user tells their name: 'My name is Ayesha'"""
    m = re.search(r"\bmy name is (\w{2,30})", query, flags=re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    return None

def is_small_talk(query: str) -> Optional[str]:
    q = query.lower().strip()
    for key, resp in SMALL_TALK_RESPONSES.items():
        if key in q:
            return resp
    return None

def is_out_of_scope(query: str) -> bool:
    q = query.lower()
    banned_words = ["hack", "password", "jailbreak", "bomb", "attack", "illegal", "exploit"]
    return any(b in q for b in banned_words)
