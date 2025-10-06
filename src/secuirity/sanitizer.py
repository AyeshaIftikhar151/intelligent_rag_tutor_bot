# src/security/sanitizer.py
import re
from typing import Tuple, List
from rapidfuzz import fuzz
from pathlib import Path
from ..logger import get_logger

logger = get_logger("sanitizer")

# Prompt-injection patterns (extendable)
INJECTION_PATTERNS = [
    r"ignore (previous|earlier) instructions",
    r"disregard (previous|earlier) instructions",
    r"follow these new instructions instead",
    r"do not refer to the documents",
    r"pretend you are",
    r"bypass",
    r"jailbreak",
    r"exfiltrate",
]

# Disallowed topics
DISALLOWED_PATTERNS = [
    r"password", r"private key", r"secret", r"credit card", r"ssn", r"social security",
    r"exploit", r"attack", r"terror", r"illegal"
]

def sanitize_user_input(text: str) -> Tuple[str, bool, List[str]]:
    """
    Sanitize input text and flag suspicious or disallowed content.
    
    Returns:
        cleaned_text: trimmed and cleaned input
        flagged: True if input contains dangerous or restricted patterns
        reasons: list of patterns matched
    """
    reasons = []
    if not isinstance(text, str):
        return "", True, ["non-string input"]

    lowered = text.strip()
    # remove suspicious system-like prefixes
    cleaned = re.sub(r"^\s*(system:|assistant:|user:)\s*", "", lowered, flags=re.IGNORECASE)

    # detect prompt-injection patterns
    for pat in INJECTION_PATTERNS:
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            reasons.append(f"prompt-injection pattern: {pat}")

    # detect disallowed topics
    for pat in DISALLOWED_PATTERNS:
        if re.search(pat, cleaned, flags=re.IGNORECASE):
            reasons.append(f"disallowed content pattern: {pat}")

    flagged = len(reasons) > 0
    if flagged:
        logger.warning("Sanitizer flagged input: %s; reasons: %s", cleaned, reasons)

    return cleaned, flagged, reasons

def safe_trim(text: str, max_chars: int = 2000) -> str:
    """
    Trim text to a maximum length, preferably ending at sentence boundary.

    Args:
        text: Input text
        max_chars: Maximum allowed characters

    Returns:
        Trimmed text
    """
    if len(text) <= max_chars:
        return text

    # try to cut at last sentence boundary before max_chars
    boundary = re.search(r'(?s).{1,%d}([.!?])' % max_chars, text)
    if boundary:
        return text[:boundary.end()]

    return text[:max_chars]
