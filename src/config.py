"""
config.py
----------
Loads environment variables and sets global configuration
for the Intelligent Tutor RAG chatbot.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------
# 1️⃣ Load environment variables from .env
# -----------------------------------------------------
load_dotenv()

# Base directory (project root)
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

# -----------------------------------------------------
# 2️⃣ Core environment variables
# -----------------------------------------------------
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db"))
PERSIST_CHROMA: bool = os.getenv("PERSIST_CHROMA", "true").lower() == "true"
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# -----------------------------------------------------
# 3️⃣ Subject-specific PDF paths
# -----------------------------------------------------
DATA_DIR: Path = BASE_DIR / "intelligent_tutor_bot" / "data"

SUBJECT_PATHS: dict[str, Path] = {
    "english": DATA_DIR / "english.pdf",
    "physics": DATA_DIR / "physics_notes.pdf",
    "biology": DATA_DIR / "bio.pdf",
    "pakistan_studies": DATA_DIR / "pakstudies_dates.pdf",
}

# -----------------------------------------------------
# 4️⃣ Vector DB directories per subject
# -----------------------------------------------------
VECTOR_DB_DIRS: dict[str, Path] = {
    subject: Path(CHROMA_DB_DIR) / subject for subject in SUBJECT_PATHS
}

# -----------------------------------------------------
# 5️⃣ LLM configuration
# -----------------------------------------------------
LLM_MODEL: str = "gpt-4o-mini"  # upgrade to gpt-4o for higher accuracy
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 1000

# -----------------------------------------------------
# 6️⃣ Utility function for debugging
# -----------------------------------------------------
def check_config() -> None:
    """Prints a configuration summary."""
    print("✅ CONFIGURATION SUMMARY")
    print(f"Base directory: {BASE_DIR}")
    print(f"ChromaDB directory: {CHROMA_DB_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Persistence enabled: {PERSIST_CHROMA}")
    print("\nSubjects and PDF paths:")
    for name, path in SUBJECT_PATHS.items():
        print(f"  • {name}: exists={path.exists()} ({path.name})")
    print("\nLLM Model:", LLM_MODEL)
    print("API Key loaded:", bool(OPENAI_API_KEY))


# -----------------------------------------------------
# 7️⃣ Run test if executed directly
# -----------------------------------------------------
if __name__ == "__main__":
    check_config()
