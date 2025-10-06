# src/utils/config_loader.py
import os
from dotenv import load_dotenv

def load_config():
    """
    Load configuration from environment variables or defaults.
    
    Environment variables:
    - CHROMA_DB_DIR: directory where Chroma vector stores are saved
    - LLM_MODEL: LLM model name for ChatOpenAI
    - EMBEDDING_MODEL: Embedding model name for HuggingFaceEmbeddings
    """
    load_dotenv()  # Load variables from .env file if present

    return {
        "CHROMA_DB_DIR": os.getenv("CHROMA_DB_DIR", "chroma_db"),
        "LLM_MODEL": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    }
