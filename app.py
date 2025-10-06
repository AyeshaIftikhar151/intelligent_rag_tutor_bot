"""
app.py — Intelligent Multi-Subject RAG Tutor Chatbot
-----------------------------------------------------
Command-line chatbot that uses per-subject RAG retrieval to answer
questions in English, Physics, Biology, and Pakistan Studies.
"""

import os
import sys
from typing import Optional

# ---------------------------
# Add src folder to path
# ---------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from src.utils.memory_manager import MemoryManager
from src.secuirity.sanitizer import sanitize_user_input
from src.utils.guardrails import is_small_talk, is_out_of_scope
from src.rag.hybrib_retriever import hybrid_rank, build_context_string
from src.utils.config_loader import load_config

# ---------------------------
# Load configuration
# ---------------------------
config = load_config()
CHROMA_DIR = config["CHROMA_DB_DIR"]
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
LLM_MODEL = config["LLM_MODEL"]

# ---------------------------
# Initialize embeddings and vector stores
# ---------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
SUBJECTS = {
    "english": Chroma(persist_directory=f"{CHROMA_DIR}/english", embedding_function=embeddings),
    "physics": Chroma(persist_directory=f"{CHROMA_DIR}/physics", embedding_function=embeddings),
    "biology": Chroma(persist_directory=f"{CHROMA_DIR}/biology", embedding_function=embeddings),
    "pakistan_studies": Chroma(persist_directory=f"{CHROMA_DIR}/pakistan_studies", embedding_function=embeddings),
}
print("✅ ChromaDB loaded for subjects:", ", ".join(SUBJECTS.keys()))

# ---------------------------
# Initialize memory and LLM
# ---------------------------
memory_manager = MemoryManager()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# ---------------------------
# Prompt template
# ---------------------------
prompt_template = """
You are an expert tutor. Use only the context from PDFs to answer accurately.
If the answer is not in the context, politely say you don’t know.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---------------------------
# Subject detection
# ---------------------------
def detect_subject(query: str) -> Optional[str]:
    """Return the subject inferred from the user's query."""
    q = query.lower()
    keywords = {
        "english": ["grammar", "english", "sentence", "verb", "noun", "essay", "adjective", "paragraph"],
        "physics": ["physics", "force", "motion", "law", "energy", "newton", "velocity", "optics", "gravity"],
        "biology": ["biology", "cell", "organism", "photosynthesis", "dna", "protein", "enzyme", "plant", "animal"],
        "pakistan_studies": ["pakistan", "independence", "quaid", "history", "movement", "1947", "constitution"]
    }
    for subject, keys in keywords.items():
        if any(word in q for word in keys):
            return subject
    return None

# ---------------------------
# RAG answer retrieval
# ---------------------------
def get_rag_answer(subject: str, query: str) -> str:
    """Retrieve relevant documents and return an LLM-generated answer."""
    retriever = SUBJECTS[subject].as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    reranked = hybrid_rank(docs, query, alpha=0.7, top_k=4)
    context = build_context_string(reranked)
    full_prompt = prompt.format(context=context, question=query)
    answer = llm.invoke(full_prompt)
    
    # Save to per-subject memory
    mem = memory_manager.get_memory(subject)
    if mem:
        mem.save_context({"input": query}, {"output": str(answer)})
    return answer

# ---------------------------
# Chat loop
# ---------------------------
def chat() -> None:
    """Main REPL loop for the CLI chatbot."""
    print("\n🤖 Tutor RAG bot is ready! Type 'exit' to quit.\n")

    while True:
        query = input("🧑 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Study smart and stay curious!")
            break

        # Show memory
        if query.lower() == "show memory":
            print("\n🧠 Conversation Memory (All Subjects):")
            for subj, mem in memory_manager.memories.items():
                print(f"\n🗂️ {subj.capitalize()} memory:")
                for m in getattr(mem, "chat_memory", []):
                    content = getattr(m, "content", str(m))
                    print(f"  {getattr(m, 'type', 'msg')}: {content}")
            continue

        # Sanitize input
        cleaned_query, flagged, reasons = sanitize_user_input(query)
        if flagged:
            print(f"⚠️ Input rejected: {'; '.join(reasons)}")
            continue

        # Small talk
        if is_small_talk(cleaned_query):
            print("🤖 Tutor:", is_small_talk(cleaned_query))
            continue

        # Out-of-scope queries
        if is_out_of_scope(cleaned_query):
            print("🤖 Tutor: Sorry, I can only help with English, Physics, Biology, or Pakistan Studies.")
            continue

        # Subject detection
        subject = detect_subject(cleaned_query)
        if not subject:
            print("🤖 Tutor: Please ask something related to English, Physics, Biology, or Pakistan Studies.")
            continue

        # Generate and display answer
        try:
            answer = get_rag_answer(subject, cleaned_query)
            print(f"📘 [{subject.capitalize()} Tutor]: {answer}")
        except Exception as e:
            print(f"⚠️ Error generating answer: {str(e)}")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    chat()
