# app_streamlit.py
import streamlit as st
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils.config_loader import load_config
from src.secuirity.sanitizer import sanitize_user_input
from src.utils.guardrails import is_small_talk, extract_name
from src.utils.memory_manager import MemoryManager
from src.rag.hybrib_retriever import hybrid_rank, build_context_string
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="EduTutor RAG", layout="wide")
st.title("EduTutor — Secure Multi-Subject RAG Tutor")

# ---------------------------
# Load config & models
# ---------------------------
config = load_config()
CHROMA_DIR = config["CHROMA_DB_DIR"]
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
LLM_MODEL = config["LLM_MODEL"]

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)

# ---------------------------
# Load Chroma collections
# ---------------------------
@st.cache_resource
def load_collections():
    return {
        "english": Chroma(persist_directory=f"{CHROMA_DIR}/english", embedding_function=embeddings),
        "physics": Chroma(persist_directory=f"{CHROMA_DIR}/physics", embedding_function=embeddings),
        "biology": Chroma(persist_directory=f"{CHROMA_DIR}/biology", embedding_function=embeddings),
        "pakistan_studies": Chroma(persist_directory=f"{CHROMA_DIR}/pakistan_studies", embedding_function=embeddings),
    }

SUBJECTS = load_collections()
memory_manager = MemoryManager()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Settings")
    subject_choice = st.selectbox(
        "Select subject (manual override)",
        ["auto", "english", "physics", "biology", "pakistan_studies"]
    )

    uploaded = st.file_uploader("Upload PDF for selected subject (PDF only)", type=["pdf"], accept_multiple_files=False)
    if uploaded:
        if subject_choice == "auto":
            st.warning("Please pick a subject in the dropdown to upload.")
        else:
            save_path = os.path.join("intelligent_tutor_bot", "data", {
                "english": "english.pdf",
                "physics": "physics_notes.pdf",
                "biology": "bio.pdf",
                "pakistan_studies": "pakstudies_dates.pdf"
            }[subject_choice])
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Saved PDF for {subject_choice}. Run ingestion or restart app to reload vectors.")
    st.markdown("---")
    if st.button("Show Memory (all)"):
        st.session_state.show_memory = True

# ---------------------------
# Session state
# ---------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "show_memory" not in st.session_state:
    st.session_state.show_memory = False

# ---------------------------
# RAG answer function ✅ FIXED
# ---------------------------
def get_rag_answer(subject: str, query: str, k_docs=8, top_k=4, alpha=0.7) -> str:
    """Retrieve docs, rerank, build context, call LLM, save memory, return plain text."""
    docs = SUBJECTS[subject].similarity_search(query, k=k_docs)
    reranked = hybrid_rank(docs, query, alpha=alpha, top_k=top_k)
    context = build_context_string(reranked)

    prompt_text = f"""
You are an expert tutor. Use only the context below to answer accurately.
If the answer is not found, say you don’t know.

Context:
{context}

Question:
{query}

Answer:
""".strip()

    response_obj = llm.invoke(prompt_text)

    # ✅ Extract only the text portion cleanly
    if hasattr(response_obj, "content"):
        answer = response_obj.content.strip()
    elif isinstance(response_obj, dict) and "output_text" in response_obj:
        answer = response_obj["output_text"].strip()
    else:
        answer = str(response_obj).strip()

    # ✅ Save to memory
    mem = memory_manager.get_memory(subject)
    if mem:
        mem.save_context({"input": query}, {"output": answer})

    return answer

# ---------------------------
# Chat layout
# ---------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Chat")
    for role, text in st.session_state.conversation:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Tutor:** {text}")

    query = st.text_input("Type your question and press Enter", key="input")

with col2:
    st.subheader("Controls")
    if st.button("Clear chat"):
        st.session_state.conversation = []
    st.markdown("💡 Tip: Say *'My name is <name>'* to store your name for this session.")

# ---------------------------
# Handle user input
# ---------------------------
if query:
    cleaned, flagged, reasons = sanitize_user_input(query)
    if flagged:
        st.error("❌ Input rejected for safety: " + "; ".join(reasons))
    else:
        maybe_name = extract_name(cleaned)
        if maybe_name:
            memory_manager.set_user_name(maybe_name)
            st.session_state.conversation.append(("tutor", f"Nice to meet you, {maybe_name}!"))
            st.success(f"Name saved: {maybe_name}")
        elif is_small_talk(cleaned):
            name = memory_manager.get_user_name() or "there"
            st.session_state.conversation.append(("tutor", f"Hello {name}, how can I help you today?"))
        else:
            # Detect subject
            subject = subject_choice if subject_choice != "auto" else None
            ql = cleaned.lower()
            if not subject:
                if any(w in ql for w in ["noun","verb","grammar","english"]): subject="english"
                elif any(w in ql for w in ["force","energy","motion","physics"]): subject="physics"
                elif any(w in ql for w in ["cell","protein","dna","biology"]): subject="biology"
                elif any(w in ql for w in ["pakistan","1947","independence","quaid"]): subject="pakistan_studies"

            if subject:
                try:
                    answer = get_rag_answer(subject, cleaned)
                    st.session_state.conversation.append(("user", cleaned))
                    st.session_state.conversation.append(("tutor", answer))
                except Exception as e:
                    st.error(f"⚠️ Generation error: {str(e)}")
            else:
                st.warning("⚠️ Couldn't detect subject. Try selecting manually from the sidebar.")

# ---------------------------
# Show memory
# ---------------------------
if st.session_state.show_memory:
    st.subheader("🧠 Stored Memory (per subject)")
    for subj, mem in memory_manager.memories.items():
        st.markdown(f"### {subj.capitalize()}")
        for m in getattr(mem, "chat_memory", []):
            content = getattr(m, "content", str(m))
            st.write(f"{getattr(m, 'type', 'msg')}: {content}")
    st.session_state.show_memory = False

# ---------------------------
# Download chat history
# ---------------------------
if st.session_state.conversation:
    st.markdown("---")
    chat_export = "\n".join([
        f"{'User' if r=='user' else 'Tutor'}: {t}"
        for r, t in st.session_state.conversation
    ])
    st.download_button(
        label="📥 Download Chat History",
        data=chat_export,
        file_name="chat_history.txt",
        mime="text/plain"
    )
