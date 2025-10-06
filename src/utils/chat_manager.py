# src/utils/chat_manager.py
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from src.utils.memory_manager import MemoryManager
from src.rag.hybrib_retriever import hybrid_rank, build_context_string

class ChatManager:
    """Manages chat sessions, subject detection, and RAG-based responses."""

    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config["EMBEDDING_MODEL"])
        self.llm = ChatOpenAI(model=config["LLM_MODEL"], temperature=0.2)
        self.memory_manager = MemoryManager()

        # Load subject vectorstores
        CHROMA_DIR = config["CHROMA_DB_DIR"]
        self.subjects = {
            "english": Chroma(persist_directory=f"{CHROMA_DIR}/english", embedding_function=self.embeddings),
            "physics": Chroma(persist_directory=f"{CHROMA_DIR}/physics", embedding_function=self.embeddings),
            "biology": Chroma(persist_directory=f"{CHROMA_DIR}/biology", embedding_function=self.embeddings),
            "pakistan_studies": Chroma(persist_directory=f"{CHROMA_DIR}/pakistan_studies", embedding_function=self.embeddings),
        }

        # Prompt template
        self.prompt = PromptTemplate(
            template="""
You are an expert tutor. Use only the context from PDFs to answer accurately.
If the answer is not in the context, politely say you don’t know.

Context:
{context}

Question:
{question}

Answer:
""",
            input_variables=["context", "question"]
        )

    # ---------------------------
    # Detect subject
    # ---------------------------
    def detect_subject(self, query: str):
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
    # Generate RAG answer
    # ---------------------------
    def get_rag_answer(self, subject, query, k_docs=3, top_k=4, alpha=0.7):
        """Retrieve context from vectorstore, rerank, and get LLM answer."""
        retriever = self.subjects[subject].as_retriever(search_kwargs={"k": k_docs})
        docs = retriever.get_relevant_documents(query)
        reranked = hybrid_rank(docs, query, alpha=alpha, top_k=top_k)
        context = build_context_string(reranked)
        full_prompt = self.prompt.format(context=context, question=query)
        answer = self.llm.invoke(full_prompt)

        # Save to per-subject memory
        mem = self.memory_manager.get_memory(subject)
        if mem:
            mem.save_context({"input": query}, {"output": str(answer)})
        return answer

    # ---------------------------
    # Handle small talk
    # ---------------------------
    def handle_small_talk(self, query):
        """Return canned responses for greetings and common phrases."""
        responses = {
            "hello": "Hello! How can I assist you today?",
            "hi": "Hi there! What subject would you like to learn about?",
            "how are you": "I’m just a virtual tutor, but I’m always ready to help you learn! 😊",
            "thanks": "You're very welcome!",
            "thank you": "Glad to help!",
        }
        for key, value in responses.items():
            if key in query.lower():
                return value
        return "I'm here and ready to assist you with your studies!"
