# src/prompts/templates.py
"""
Centralized location for all LLM prompt templates used in EduTutor RAG.
"""

from langchain.prompts import PromptTemplate

# ---------------------------
# Tutor Prompt
# ---------------------------
tutor_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an intelligent multi-subject tutor. "
        "Answer clearly and concisely based strictly on the context provided. "
        "Do not guess or provide information outside the context.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
)

# ---------------------------
# Summarization / Explanation Prompt
# ---------------------------
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following educational text in simple language for students:\n\n{text}"
    ),
)
