# src/agent/qa_agents.py
"""
QA Agent Factory for EduTutor RAG.

Creates RetrievalQA agents for each subject using the same LLM and retriever.
These agents are used by the router to handle queries intelligently per subject.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------------------------
# Build QA agent
# ---------------------------
def build_qa_agent(subject_name: str, llm, retriever) -> RetrievalQA:
    """
    Construct a RetrievalQA chain for a specific subject.

    Parameters:
    - subject_name: Name of the subject (e.g., "biology").
    - llm: Language model instance (ChatOpenAI, etc.).
    - retriever: Vectorstore retriever for the subject.

    Returns:
    - RetrievalQA instance configured with a subject-specific prompt.
    """
    template = (
        "You are a knowledgeable and friendly tutor in {subject_name}.\n"
        "Use only the following context to answer accurately and clearly.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        partial_variables={"subject_name": subject_name.capitalize()}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        verbose=False
    )
    return qa_chain
