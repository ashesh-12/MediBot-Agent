"""
tools/disease_qa.py
LangChain tool — answers specific questions about
diseases, their causes, risk factors, and diagnosis.
"""

from langchain.tools import tool
from rag.retriever import get_retriever


@tool
def disease_qa(question: str) -> str:
    """
    Use this tool when the user asks a specific question about a disease,
    its causes, risk factors, complications, or diagnosis.
    Input should be a clear question about a disease or medical condition.
    Returns relevant medical information from the knowledge base.
    """
    retriever = get_retriever(k=4)
    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found for that disease or condition."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i} — Page {page}]\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(context_parts)

    return (
        f"Here is the medical information regarding '{question}':\n\n{context}"
    )