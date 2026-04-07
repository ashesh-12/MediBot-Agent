"""
tools/symptom_checker.py
LangChain tool — given a list of symptoms, retrieves
the most relevant medical context and returns a
structured summary of possible conditions.
"""

from langchain.tools import tool
from rag.retriever import get_retriever


@tool
def symptom_checker(symptoms: str) -> str:
    """
    Use this tool when the user describes symptoms they are experiencing.
    Input should be a comma-separated list of symptoms or a natural
    language description of symptoms.
    Returns possible conditions and relevant medical context.
    """
    retriever = get_retriever(k=4)
    query = f"symptoms: {symptoms}"
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant medical information found for the given symptoms."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i} — Page {page}]\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(context_parts)

    return (
        f"Based on the symptoms '{symptoms}', "
        f"here is the relevant medical information:\n\n{context}"
    )