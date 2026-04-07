"""
tools/treatment_summary.py
LangChain tool — retrieves treatment options and
summarizes them using the LLM.
"""

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from rag.retriever import get_retriever
import os


TREATMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a medical assistant. Using only the context provided, "
        "give a concise summary of treatment options. "
        "List treatments clearly. "
        "Always end with: 'Please consult a qualified doctor before "
        "starting any treatment.'\n\nContext:\n{context}"
    ),
    ("human", "{question}"),
])


@tool
def treatment_summary(condition: str) -> str:
    """
    Use this tool when the user asks about treatments, medications,
    therapies, or management options for a disease or condition.
    Input should be the name of a disease or condition.
    Returns a structured treatment summary with sources.
    """
    retriever = get_retriever(k=5)
    query = f"treatment options for {condition}"
    docs = retriever.invoke(query)

    if not docs:
        return f"No treatment information found for '{condition}'."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i} — Page {page}]\n{doc.page_content.strip()}"
        )

    context = "\n\n".join(context_parts)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    chain = TREATMENT_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "question": f"What are the treatments for {condition}?",
    })

    sources = ", ".join(
        f"p.{doc.metadata.get('page', '?')}" for doc in docs
    )

    return f"{response.content}\n\n📚 Sources: {response.content}\n Pages: {sources}"