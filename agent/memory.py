"""
agent/memory.py
Manages conversation history for the MediBot agent.
Keeps a rolling window of the last N exchanges.
"""

from langchain.memory import ConversationBufferWindowMemory


def get_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """
    Returns a conversation buffer memory that keeps
    the last k human/AI exchanges in context.

    Args:
        k: Number of conversation turns to remember.

    Returns:
        ConversationBufferWindowMemory instance.
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
    )