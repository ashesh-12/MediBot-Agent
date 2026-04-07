"""
tests/test_agent.py
Unit tests for the MediBot agent and memory.
"""

import pytest
from unittest.mock import patch, MagicMock
from agent.memory import get_memory
from agent.medibot_agent import get_llm, TOOLS


# ── Memory Tests ───────────────────────────────────────────────

class TestMemory:

    def test_memory_returns_correct_type(self):
        """Memory should return ConversationBufferWindowMemory."""
        from langchain.memory import ConversationBufferWindowMemory
        memory = get_memory(k=3)
        assert isinstance(memory, ConversationBufferWindowMemory)

    def test_memory_key(self):
        """Memory key must be chat_history for agent compatibility."""
        memory = get_memory()
        assert memory.memory_key == "chat_history"

    def test_memory_window_size(self):
        """Memory window k should match what was passed."""
        memory = get_memory(k=7)
        assert memory.k == 7

    def test_memory_returns_messages(self):
        """Memory should return messages not strings."""
        memory = get_memory()
        assert memory.return_messages is True


# ── LLM Tests ─────────────────────────────────────────────────

class TestLLM:

    def test_llm_returns_correct_type(self):
        """LLM should return a ChatGroq instance."""
        from langchain_groq import ChatGroq
        llm = get_llm()
        assert isinstance(llm, ChatGroq)

    def test_llm_model_name(self):
        """Should use the correct Groq model."""
        llm = get_llm()
        assert llm.model_name == "llama-3.3-70b-versatile"


# ── Tools Tests ────────────────────────────────────────────────

class TestTools:

    def test_correct_number_of_tools(self):
        """Agent should have exactly 3 tools."""
        assert len(TOOLS) == 3

    def test_tool_names(self):
        """All 3 expected tools should be present."""
        tool_names = [t.name for t in TOOLS]
        assert "symptom_checker" in tool_names
        assert "disease_qa" in tool_names
        assert "treatment_summary" in tool_names

    def test_tools_have_descriptions(self):
        """Every tool must have a non-empty description."""
        for tool in TOOLS:
            assert tool.description is not None
            assert len(tool.description) > 10