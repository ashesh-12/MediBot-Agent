"""
tests/test_tools.py
Unit tests for MediBot tools.
Tests retrieval and output format of all 3 tools.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def mock_docs():
    """Sample documents simulating FAISS retrieval."""
    return [
        Document(
            page_content="Fever is a common symptom of infection.",
            metadata={"source": "medical_book.pdf", "page": 10}
        ),
        Document(
            page_content="Headache may indicate tension or migraine.",
            metadata={"source": "medical_book.pdf", "page": 25}
        ),
    ]


# ── Symptom Checker Tests ──────────────────────────────────────

class TestSymptomChecker:

    @patch("tools.symptom_checker.get_retriever")
    def test_returns_string(self, mock_get_retriever, mock_docs):
        """Output must always be a string."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever

        from tools.symptom_checker import symptom_checker
        result = symptom_checker.invoke("fever and headache")
        assert isinstance(result, str)

    @patch("tools.symptom_checker.get_retriever")
    def test_contains_symptom_in_output(self, mock_get_retriever, mock_docs):
        """Output should reference the input symptoms."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever

        from tools.symptom_checker import symptom_checker
        result = symptom_checker.invoke("fever and headache")
        assert "fever and headache" in result

    @patch("tools.symptom_checker.get_retriever")
    def test_empty_retrieval(self, mock_get_retriever):
        """Should handle empty retrieval gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from tools.symptom_checker import symptom_checker
        result = symptom_checker.invoke("xyz unknown symptom")
        assert "No relevant" in result


# ── Disease Q&A Tests ──────────────────────────────────────────

class TestDiseaseQA:

    @patch("tools.disease_qa.get_retriever")
    def test_returns_string(self, mock_get_retriever, mock_docs):
        """Output must always be a string."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever

        from tools.disease_qa import disease_qa
        result = disease_qa.invoke("What causes diabetes?")
        assert isinstance(result, str)

    @patch("tools.disease_qa.get_retriever")
    def test_source_pages_in_output(self, mock_get_retriever, mock_docs):
        """Output should contain source page references."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever

        from tools.disease_qa import disease_qa
        result = disease_qa.invoke("What causes diabetes?")
        assert "Page" in result

    @patch("tools.disease_qa.get_retriever")
    def test_empty_retrieval(self, mock_get_retriever):
        """Should handle empty retrieval gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from tools.disease_qa import disease_qa
        result = disease_qa.invoke("nonexistent disease xyz")
        assert "No relevant" in result


# ── Treatment Summary Tests ────────────────────────────────────

class TestTreatmentSummary:

    @patch("tools.treatment_summary.get_retriever")
    @patch("tools.treatment_summary.ChatGroq")
    def test_returns_string(self, mock_llm_class, mock_get_retriever, mock_docs):
        """Output must always be a string."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(
            content="Take paracetamol and rest."
        )
        mock_llm.__or__ = MagicMock(return_value=mock_chain)

        from tools.treatment_summary import treatment_summary
        result = treatment_summary.invoke("common cold")
        assert isinstance(result, str)

    @patch("tools.treatment_summary.get_retriever")
    def test_empty_retrieval(self, mock_get_retriever):
        """Should handle empty retrieval gracefully."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from tools.treatment_summary import treatment_summary
        result = treatment_summary.invoke("xyz unknown condition")
        assert "No treatment information found" in result