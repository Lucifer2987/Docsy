"""
Tests for src/retrieval/retriever.py

No FAISS index is loaded. We verify create_retriever wires the vector store
correctly, passing search_type and k through to the underlying method.
"""
from unittest.mock import MagicMock, call

import pytest

from retrieval.retriever import create_retriever


class TestCreateRetriever:
    def test_returns_retriever_object(self):
        """create_retriever must return whatever as_retriever returns."""
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        result = create_retriever(mock_vs)

        assert result is mock_retriever

    def test_uses_similarity_search_type(self):
        """as_retriever must be called with search_type='similarity'."""
        mock_vs = MagicMock()
        create_retriever(mock_vs)

        _, kwargs = mock_vs.as_retriever.call_args
        assert kwargs.get("search_type") == "similarity"

    def test_default_k_is_three(self):
        """Default k should be 3 (as configured in retriever.py)."""
        mock_vs = MagicMock()
        create_retriever(mock_vs)

        _, kwargs = mock_vs.as_retriever.call_args
        assert kwargs.get("search_kwargs", {}).get("k") == 3

    def test_custom_k_is_passed_through(self):
        """Passing k=5 must result in search_kwargs={'k': 5}."""
        mock_vs = MagicMock()
        create_retriever(mock_vs, k=5)

        _, kwargs = mock_vs.as_retriever.call_args
        assert kwargs.get("search_kwargs", {}).get("k") == 5

    def test_as_retriever_called_exactly_once(self):
        """as_retriever should be called exactly once per create_retriever call."""
        mock_vs = MagicMock()
        create_retriever(mock_vs)
        mock_vs.as_retriever.assert_called_once()
