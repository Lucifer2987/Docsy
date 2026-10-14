"""
Tests for src/rag/pipeline.py (RAGPipeline)

No real Gemini calls are made. The retriever and RAG chain are mocked so tests
run instantly, deterministically, and without consuming any API quota.
"""
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Import after setting pythonpath via pyproject.toml
from rag.pipeline import RAGPipeline


class MockAPIError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message)
        self.code = code
        self.status_code = code


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(content="Some relevant text.", source="data/pdfs/sample.pdf", page=0):
    """Return a minimal LangChain Document mock."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


def _build_pipeline(retrieved_docs, chain_response_content="Mocked answer."):
    """
    Build a RAGPipeline whose retriever and chain are fully mocked.

    Parameters
    ----------
    retrieved_docs : list
        What the mocked retriever will return for any .invoke() call.
    chain_response_content : str | list
        The .content of the mocked LLM response object.
    """
    mock_response = MagicMock()
    mock_response.content = chain_response_content

    with (
        patch("rag.pipeline.create_retriever") as mock_create_retriever,
        patch("rag.pipeline.create_rag_chain") as mock_create_chain,
    ):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = retrieved_docs
        mock_create_retriever.return_value = mock_retriever

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_create_chain.return_value = mock_chain

        pipeline = RAGPipeline(vector_store=MagicMock(), k=2)

    # Expose mocks for per-test assertions
    pipeline._mock_retriever = mock_retriever
    pipeline._mock_chain = mock_chain
    pipeline._mock_response = mock_response
    return pipeline


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestAskInputValidation:
    def test_empty_string_raises_value_error(self):
        pipeline = _build_pipeline([])
        with pytest.raises(ValueError) as exc_info:
            pipeline.ask("")
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_raises_value_error(self):
        pipeline = _build_pipeline([])
        with pytest.raises(ValueError):
            pipeline.ask("   ")

    def test_none_raises_value_error(self):
        """Passing None should be caught by the empty-string check."""
        pipeline = _build_pipeline([])
        # None is falsy — the guard `if not question` triggers
        with pytest.raises((ValueError, AttributeError)):
            pipeline.ask(None)


# ---------------------------------------------------------------------------
# Empty retrieval
# ---------------------------------------------------------------------------

class TestAskEmptyRetrieval:
    def test_returns_fallback_dict_when_no_docs_retrieved(self):
        pipeline = _build_pipeline(retrieved_docs=[])
        result = pipeline.ask("What is X?")
        assert "answer" in result
        assert "sources" in result

    def test_fallback_sources_is_empty_list(self):
        pipeline = _build_pipeline(retrieved_docs=[])
        result = pipeline.ask("What is X?")
        assert result["sources"] == []

    def test_fallback_answer_contains_explanatory_text(self):
        pipeline = _build_pipeline(retrieved_docs=[])
        result = pipeline.ask("What is X?")
        # Answer should communicate that no content was found
        assert len(result["answer"]) > 10

    def test_chain_not_called_when_no_docs_retrieved(self):
        pipeline = _build_pipeline(retrieved_docs=[])
        pipeline.ask("What is X?")
        pipeline._mock_chain.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestAskHappyPath:
    def test_returns_answer_and_sources(self):
        docs = [_make_doc("Missing values were filled with mean.", "sample.pdf", 0)]
        pipeline = _build_pipeline(docs, chain_response_content="Mean imputation was used.")
        result = pipeline.ask("How were missing values handled?")
        assert result["answer"] == "Mean imputation was used."
        assert len(result["sources"]) == 1

    def test_source_metadata_correct(self):
        docs = [_make_doc("Content.", "sample.pdf", 2)]
        pipeline = _build_pipeline(docs)
        result = pipeline.ask("What is discussed?")
        assert result["sources"][0]["source"] == "sample.pdf"
        assert result["sources"][0]["page"] == 3  # page index 2 -> display page 3

    def test_chain_is_invoked_with_context_and_question(self):
        docs = [_make_doc("Some text.", "sample.pdf", 0)]
        pipeline = _build_pipeline(docs)
        pipeline.ask("My question")
        call_kwargs = pipeline._mock_chain.invoke.call_args[0][0]
        assert "context" in call_kwargs
        assert call_kwargs["question"] == "My question"


# ---------------------------------------------------------------------------
# Source deduplication
# ---------------------------------------------------------------------------

class TestAskSourceDeduplication:
    def test_duplicate_source_page_pairs_are_deduplicated(self):
        """Two docs from the same source+page must appear only once in sources."""
        docs = [
            _make_doc("Chunk A.", "sample.pdf", 1),
            _make_doc("Chunk B.", "sample.pdf", 1),  # same source + page
        ]
        pipeline = _build_pipeline(docs)
        result = pipeline.ask("Question?")
        assert len(result["sources"]) == 1

    def test_different_pages_are_not_deduplicated(self):
        """Docs from different pages of the same file must both appear."""
        docs = [
            _make_doc("Chunk A.", "sample.pdf", 0),
            _make_doc("Chunk B.", "sample.pdf", 1),
        ]
        pipeline = _build_pipeline(docs)
        result = pipeline.ask("Question?")
        assert len(result["sources"]) == 2

    def test_source_order_follows_retrieval_order(self):
        """Sources should appear in the order they were retrieved."""
        docs = [
            _make_doc("Chunk A.", "sample.pdf", 3),
            _make_doc("Chunk B.", "sample.pdf", 1),
        ]
        pipeline = _build_pipeline(docs)
        result = pipeline.ask("Question?")
        assert result["sources"][0]["page"] == 4  # page index 3 -> display 4
        assert result["sources"][1]["page"] == 2


# ---------------------------------------------------------------------------
# _extract_response_text
# ---------------------------------------------------------------------------

class TestExtractResponseText:
    def _pipeline(self):
        return _build_pipeline([_make_doc()])

    def test_string_content_returned_directly(self):
        p = self._pipeline()
        resp = MagicMock()
        resp.content = "Direct string."
        assert p._extract_response_text(resp) == "Direct string."

    def test_list_content_text_blocks_joined(self):
        p = self._pipeline()
        resp = MagicMock()
        resp.content = [
            {"type": "text", "text": "Part one."},
            {"type": "text", "text": "Part two."},
        ]
        result = p._extract_response_text(resp)
        assert "Part one." in result
        assert "Part two." in result

    def test_list_content_non_text_blocks_ignored(self):
        p = self._pipeline()
        resp = MagicMock()
        resp.content = [
            {"type": "tool_use", "id": "call_1"},
            {"type": "text", "text": "Only text block."},
        ]
        result = p._extract_response_text(resp)
        assert "Only text block." in result
        assert "tool_use" not in result

    def test_unknown_content_type_falls_back_to_str(self):
        p = self._pipeline()
        resp = MagicMock()
        resp.content = 12345  # neither str nor list
        result = p._extract_response_text(resp)
        assert result == "12345"


# ---------------------------------------------------------------------------
# API error handling — 429 quota exhausted
# ---------------------------------------------------------------------------

class TestAskQuotaError:
    def test_client_error_429_raises_runtime_error(self):
        docs = [_make_doc()]
        pipeline = _build_pipeline(docs)

        quota_error = MockAPIError(code=429, message="Rate limit reached: 429 RESOURCE_EXHAUSTED")
        pipeline._mock_chain.invoke.side_effect = quota_error

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.ask("Question?")

        msg = str(exc_info.value).lower()
        assert "429" in msg or "quota" in msg or "rate limit" in msg

    def test_quota_error_message_contains_guidance(self):
        docs = [_make_doc()]
        pipeline = _build_pipeline(docs)

        quota_error = MockAPIError(code=429, message="Rate limit reached: 429 RESOURCE_EXHAUSTED")
        pipeline._mock_chain.invoke.side_effect = quota_error

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.ask("Question?")

        # The message must tell the user what to do — not just rethrow the raw error.
        msg = str(exc_info.value).lower()
        assert any(keyword in msg for keyword in ["wait", "limit", "429", "groq"])


# ---------------------------------------------------------------------------
# API error handling — 5xx server error
# ---------------------------------------------------------------------------

class TestAskServerError:
    def test_server_error_raises_runtime_error(self):
        docs = [_make_doc()]
        pipeline = _build_pipeline(docs)

        server_error = MockAPIError(code=503, message="Service unavailable")
        pipeline._mock_chain.invoke.side_effect = server_error

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.ask("Question?")

        msg = str(exc_info.value).lower()
        assert "unavailable" in msg or "5xx" in msg or "temporarily" in msg

    def test_server_error_message_is_human_readable(self):
        docs = [_make_doc()]
        pipeline = _build_pipeline(docs)

        server_error = MockAPIError(code=503, message="Service unavailable")
        pipeline._mock_chain.invoke.side_effect = server_error

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.ask("Question?")

        # Should not expose raw exception class names in the message
        assert len(str(exc_info.value)) > 20


# ---------------------------------------------------------------------------
# Unexpected exception passthrough
# ---------------------------------------------------------------------------

class TestAskUnexpectedException:
    def test_unexpected_exception_wrapped_in_runtime_error(self):
        docs = [_make_doc()]
        pipeline = _build_pipeline(docs)
        pipeline._mock_chain.invoke.side_effect = OSError("network unreachable")

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.ask("Question?")

        assert "unexpected" in str(exc_info.value).lower() or "network" in str(exc_info.value).lower()
