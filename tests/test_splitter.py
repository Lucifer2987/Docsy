"""
Tests for src/ingestion/splitter.py

No Gemini calls, no embeddings, no network I/O.
We exercise the splitter with in-memory Document objects.
"""
from unittest.mock import MagicMock

import pytest

from ingestion.splitter import split_documents


def _make_doc(content, source="test.pdf", page=0):
    """Minimal LangChain Document-like object."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


class TestSplitDocuments:
    def test_returns_list(self):
        """split_documents must always return a list."""
        docs = [_make_doc("Short text.")]
        result = split_documents(docs)
        assert isinstance(result, list)

    def test_short_text_stays_as_single_chunk(self):
        """Text shorter than chunk_size should not be split further."""
        short = "This is a short sentence."
        docs = [_make_doc(short)]
        chunks = split_documents(docs)
        # All content should still be present
        combined = " ".join(c.page_content for c in chunks)
        assert short in combined

    def test_long_text_is_split_into_multiple_chunks(self):
        """Text longer than chunk_size (800) must produce more than one chunk."""
        # 3000 characters of text — well above the 800-char chunk_size
        long_text = ("word " * 600).strip()
        docs = [_make_doc(long_text)]
        chunks = split_documents(docs)
        assert len(chunks) > 1

    def test_chunk_size_not_exceeded(self):
        """No individual chunk should exceed chunk_size + overlap (800+150=950)."""
        long_text = ("word " * 600).strip()
        docs = [_make_doc(long_text)]
        chunks = split_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= 950, (
                f"Chunk length {len(chunk.page_content)} exceeds allowed max"
            )

    def test_empty_document_list_returns_empty_list(self):
        """Passing an empty list must return an empty list without raising."""
        result = split_documents([])
        assert result == []

    def test_metadata_is_preserved_in_chunks(self):
        """source and page metadata from the original Document must survive chunking."""
        long_text = ("word " * 600).strip()
        docs = [_make_doc(long_text, source="data/pdfs/sample.pdf", page=2)]
        chunks = split_documents(docs)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "data/pdfs/sample.pdf"
            assert chunk.metadata.get("page") == 2

    def test_multiple_pages_produce_chunks_with_correct_pages(self):
        """Chunks from page 0 and page 1 should retain their respective page numbers."""
        page0 = _make_doc("word " * 300, source="doc.pdf", page=0)
        page1 = _make_doc("word " * 300, source="doc.pdf", page=1)
        chunks = split_documents([page0, page1])
        pages = {c.metadata["page"] for c in chunks}
        assert 0 in pages
        assert 1 in pages
