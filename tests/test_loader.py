"""
Tests for src/ingestion/loader.py

All tests are purely local -- no real PDF is opened, no Gemini calls are made.
PyPDFLoader is patched so the tests run without any disk I/O beyond tmp_path.
"""
from unittest.mock import MagicMock, patch

import pytest

from ingestion.loader import load_pdf


def _make_doc(content="Some text.", source="fake.pdf", page=0):
    """Return a minimal LangChain-style Document mock."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source, "page": page}
    return doc


class TestLoadPdfMissingFile:
    def test_raises_file_not_found(self, tmp_path):
        missing = str(tmp_path / "no_such_file.pdf")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_pdf(missing)
        assert "no_such_file.pdf" in str(exc_info.value)

    def test_error_message_is_human_readable(self, tmp_path):
        missing = str(tmp_path / "missing.pdf")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_pdf(missing)
        msg = str(exc_info.value)
        assert "missing.pdf" in msg
        assert len(msg) > 10


class TestLoadPdfCorruptFile:
    def test_raises_value_error_on_loader_exception(self, tmp_path):
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a pdf")
        with patch("ingestion.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = Exception("PDF parse error")
            with pytest.raises(ValueError) as exc_info:
                load_pdf(str(bad_pdf))
        assert "bad.pdf" in str(exc_info.value)

    def test_value_error_chains_original_exception(self, tmp_path):
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"x")
        original = Exception("low-level parse failure")
        with patch("ingestion.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.side_effect = original
            with pytest.raises(ValueError) as exc_info:
                load_pdf(str(bad_pdf))
        assert exc_info.value.__cause__ is original


class TestLoadPdfEmptyDocuments:
    def test_raises_value_error_when_no_docs_returned(self, tmp_path):
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"%PDF-1.4 %%EOF")
        with patch("ingestion.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = []
            with pytest.raises(ValueError) as exc_info:
                load_pdf(str(empty_pdf))
        msg = str(exc_info.value)
        assert "empty.pdf" in msg or "No text" in msg


class TestLoadPdfHappyPath:
    def test_returns_list_of_documents(self, tmp_path):
        real_pdf = tmp_path / "sample.pdf"
        real_pdf.write_bytes(b"%PDF-1.4 %%EOF")
        mock_docs = [_make_doc("Page 1.", "sample.pdf", 0), _make_doc("Page 2.", "sample.pdf", 1)]
        with patch("ingestion.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = mock_docs
            result = load_pdf(str(real_pdf))
        assert result == mock_docs
        assert len(result) == 2

    def test_loader_called_with_correct_path(self, tmp_path):
        real_pdf = tmp_path / "sample.pdf"
        real_pdf.write_bytes(b"%PDF-1.4 %%EOF")
        with patch("ingestion.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [_make_doc()]
            load_pdf(str(real_pdf))
        MockLoader.assert_called_once_with(str(real_pdf))

    def test_document_metadata_is_preserved(self, tmp_path):
        real_pdf = tmp_path / "sample.pdf"
        real_pdf.write_bytes(b"%PDF-1.4 %%EOF")
        doc = _make_doc("Text.", "sample.pdf", 3)
        with patch("ingestion.loader.PyPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [doc]
            result = load_pdf(str(real_pdf))
        assert result[0].metadata["page"] == 3
        assert result[0].metadata["source"] == "sample.pdf"
