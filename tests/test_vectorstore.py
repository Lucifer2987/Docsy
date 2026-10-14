"""
Tests for src/vectorstore/store.py

Covers: create_vector_store, save_vector_store, load_vector_store,
        write_index_meta, and is_index_fresh.

No real embeddings or FAISS operations are performed.
HuggingFaceEmbeddings and FAISS are fully mocked.
"""
import json
import os
import time
from unittest.mock import MagicMock, patch, call

import pytest

from vectorstore.store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    write_index_meta,
    is_index_fresh,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(content="text", source="doc.pdf", page=0):
    chunk = MagicMock()
    chunk.page_content = content
    chunk.metadata = {"source": source, "page": page}
    return chunk


# ---------------------------------------------------------------------------
# create_vector_store
# ---------------------------------------------------------------------------

class TestCreateVectorStore:
    def test_returns_faiss_store(self):
        """create_vector_store must return whatever FAISS.from_documents returns."""
        mock_store = MagicMock()
        with (
            patch("vectorstore.store.get_embedding_model") as mock_embed,
            patch("vectorstore.store.FAISS") as MockFAISS,
        ):
            MockFAISS.from_documents.return_value = mock_store
            result = create_vector_store([_make_chunk()])

        assert result is mock_store

    def test_passes_chunks_and_embeddings_to_faiss(self):
        chunks = [_make_chunk("a"), _make_chunk("b")]
        mock_emb = MagicMock()
        mock_store = MagicMock()

        with (
            patch("vectorstore.store.get_embedding_model", return_value=mock_emb),
            patch("vectorstore.store.FAISS") as MockFAISS,
        ):
            MockFAISS.from_documents.return_value = mock_store
            create_vector_store(chunks)

        MockFAISS.from_documents.assert_called_once_with(chunks, mock_emb)


# ---------------------------------------------------------------------------
# save_vector_store
# ---------------------------------------------------------------------------

class TestSaveVectorStore:
    def test_creates_index_directory(self, tmp_path):
        index_dir = str(tmp_path / "vectorstore")
        mock_vs = MagicMock()
        save_vector_store(mock_vs, index_dir)
        assert os.path.isdir(index_dir)

    def test_calls_save_local_with_faiss_subfolder(self, tmp_path):
        index_dir = str(tmp_path / "vectorstore")
        mock_vs = MagicMock()
        save_vector_store(mock_vs, index_dir)
        # FAISS files should land in <index_dir>/faiss_index/
        expected_path = os.path.join(index_dir, "faiss_index")
        mock_vs.save_local.assert_called_once_with(expected_path)


# ---------------------------------------------------------------------------
# load_vector_store
# ---------------------------------------------------------------------------

class TestLoadVectorStore:
    def test_raises_file_not_found_when_no_index(self, tmp_path):
        """load_vector_store must raise FileNotFoundError if no index exists."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_vector_store(str(tmp_path))
        assert "faiss_index" in str(exc_info.value) or "index" in str(exc_info.value).lower()

    def test_returns_loaded_store(self, tmp_path):
        """When index files exist, load_vector_store must return FAISS.load_local result."""
        # Create the directory + dummy index.faiss file so the existence check passes
        faiss_dir = tmp_path / "faiss_index"
        faiss_dir.mkdir()
        (faiss_dir / "index.faiss").write_bytes(b"dummy")

        mock_vs = MagicMock()
        with (
            patch("vectorstore.store.get_embedding_model") as mock_embed,
            patch("vectorstore.store.FAISS") as MockFAISS,
        ):
            MockFAISS.load_local.return_value = mock_vs
            result = load_vector_store(str(tmp_path))

        assert result is mock_vs

    def test_passes_allow_dangerous_deserialization(self, tmp_path):
        """allow_dangerous_deserialization=True must always be passed."""
        faiss_dir = tmp_path / "faiss_index"
        faiss_dir.mkdir()
        (faiss_dir / "index.faiss").write_bytes(b"dummy")

        with (
            patch("vectorstore.store.get_embedding_model"),
            patch("vectorstore.store.FAISS") as MockFAISS,
        ):
            MockFAISS.load_local.return_value = MagicMock()
            load_vector_store(str(tmp_path))

        _, kwargs = MockFAISS.load_local.call_args
        assert kwargs.get("allow_dangerous_deserialization") is True


# ---------------------------------------------------------------------------
# write_index_meta
# ---------------------------------------------------------------------------

class TestWriteIndexMeta:
    def test_creates_meta_json_file(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        write_index_meta(str(tmp_path), str(pdf))
        assert (tmp_path / "meta.json").exists()

    def test_meta_contains_source_path(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        write_index_meta(str(tmp_path), str(pdf))
        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["source_path"] == os.path.abspath(str(pdf))

    def test_meta_contains_mtime_and_size(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 sample content here")
        stat = pdf.stat()
        write_index_meta(str(tmp_path), str(pdf))
        with open(tmp_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["source_mtime"] == stat.st_mtime
        assert meta["source_size"] == stat.st_size

    def test_creates_parent_directory_if_absent(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        new_dir = tmp_path / "nested" / "store"
        write_index_meta(str(new_dir), str(pdf))
        assert (new_dir / "meta.json").exists()


# ---------------------------------------------------------------------------
# is_index_fresh
# ---------------------------------------------------------------------------

class TestIsIndexFresh:
    def _build_fresh_index(self, tmp_path):
        """Create a PDF and a matching index+meta so is_index_fresh returns True."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 some content")
        index_dir = tmp_path / "store"
        faiss_dir = index_dir / "faiss_index"
        faiss_dir.mkdir(parents=True)
        (faiss_dir / "index.faiss").write_bytes(b"dummy")
        write_index_meta(str(index_dir), str(pdf))
        return str(index_dir), str(pdf)

    def test_returns_true_when_index_matches_pdf(self, tmp_path):
        index_dir, pdf = self._build_fresh_index(tmp_path)
        assert is_index_fresh(index_dir, pdf) is True

    def test_returns_false_when_meta_missing(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        # No meta.json written
        assert is_index_fresh(str(tmp_path), str(pdf)) is False

    def test_returns_false_when_faiss_file_missing(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        # Write meta but no FAISS file
        write_index_meta(str(tmp_path), str(pdf))
        assert is_index_fresh(str(tmp_path), str(pdf)) is False

    def test_returns_false_when_pdf_content_changes(self, tmp_path):
        index_dir, pdf_path = self._build_fresh_index(tmp_path)
        # Overwrite the PDF with different content (changes size and mtime)
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 completely different content that is much longer")
        assert is_index_fresh(index_dir, pdf_path) is False

    def test_returns_false_when_meta_json_is_corrupt(self, tmp_path):
        index_dir, pdf = self._build_fresh_index(tmp_path)
        # Corrupt the meta.json
        with open(os.path.join(index_dir, "meta.json"), "w") as f:
            f.write("this is not valid JSON }{")
        assert is_index_fresh(index_dir, pdf) is False

    def test_returns_false_when_index_dir_does_not_exist(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        non_existent = str(tmp_path / "no_such_dir")
        assert is_index_fresh(non_existent, str(pdf)) is False
