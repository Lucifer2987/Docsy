from pathlib import Path
from unittest.mock import MagicMock
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


class TestAppInitialState:
    """Test initial rendering and default states without an uploaded document."""

    def test_initial_session_state_and_widgets(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)

        # Assert no exceptions occurred during execution
        assert not at.exception

        # Session state default initializations
        assert at.session_state["messages"] == []
        assert at.session_state["pipeline"] is None
        assert at.session_state["vector_store"] is None
        assert at.session_state["indexed_pdf_set"] == frozenset()
        assert at.session_state["doc_catalog"] == []
        assert at.session_state["current_pdf_id"] is None
        assert at.session_state["doc_stats"] is None

        # Verify welcome banner is shown
        assert len(at.info) == 1
        assert "Welcome to Docsy!" in at.info[0].value

        # Verify sidebar components
        assert len(at.sidebar.file_uploader) == 1
        assert at.sidebar.file_uploader[0].label == "Upload one or more PDF files"
        assert len(at.sidebar.slider) == 1
        assert at.sidebar.slider[0].value == 3
        assert len(at.sidebar.button) >= 1


class TestAppInteractions:
    """Test user interaction flows in session state and controls."""

    def test_clear_chat_button_resets_messages(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)

        # Populate messages in session state
        at.session_state["messages"] = [
            {"role": "user", "content": "What is data cleaning?", "sources": None},
            {"role": "assistant", "content": "It is removing noise.", "sources": [{"source": "sample.pdf", "page": 1}]},
        ]

        # Click the clear chat button
        at.sidebar.button[0].click().run(timeout=30)

        # Assert messages are cleared
        assert not at.exception
        assert at.session_state["messages"] == []

    def test_slider_k_updates_existing_retriever(self):
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)

        # Setup mock vector_store and pipeline in session_state
        mock_vs = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search_kwargs = {"k": 3}
        mock_pipeline = MagicMock()
        mock_pipeline.retriever = mock_retriever

        at.session_state["vector_store"] = mock_vs
        at.session_state["pipeline"] = mock_pipeline
        at.session_state["current_pdf_id"] = "test.pdf_100"

        # Change slider to 5
        at.sidebar.slider[0].set_value(5).run(timeout=30)

        assert not at.exception
        # create_retriever should have been called on mock_vs with k=5
        mock_vs.as_retriever.assert_called_with(search_type="similarity", search_kwargs={"k": 5})

    def test_document_catalog_display(self):
        """Test document catalog rendering and clear all documents action."""
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)

        # Set doc catalog
        at.session_state["doc_catalog"] = [
            {"name": "doc1.pdf", "pages": 5, "chunks": 12},
            {"name": "doc2.pdf", "pages": 3, "chunks": 7},
        ]
        at.session_state["pipeline"] = MagicMock()
        at.session_state["vector_store"] = MagicMock()
        at.session_state["indexed_pdf_set"] = frozenset(["doc1.pdf_10", "doc2.pdf_20"])

        at.run(timeout=30)
        assert not at.exception
        # Two buttons in sidebar: Clear Chat and Clear All Documents
        assert len(at.sidebar.button) == 2

