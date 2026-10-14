import os
import sys
import tempfile
from pathlib import Path

# Ensure src/ is accessible for imports regardless of launch directory
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from ingestion.loader import load_pdf
from ingestion.splitter import split_documents
from vectorstore.store import create_vector_store
from retrieval.retriever import create_retriever
from rag.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Docsy - RAG PDF Reader",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None

if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "current_pdf_id" not in st.session_state:
    st.session_state["current_pdf_id"] = None

if "doc_stats" not in st.session_state:
    st.session_state["doc_stats"] = None


# ---------------------------------------------------------------------------
# Sidebar: Document Management & Settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Docsy")
    st.caption("Grounded RAG PDF Assistant")
    st.divider()

    st.subheader("1. Ingestion")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        help="Upload a PDF document to index and query.",
    )

    st.subheader("2. Retrieval Settings")
    top_k = st.slider(
        "Context Chunks (k)",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of relevant document chunks retrieved to answer each question.",
    )

    # Dynamically update retriever k if pipeline already exists
    if (
        st.session_state["pipeline"] is not None
        and st.session_state["vector_store"] is not None
    ):
        current_k = st.session_state["pipeline"].retriever.search_kwargs.get("k")
        if current_k != top_k:
            st.session_state["pipeline"].retriever = create_retriever(
                st.session_state["vector_store"],
                k=top_k,
            )

    st.divider()

    # Document stats card
    if st.session_state["doc_stats"]:
        stats = st.session_state["doc_stats"]
        st.subheader("📄 Active Document")
        st.markdown(f"**Name:** `{stats['name']}`")
        st.markdown(f"**Pages:** `{stats['pages']}`")
        st.markdown(f"**Chunks:** `{stats['chunks']}`")
        st.markdown("**Status:** Ready ✅")
        st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# PDF Ingestion Workflow
# ---------------------------------------------------------------------------

if uploaded_file is not None:
    pdf_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state["current_pdf_id"] != pdf_id:
        temp_path = None
        try:
            with st.spinner("Extracting text and generating embeddings..."):
                # Write uploaded bytes to a safe temporary file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                # 1. Load PDF
                docs = load_pdf(temp_path)

                # Set source metadata to original filename for clear citations
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                # 2. Split documents into chunks
                chunks = split_documents(docs)
                if not chunks:
                    st.error(
                        "No text could be extracted from this PDF. "
                        "The document may contain only scanned images."
                    )
                    st.stop()

                # 3. Create FAISS vector store
                vector_store = create_vector_store(chunks)

                # 4. Initialize RAG pipeline
                pipeline = RAGPipeline(vector_store, k=top_k)

                # 5. Store in session state
                st.session_state["pipeline"] = pipeline
                st.session_state["vector_store"] = vector_store
                st.session_state["current_pdf_id"] = pdf_id
                st.session_state["doc_stats"] = {
                    "name": uploaded_file.name,
                    "pages": len(docs),
                    "chunks": len(chunks),
                }
                st.session_state["messages"] = []

            st.toast(f"'{uploaded_file.name}' indexed successfully!", icon="✅")
            st.rerun()

        except FileNotFoundError as exc:
            st.error(f"File error: {exc}")
        except ValueError as exc:
            st.error(f"Document error: {exc}")
        except Exception as exc:
            st.error(f"Failed to process PDF: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

elif st.session_state["current_pdf_id"] is not None:
    # User removed the uploaded file
    st.session_state["pipeline"] = None
    st.session_state["vector_store"] = None
    st.session_state["current_pdf_id"] = None
    st.session_state["doc_stats"] = None
    st.session_state["messages"] = []
    st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">Docsy Reader</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions grounded in your document. Answers include exact page citations.</div>',
    unsafe_allow_html=True,
)

# Empty state banner if no document is loaded
if st.session_state["pipeline"] is None:
    st.info(
        "👋 **Welcome to Docsy!** To get started:\n\n"
        "1. Upload a PDF using the sidebar on the left.\n"
        "2. Wait a few seconds for the document to be indexed.\n"
        "3. Ask any question in the chat input below."
    )
else:
    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['source']}** — Page {src['page']}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Record user message
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "sources": None,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document & formulating answer..."):
                try:
                    result = st.session_state["pipeline"].ask(prompt)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander(f"📄 Sources ({len(sources)})"):
                            for src in sources:
                                st.markdown(f"- **{src['source']}** — Page {src['page']}")

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except ValueError as exc:
                    st.warning(f"Input error: {exc}")
                except RuntimeError as exc:
                    st.error(f"API / Pipeline error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

