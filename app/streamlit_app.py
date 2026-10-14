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

if "indexed_pdf_set" not in st.session_state:
    # frozenset of "filename_size" fingerprints for the currently indexed files
    st.session_state["indexed_pdf_set"] = frozenset()

if "doc_catalog" not in st.session_state:
    # List of dicts: {name, pages, chunks}
    st.session_state["doc_catalog"] = []

if "current_pdf_id" not in st.session_state:
    st.session_state["current_pdf_id"] = None

if "doc_stats" not in st.session_state:
    st.session_state["doc_stats"] = None

if "uploader_active" not in st.session_state:
    st.session_state["uploader_active"] = False


# ---------------------------------------------------------------------------
# Helper: build fingerprint set from uploaded file list
# ---------------------------------------------------------------------------

def _pdf_fingerprint_set(files) -> frozenset:
    """Return a frozenset of 'filename_size' strings for the given file list."""
    return frozenset(f"{f.name}_{f.size}" for f in files)


# ---------------------------------------------------------------------------
# Sidebar: Document Management & Settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Docsy")
    st.caption("Grounded RAG PDF Assistant")
    st.divider()

    st.subheader("1. Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents. All files are indexed into a single searchable knowledge base.",
    )

    st.subheader("2. Retrieval Settings")
    top_k = st.slider(
        "Context Chunks (k)",
        min_value=1,
        max_value=8,
        value=3,
        help="Number of relevant chunks retrieved per question. With multiple PDFs, a higher k gives broader coverage.",
    )

    # Dynamically update retriever k or refresh pipeline if LLM changed
    if (
        st.session_state["pipeline"] is not None
        and st.session_state["vector_store"] is not None
    ):
        try:
            from langchain_groq import ChatGroq
            steps = getattr(st.session_state["pipeline"].rag_chain, "steps", [])
            if steps and not isinstance(steps[-1], ChatGroq):
                st.session_state["pipeline"] = RAGPipeline(
                    st.session_state["vector_store"],
                    k=top_k,
                )
        except Exception:
            pass

        current_k = st.session_state["pipeline"].retriever.search_kwargs.get("k")
        if current_k != top_k:
            st.session_state["pipeline"].retriever = create_retriever(
                st.session_state["vector_store"],
                k=top_k,
            )

    st.divider()

    # Document catalog card
    if st.session_state["doc_catalog"]:
        st.subheader("📂 Indexed Documents")
        for doc in st.session_state["doc_catalog"]:
            with st.expander(f"📄 {doc['name']}", expanded=False):
                st.markdown(f"**Pages:** {doc['pages']}")
                st.markdown(f"**Chunks:** {doc['chunks']}")
        st.divider()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    if st.session_state["doc_catalog"] and st.button(
        "🗑️ Clear All Documents", use_container_width=True
    ):
        st.session_state["uploader_active"] = False
        st.session_state["pipeline"] = None
        st.session_state["vector_store"] = None
        st.session_state["indexed_pdf_set"] = frozenset()
        st.session_state["doc_catalog"] = []
        st.session_state["current_pdf_id"] = None
        st.session_state["doc_stats"] = None
        st.session_state["messages"] = []
        st.rerun()


# ---------------------------------------------------------------------------
# PDF Ingestion Workflow
# ---------------------------------------------------------------------------

if uploaded_files:
    st.session_state["uploader_active"] = True
    current_fingerprint_set = _pdf_fingerprint_set(uploaded_files)

    if st.session_state["indexed_pdf_set"] != current_fingerprint_set:
        # File set changed — rebuild the combined index
        all_chunks = []
        catalog = []
        error_occurred = False

        progress_bar = st.progress(0, text="Starting ingestion...")

        for i, uploaded_file in enumerate(uploaded_files):
            progress_fraction = i / len(uploaded_files)
            progress_bar.progress(
                progress_fraction,
                text=f"Ingesting '{uploaded_file.name}' ({i + 1}/{len(uploaded_files)})...",
            )
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                docs = load_pdf(temp_path)

                # Map source metadata to original uploaded filename
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name

                chunks = split_documents(docs)
                if not chunks:
                    st.warning(
                        f"'{uploaded_file.name}' produced no text chunks. "
                        "It may be a scanned/image-only PDF and will be skipped."
                    )
                    continue

                all_chunks.extend(chunks)
                catalog.append({
                    "name": uploaded_file.name,
                    "pages": len(docs),
                    "chunks": len(chunks),
                })

            except FileNotFoundError as exc:
                st.error(f"File error for '{uploaded_file.name}': {exc}")
                error_occurred = True
            except ValueError as exc:
                st.error(f"Document error for '{uploaded_file.name}': {exc}")
                error_occurred = True
            except Exception as exc:
                st.error(f"Failed to process '{uploaded_file.name}': {exc}")
                error_occurred = True
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        if all_chunks and not error_occurred:
            progress_bar.progress(0.9, text="Building FAISS index...")
            try:
                vector_store = create_vector_store(all_chunks)
                pipeline = RAGPipeline(vector_store, k=top_k)

                st.session_state["pipeline"] = pipeline
                st.session_state["vector_store"] = vector_store
                st.session_state["indexed_pdf_set"] = current_fingerprint_set
                st.session_state["doc_catalog"] = catalog
                st.session_state["current_pdf_id"] = "_".join(f.name for f in uploaded_files)
                st.session_state["doc_stats"] = {
                    "name": ", ".join(d["name"] for d in catalog),
                    "pages": sum(d["pages"] for d in catalog),
                    "chunks": sum(d["chunks"] for d in catalog),
                }
                st.session_state["messages"] = []

                progress_bar.progress(1.0, text="Done!")
                total_chunks = sum(d["chunks"] for d in catalog)
                st.toast(
                    f"{len(catalog)} document(s) indexed — {total_chunks} total chunks.",
                    icon="✅",
                )
                st.rerun()

            except Exception as exc:
                st.error(f"Failed to build vector index: {exc}")
        elif not all_chunks:
            progress_bar.empty()
            st.error(
                "No usable text was extracted from any of the uploaded PDFs. "
                "Please check that your files contain selectable text."
            )
        else:
            progress_bar.empty()

elif not uploaded_files and st.session_state.get("uploader_active", False):
    # All files were removed by the user from the uploader
    st.session_state["uploader_active"] = False
    st.session_state["pipeline"] = None
    st.session_state["vector_store"] = None
    st.session_state["indexed_pdf_set"] = frozenset()
    st.session_state["doc_catalog"] = []
    st.session_state["current_pdf_id"] = None
    st.session_state["doc_stats"] = None
    st.session_state["messages"] = []
    st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">Docsy Reader</div>', unsafe_allow_html=True)
doc_count = len(st.session_state["doc_catalog"])
subtitle = (
    f"Searching across {doc_count} document(s). Answers include exact source and page citations."
    if doc_count > 0
    else "Ask questions grounded in your documents. Answers include exact page citations."
)
st.markdown(
    f'<div class="sub-header">{subtitle}</div>',
    unsafe_allow_html=True,
)

# Empty state banner
if st.session_state["pipeline"] is None:
    st.info(
        "👋 **Welcome to Docsy!** To get started:\n\n"
        "1. Upload one or more PDFs using the sidebar on the left.\n"
        "2. Wait for the documents to be indexed.\n"
        "3. Ask any question in the chat below — answers will cite the exact source and page."
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
    if prompt := st.chat_input("Ask a question across your documents..."):
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "sources": None,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents & formulating answer..."):
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


