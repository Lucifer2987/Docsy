"""
FAISS vector store creation, persistence, and staleness detection.

Public API
----------
create_vector_store(chunks)          Build an in-memory FAISS store from chunks.
save_vector_store(vs, index_dir)     Persist the store to disk.
load_vector_store(index_dir)         Load a previously saved store from disk.
write_index_meta(index_dir, pdf_path) Write a sidecar JSON with PDF mtime+size.
is_index_fresh(index_dir, pdf_path)  Return True if the saved index matches the PDF.
"""
import json
import os

from langchain_community.vectorstores import FAISS

from embeddings.embedder import get_embedding_model

# Filenames written inside index_dir
_FAISS_FOLDER = "faiss_index"   # subfolder holding index.faiss + index.pkl
_META_FILE    = "meta.json"      # sidecar with source-file provenance


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def create_vector_store(chunks):
    """Embed *chunks* with HuggingFace and return an in-memory FAISS store."""
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_vector_store(vector_store, index_dir: str) -> None:
    """
    Save *vector_store* to *index_dir/<_FAISS_FOLDER>/*.

    FAISS.save_local writes two files:
        index.faiss  — the raw binary index
        index.pkl    — the document store (metadata, page_content)

    The directory is created if it does not exist.
    """
    os.makedirs(index_dir, exist_ok=True)
    faiss_path = os.path.join(index_dir, _FAISS_FOLDER)
    vector_store.save_local(faiss_path)


def load_vector_store(index_dir: str):
    """
    Load and return a FAISS vector store from *index_dir/<_FAISS_FOLDER>/*.

    ``allow_dangerous_deserialization=True`` is required by LangChain when
    loading a pickle-backed store. This is safe here because we only ever
    load indexes that *we* wrote to a local, trusted directory — never
    arbitrary user-supplied data.

    Raises:
        FileNotFoundError: If the expected FAISS files are not present.
    """
    faiss_path = os.path.join(index_dir, _FAISS_FOLDER)
    index_file = os.path.join(faiss_path, "index.faiss")

    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"No saved FAISS index found at '{faiss_path}'. "
            "Run the pipeline once to build and save the index."
        )

    embeddings = get_embedding_model()
    return FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

def write_index_meta(index_dir: str, pdf_path: str) -> None:
    """
    Write a JSON sidecar recording the source PDF's mtime and size.

    This is used by ``is_index_fresh`` to decide whether a saved index
    still matches the current PDF without re-embedding anything.
    """
    os.makedirs(index_dir, exist_ok=True)
    stat = os.stat(pdf_path)
    meta = {
        "source_path": os.path.abspath(pdf_path),
        "source_mtime": stat.st_mtime,
        "source_size":  stat.st_size,
    }
    meta_path = os.path.join(index_dir, _META_FILE)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def is_index_fresh(index_dir: str, pdf_path: str) -> bool:
    """
    Return True if a saved FAISS index exists **and** matches *pdf_path*.

    Matching is based on absolute path, mtime, and file size.  If any of
    those differ — or the meta file / FAISS files are absent — returns False,
    signalling that a rebuild is needed.
    """
    meta_path  = os.path.join(index_dir, _META_FILE)
    faiss_path = os.path.join(index_dir, _FAISS_FOLDER, "index.faiss")

    # Both the sidecar and the FAISS binary must exist
    if not os.path.exists(meta_path) or not os.path.exists(faiss_path):
        return False

    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    stat = os.stat(pdf_path)
    return (
        meta.get("source_path")  == os.path.abspath(pdf_path)
        and meta.get("source_mtime") == stat.st_mtime
        and meta.get("source_size")  == stat.st_size
    )