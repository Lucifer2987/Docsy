import sys

from ingestion.loader import load_pdf
from ingestion.splitter import split_documents

from vectorstore.store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    write_index_meta,
    is_index_fresh,
)
from rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PDF_PATH   = "data/pdfs/sample.pdf"
INDEX_DIR  = "data/vectorstore"

# ---------------------------------------------------------------------------
# 1. Load PDF
# ---------------------------------------------------------------------------

try:
    documents = load_pdf(PDF_PATH)
except FileNotFoundError as exc:
    print(f"Error: {exc}")
    sys.exit(1)
except ValueError as exc:
    print(f"Error: {exc}")
    sys.exit(1)

print(f"Loaded {len(documents)} page(s) from '{PDF_PATH}'.")

# ---------------------------------------------------------------------------
# 2. Vector store — load cached or rebuild
# ---------------------------------------------------------------------------

if is_index_fresh(INDEX_DIR, PDF_PATH):
    print("Loading existing FAISS index from disk (PDF unchanged)...")
    try:
        vector_store = load_vector_store(INDEX_DIR)
        print("Index loaded.")
    except FileNotFoundError as exc:
        print(f"Warning: {exc}. Rebuilding index...")
        vector_store = None
else:
    vector_store = None

if vector_store is None:
    print("Building FAISS index (this may take a few seconds)...")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")

    if not chunks:
        print("Error: No text chunks were produced. The PDF may contain only images.")
        sys.exit(1)

    vector_store = create_vector_store(chunks)
    save_vector_store(vector_store, INDEX_DIR)
    write_index_meta(INDEX_DIR, PDF_PATH)
    print(f"Index saved to '{INDEX_DIR}'.")

# ---------------------------------------------------------------------------
# 3. RAG Pipeline
# ---------------------------------------------------------------------------

pipeline = RAGPipeline(vector_store, k=2)

# ---------------------------------------------------------------------------
# 4. Ask a question
# ---------------------------------------------------------------------------

question = "How were missing values handled?"

try:
    result = pipeline.ask(question)
except ValueError as exc:
    print(f"Input error: {exc}")
    sys.exit(1)
except RuntimeError as exc:
    print(f"API error: {exc}")
    sys.exit(1)

print("\n==============================")
print("ANSWER")
print("==============================")
print(result["answer"])

print("\n==============================")
print("SOURCES")
print("==============================")
for source in result["sources"]:
    print(f"- {source['source']} | Page {source['page']}")