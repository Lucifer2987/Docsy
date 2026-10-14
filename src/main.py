import sys

from ingestion.loader import load_pdf
from ingestion.splitter import split_documents

from vectorstore.store import create_vector_store

from rag.pipeline import RAGPipeline


# -------------------------
# 1. Load PDF
# -------------------------

try:
    documents = load_pdf("data/pdfs/sample.pdf")
except FileNotFoundError as exc:
    print(f"Error: {exc}")
    sys.exit(1)
except ValueError as exc:
    print(f"Error: {exc}")
    sys.exit(1)

print(f"Total pages: {len(documents)}")


# -------------------------
# 2. Split PDF
# -------------------------

chunks = split_documents(documents)

print(f"Total chunks: {len(chunks)}")

if not chunks:
    print("Error: No text chunks were produced. The PDF may contain only images.")
    sys.exit(1)


# -------------------------
# 3. Create Vector Store
# -------------------------

vector_store = create_vector_store(chunks)

print("Vector store created!")


# -------------------------
# 4. Create RAG Pipeline
# -------------------------

pipeline = RAGPipeline(vector_store, k=2)


# -------------------------
# 5. Ask Question
# -------------------------

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