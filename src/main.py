from ingestion.loader import load_pdf
from ingestion.splitter import split_documents

from vectorstore.store import create_vector_store

from rag.pipeline import RAGPipeline


# -------------------------
# Helper: Extract LLM text
# -------------------------

def extract_response_text(response):
    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):
        text_parts = []

        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        return "\n".join(text_parts)

    return str(response.content)


# -------------------------
# 1. Load PDF
# -------------------------

documents = load_pdf("data/pdfs/sample.pdf")

print(f"Total pages: {len(documents)}")


# -------------------------
# 2. Split PDF
# -------------------------

chunks = split_documents(documents)

print(f"Total chunks: {len(chunks)}")


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

# -------------------------
# 6. Ask RAG Pipeline
# -------------------------

result = pipeline.ask(question)

print("\n==============================")
print("ANSWER")
print("==============================")

print(result["answer"])

print("\n==============================")
print("SOURCES")
print("==============================")

for source in result["sources"]:
    print(f"- {source['source']} | Page {source['page']}")