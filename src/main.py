from ingestion.loader import load_pdf
from ingestion.splitter import split_documents

from vectorstore.store import create_vector_store
from retrieval.retriever import create_retriever

from generation.chain import create_rag_chain


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
# 4. Create Retriever
# -------------------------

retriever = create_retriever(vector_store, k=2)


# -------------------------
# 5. Create LLM Chain
# -------------------------

rag_chain = create_rag_chain()


# -------------------------
# 6. Ask Question
# -------------------------

question = "What data preprocessing techniques were implemented?"


# -------------------------
# 7. Retrieve relevant documents
# -------------------------

retrieved_docs = retriever.invoke(question)


# -------------------------
# 8. Build context with metadata
# -------------------------

context_parts = []

for doc in retrieved_docs:

    page_number = doc.metadata.get("page", 0) + 1
    source = doc.metadata.get("source", "Unknown")

    context_parts.append(
        f"""
Source: {source}
Page: {page_number}

Content:
{doc.page_content}
"""
    )

context = "\n\n".join(context_parts)


# -------------------------
# 9. Send context + question to LLM
# -------------------------

response = rag_chain.invoke({
    "context": context,
    "question": question
})


# -------------------------
# 10. Extract clean answer
# -------------------------

answer = extract_response_text(response)


# -------------------------
# 11. Print answer
# -------------------------

print("\n==============================")
print("ANSWER")
print("==============================")

print(answer)