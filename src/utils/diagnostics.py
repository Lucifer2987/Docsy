"""
Diagnostic utilities for Docsy.

These functions are for local development and debugging only.
Do not call them from the main application or in automated tests.
"""


def run_retrieval_diagnostic(vector_store, query: str, k: int = 3) -> None:
    """
    Run a FAISS similarity search and print scores + content previews.

    Args:
        vector_store: A FAISS vector store instance.
        query:        The query string to search with.
        k:            Number of top results to retrieve.

    Note:
        FAISS returns L2 (Euclidean) distances by default.
        Lower scores mean the chunk is closer (more similar) to the query.
    """
    results = vector_store.similarity_search_with_score(query, k=k)

    print("\n==============================")
    print("RETRIEVAL DIAGNOSTIC")
    print(f"Query : {query!r}")
    print(f"Top-{k} results (lower L2 score = more similar)")
    print("==============================")

    for i, (doc, score) in enumerate(results, start=1):
        page = doc.metadata.get("page", 0) + 1
        source = doc.metadata.get("source", "Unknown")
        preview = doc.page_content[:300].replace("\n", " ")

        print(f"\n[{i}] Score: {score:.4f} | Source: {source} | Page: {page}")
        print(f"    {preview}")

    print("==============================\n")
