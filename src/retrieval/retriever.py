def create_retriever(vector_store, k=2):

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever