from langchain_community.vectorstores import FAISS

from embeddings.embedder import get_embedding_model


def create_vector_store(chunks):

    embeddings = get_embedding_model()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store