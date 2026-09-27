from retrieval.retriever import create_retriever
from generation.chain import create_rag_chain


class RAGPipeline:

    def __init__(self, vector_store, k=2):
        self.retriever = create_retriever(vector_store, k=k)
        self.rag_chain = create_rag_chain()

    def _extract_response_text(self, response):
        if isinstance(response.content, str):
            return response.content

        if isinstance(response.content, list):
            text_parts = []

            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            return "\n".join(text_parts)

        return str(response.content)

    def ask(self, question):

        # 1. Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(question)

        # 2. Build context
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

        # 3. Generate answer
        response = self.rag_chain.invoke({
            "context": context,
            "question": question
        })

        # 4. Extract clean answer
        answer = self._extract_response_text(response)

        # 5. Extract sources
        sources = []

        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0) + 1

            sources.append({
                "source": source,
                "page": page
            })

        return {
            "answer": answer,
            "sources": sources
        }