from retrieval.retriever import create_retriever
from generation.chain import create_rag_chain


class RAGPipeline:

    def __init__(self, vector_store, k: int = 2):
        self.retriever = create_retriever(vector_store, k=k)
        self.rag_chain = create_rag_chain()

    def _extract_response_text(self, response) -> str:
        """Extract plain text from an LLM response object."""
        if isinstance(response.content, str):
            return response.content

        if isinstance(response.content, list):
            text_parts = []
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "\n".join(text_parts)

        return str(response.content)

    def ask(self, question: str) -> dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The user's question string.

        Returns:
            A dict with keys:
                "answer"  — The generated answer string.
                "sources" — A deduplicated list of {source, page} dicts.

        Raises:
            ValueError:  If the question is empty.
            RuntimeError: If the Gemini API returns a quota (429) or
                          availability (503) error, or any other API failure.
        """
        # --- Input validation ---
        if not question or not question.strip():
            raise ValueError("Question cannot be empty. Please enter a question.")

        # --- 1. Retrieve relevant documents ---
        retrieved_docs = self.retriever.invoke(question)

        # --- 2. Handle empty retrieval ---
        if not retrieved_docs:
            return {
                "answer": (
                    "I couldn't find relevant content in the document "
                    "to answer your question."
                ),
                "sources": [],
            }

        # --- 3. Build context ---
        context_parts = []
        for doc in retrieved_docs:
            page_number = doc.metadata.get("page", 0) + 1
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page_number}\n\n"
                f"Content:\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # --- 4. Generate answer ---
        try:
            response = self.rag_chain.invoke({
                "context": context,
                "question": question,
            })
        except Exception as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            err_msg = str(exc).lower()
            if code == 429 or "429" in err_msg or "rate limit" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                raise RuntimeError(
                    "LLM API rate limit / quota exhausted (HTTP 429). "
                    "You have reached the request limit. "
                    "Please wait for your quota to reset or check your API key at aistudio or groq."
                ) from exc
            if code in (500, 502, 503, 504) or "503" in err_msg or "unavailable" in err_msg or "5xx" in err_msg or "internal server" in err_msg:
                raise RuntimeError(
                    "LLM API is temporarily unavailable (HTTP 5xx). "
                    "This is a temporary server issue. Please try again in a few minutes."
                ) from exc
            if code and 400 <= code < 500:
                raise RuntimeError(
                    f"LLM API client error (HTTP {code}): {exc}"
                ) from exc
            raise RuntimeError(
                f"An unexpected error occurred while contacting the LLM API: {exc}"
            ) from exc

        # --- 5. Extract clean answer ---
        answer = self._extract_response_text(response)

        # --- 6. Build deduplicated source list ---
        seen: set[tuple] = set()
        sources = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            key = (source, page)
            if key not in seen:
                seen.add(key)
                sources.append({"source": source, "page": page})

        return {
            "answer": answer,
            "sources": sources,
        }