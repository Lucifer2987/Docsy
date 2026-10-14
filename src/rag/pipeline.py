from google.genai.errors import ClientError, ServerError

from retrieval.retriever import create_retriever
from generation.chain import create_rag_chain


class RAGPipeline:

    def __init__(self, vector_store, k: int = 2):
        self.retriever = create_retriever(vector_store, k=k)
        self.rag_chain = create_rag_chain()

    def _extract_response_text(self, response) -> str:
        """Extract plain text from a Gemini response object."""
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
        except ClientError as exc:
            if exc.code == 429:
                raise RuntimeError(
                    "Gemini API quota exhausted (HTTP 429). "
                    "You have reached the free-tier daily limit for this model. "
                    "Please wait for your quota to reset (usually 24 hours) or "
                    "check your usage at https://aistudio.google.com."
                ) from exc
            raise RuntimeError(
                f"Gemini API client error (HTTP {exc.code}): {exc}"
            ) from exc
        except ServerError as exc:
            raise RuntimeError(
                "Gemini API is temporarily unavailable (HTTP 5xx). "
                "This is a temporary server issue. Please try again in a few minutes."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"An unexpected error occurred while contacting the Gemini API: {exc}"
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