"""
Tests for src/generation/prompt.py and src/generation/chain.py

No Gemini calls are made. We verify the prompt template structure and that
create_rag_chain wires prompt and LLM correctly (LLM is mocked).
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from generation.prompt import get_rag_prompt
from generation.chain import create_rag_chain


class TestGetRagPrompt:
    def test_returns_chat_prompt_template(self):
        prompt = get_rag_prompt()
        assert isinstance(prompt, ChatPromptTemplate)

    def test_prompt_has_context_variable(self):
        prompt = get_rag_prompt()
        input_vars = prompt.input_variables
        assert "context" in input_vars

    def test_prompt_has_question_variable(self):
        prompt = get_rag_prompt()
        input_vars = prompt.input_variables
        assert "question" in input_vars

    def test_prompt_contains_fallback_phrase(self):
        """The prompt must instruct the LLM to say it cannot find the answer."""
        prompt = get_rag_prompt()
        # Read the raw template text (not the repr) to avoid escaped-apostrophe issues
        template_text = prompt.messages[0].prompt.template
        assert "couldn't find" in template_text or "cannot find" in template_text

    def test_prompt_can_be_formatted(self):
        """Formatting with context and question must not raise."""
        prompt = get_rag_prompt()
        messages = prompt.format_messages(context="Some context.", question="Some question?")
        combined = " ".join(m.content for m in messages)
        assert "Some context." in combined
        assert "Some question?" in combined


class TestCreateRagChain:
    def test_returns_runnable_chain(self):
        """create_rag_chain must return a runnable (has .invoke)."""
        with patch("generation.chain.get_llm") as mock_get_llm:
            mock_get_llm.return_value = MagicMock()
            chain = create_rag_chain()
        assert hasattr(chain, "invoke")
