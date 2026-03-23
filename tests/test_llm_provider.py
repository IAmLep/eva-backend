"""
Tests for the LLM provider abstraction layer.
"""

import pytest
from unittest.mock import patch

from llm_service import (
    LLMProvider,
    GeminiProvider,
    GeminiService,
    get_llm_provider,
    _PROVIDER_REGISTRY,
)
from exceptions import ConfigurationError


class TestLLMProviderAbstraction:
    """Tests for the abstract LLMProvider and provider registry."""

    def test_gemini_provider_is_llm_provider(self):
        """GeminiProvider should be a subclass of LLMProvider."""
        assert issubclass(GeminiProvider, LLMProvider)

    def test_gemini_service_alias(self):
        """GeminiService should be an alias for GeminiProvider (backward compat)."""
        assert GeminiService is GeminiProvider

    def test_provider_registry_contains_gemini(self):
        """The provider registry should contain the gemini provider."""
        assert "gemini" in _PROVIDER_REGISTRY
        assert _PROVIDER_REGISTRY["gemini"] is GeminiProvider

    def test_get_llm_provider_returns_gemini_by_default(self):
        """get_llm_provider() should return a GeminiProvider by default."""
        provider = get_llm_provider()
        assert isinstance(provider, GeminiProvider)
        assert isinstance(provider, LLMProvider)

    def test_get_llm_provider_explicit_gemini(self):
        """get_llm_provider('gemini') should return a GeminiProvider."""
        provider = get_llm_provider("gemini")
        assert isinstance(provider, GeminiProvider)

    def test_get_llm_provider_unknown_raises(self):
        """get_llm_provider('unknown') should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            get_llm_provider("unknown_provider")

    def test_gemini_provider_has_provider_name(self):
        """GeminiProvider should report its name."""
        provider = GeminiProvider()
        assert provider.provider_name == "Gemini"

    def test_gemini_provider_mock_mode_without_key(self):
        """Without GEMINI_API_KEY, provider should be in mock mode."""
        provider = GeminiProvider()
        # In the test environment there's no API key, so mock mode
        assert provider.use_mock is True
        assert provider.is_available is False

    def test_llm_provider_abstract_methods(self):
        """LLMProvider should not be directly instantiatable."""
        with pytest.raises(TypeError):
            LLMProvider()


class TestStreamConversationWithHistory:
    """Tests for the stream_conversation_with_history method."""

    def test_gemini_provider_has_stream_with_history_method(self):
        """GeminiProvider must implement stream_conversation_with_history."""
        provider = GeminiProvider()
        assert hasattr(provider, "stream_conversation_with_history")
        assert callable(provider.stream_conversation_with_history)

    @pytest.mark.asyncio
    async def test_stream_with_history_mock_yields_chunks(self):
        """In mock mode, stream_conversation_with_history should yield text chunks."""
        provider = GeminiProvider()
        assert provider.use_mock is True  # no API key in test env

        history = [
            {"role": "user", "parts": [{"text": "Hello EVA"}]},
        ]
        chunks = []
        async for chunk in provider.stream_conversation_with_history(history=history):
            chunks.append(chunk)

        assert len(chunks) > 0
        # At least one chunk should contain text
        text_chunks = [c for c in chunks if "text" in c]
        assert len(text_chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_with_history_empty_history_does_not_crash(self):
        """stream_conversation_with_history should handle an empty history list gracefully."""
        provider = GeminiProvider()
        assert provider.use_mock is True

        chunks = []
        async for chunk in provider.stream_conversation_with_history(history=[]):
            chunks.append(chunk)

        assert len(chunks) > 0
