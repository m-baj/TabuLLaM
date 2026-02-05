"""
Tests for LLM backends.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys

from tabullam.base.llm_backend import (
    BaseLLMBackend,
    LLMResponse,
    LLMResponseDuration,
)
from tabullam.llm_backends import (
    LangchainBackend,
    OllamaBackend,
    GoogleBackend,
    create_llm_backend,
    parse_langchain_response,
)
from tabullam.exceptions import LLMBackendError


class TestLLMResponseDuration:
    """Tests for LLMResponseDuration dataclass."""

    def test_create_duration(self):
        """Test creating LLMResponseDuration instance."""
        duration = LLMResponseDuration(
            prompt_eval_duration=50.0,
            eval_duration=30.0,
            total_duration=80.0
        )
        assert duration.prompt_eval_duration == 50.0
        assert duration.eval_duration == 30.0
        assert duration.total_duration == 80.0


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating LLMResponse instance."""
        response = LLMResponse(
            content='test content',
            usage={'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15},
            duration=LLMResponseDuration(50.0, 30.0, 80.0),
            reasoning='reasoning text',
            raw_response={'raw': 'data'}
        )
        assert response.content == 'test content'
        assert response.usage['total_tokens'] == 15
        assert response.reasoning == 'reasoning text'

    def test_create_response_with_defaults(self):
        """Test creating LLMResponse with default values."""
        response = LLMResponse(content='test')
        assert response.content == 'test'
        assert response.usage == {}
        assert response.duration is None
        assert response.reasoning is None
        assert response.raw_response is None


class TestParseLangchainResponse:
    """Tests for parse_langchain_response function."""

    def test_parse_simple_string_response(self, mock_langchain_response):
        """Test parsing simple string response."""
        result = parse_langchain_response(mock_langchain_response)

        assert result.content == 'predicted_class'
        assert result.usage['input_tokens'] == 100
        assert result.usage['output_tokens'] == 10
        assert result.duration is not None

    def test_parse_response_with_reasoning(self, mock_langchain_response_with_reasoning):
        """Test parsing response with reasoning blocks."""
        result = parse_langchain_response(mock_langchain_response_with_reasoning)

        assert result.content == 'final_answer'
        assert result.reasoning is not None
        assert 'Step 1' in result.reasoning
        assert 'Step 2' in result.reasoning

    def test_parse_response_duration_conversion(self, mock_langchain_response):
        """Test that duration is converted from nanoseconds to milliseconds."""
        result = parse_langchain_response(mock_langchain_response)

        # 50000000 ns = 50 ms
        assert result.duration.prompt_eval_duration == 50.0
        assert result.duration.eval_duration == 30.0
        assert result.duration.total_duration == 80.0

    def test_parse_response_without_metadata(self):
        """Test parsing response without response_metadata."""
        response = Mock()
        response.content = 'test'
        response.response_metadata = None
        response.usage_metadata = {'input_tokens': 10, 'output_tokens': 5, 'total_tokens': 15}

        result = parse_langchain_response(response)
        assert result.content == 'test'


class TestLangchainBackend:
    """Tests for LangchainBackend."""

    def test_model_name_property(self):
        """Test model_name property."""
        backend = LangchainBackend('openai:gpt-4o-mini')
        assert backend.model_name == 'openai:gpt-4o-mini'

    def test_lazy_initialization(self):
        """Test that LLM is lazily initialized."""
        backend = LangchainBackend('openai:gpt-4o-mini')
        assert backend._llm is None

    def test_generate_with_mock_llm(self):
        """Test generate with mocked LLM."""
        backend = LangchainBackend('openai:gpt-4o-mini')

        # Manually set a mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = 'test response'
        mock_response.response_metadata = {}
        mock_response.usage_metadata = {}
        mock_llm.invoke.return_value = mock_response
        backend._llm = mock_llm

        result = backend.generate('test prompt')

        mock_llm.invoke.assert_called_once_with('test prompt')
        assert result.content == 'test response'


class TestOllamaBackend:
    """Tests for OllamaBackend."""

    def test_model_name_property(self):
        """Test model_name property includes ollama prefix."""
        backend = OllamaBackend('llama3.1:8b')
        assert backend.model_name == 'ollama:llama3.1:8b'

    def test_default_base_url(self):
        """Test default base URL."""
        backend = OllamaBackend('llama3.1:8b')
        assert backend.base_url == 'http://localhost:11434'

    def test_custom_base_url(self):
        """Test custom base URL."""
        backend = OllamaBackend('llama3.1:8b', base_url='http://custom:11434')
        assert backend.base_url == 'http://custom:11434'

    def test_lazy_initialization(self):
        """Test that LLM is lazily initialized."""
        backend = OllamaBackend('llama3.1:8b')
        assert backend._llm is None

    def test_generate_with_mock_llm(self):
        """Test generate with mocked LLM."""
        backend = OllamaBackend('llama3.1:8b')

        # Manually set a mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = 'test response'
        mock_response.response_metadata = {}
        mock_response.usage_metadata = {}
        mock_llm.invoke.return_value = mock_response
        backend._llm = mock_llm

        result = backend.generate('test prompt')

        mock_llm.invoke.assert_called_once_with('test prompt')
        assert result.content == 'test response'


class TestGoogleBackend:
    """Tests for GoogleBackend."""

    def test_model_name_property(self):
        """Test model_name property includes google prefix."""
        backend = GoogleBackend('gemini-1.5-flash')
        assert backend.model_name == 'google:gemini-1.5-flash'

    def test_default_model(self):
        """Test default model name."""
        backend = GoogleBackend()
        assert backend._model == 'gemini-1.5-flash'

    def test_lazy_initialization(self):
        """Test that LLM is lazily initialized."""
        backend = GoogleBackend('gemini-1.5-flash')
        assert backend._llm is None

    def test_generate_with_mock_llm(self):
        """Test generate with mocked LLM."""
        backend = GoogleBackend('gemini-1.5-flash')

        # Manually set a mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = 'test response'
        mock_response.response_metadata = {}
        mock_response.usage_metadata = {}
        mock_llm.invoke.return_value = mock_response
        backend._llm = mock_llm

        result = backend.generate('test prompt')

        mock_llm.invoke.assert_called_once_with('test prompt')
        assert result.content == 'test response'


class TestCreateLLMBackend:
    """Tests for create_llm_backend factory function."""

    def test_create_openai_backend(self):
        """Test creating OpenAI backend via LangChain."""
        backend = create_llm_backend('openai:gpt-4o-mini')
        assert isinstance(backend, LangchainBackend)
        assert backend._model_name == 'openai:gpt-4o-mini'

    def test_create_ollama_backend(self):
        """Test creating Ollama backend."""
        backend = create_llm_backend('ollama:llama3.1:8b')
        assert isinstance(backend, OllamaBackend)
        assert backend._model == 'llama3.1:8b'

    def test_create_google_backend(self):
        """Test creating Google backend."""
        backend = create_llm_backend('google:gemini-1.5-flash')
        assert isinstance(backend, GoogleBackend)
        assert backend._model == 'gemini-1.5-flash'

    def test_create_langchain_backend(self):
        """Test creating Langchain backend explicitly."""
        backend = create_llm_backend('langchain:anthropic:claude-3-sonnet')
        assert isinstance(backend, LangchainBackend)
        assert backend._model_name == 'anthropic:claude-3-sonnet'

    def test_create_unknown_provider_uses_langchain(self):
        """Test that unknown provider defaults to Langchain."""
        backend = create_llm_backend('anthropic:claude-3-sonnet')
        assert isinstance(backend, LangchainBackend)

    def test_invalid_spec_raises(self):
        """Test that invalid spec without colon raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_llm_backend('invalid_spec')
        assert 'Expected format' in str(exc_info.value)

    def test_kwargs_passed_to_backend(self):
        """Test that kwargs are passed to backend constructor."""
        backend = create_llm_backend('openai:gpt-4o-mini', temperature=0.5)
        assert backend.kwargs.get('temperature') == 0.5


class TestBackendErrorHandling:
    """Tests for error handling in backends."""

    def test_ollama_error_wrapped(self):
        """Test that Ollama errors are wrapped in LLMBackendError."""
        backend = OllamaBackend('llama3.1:8b')

        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Connection Error")
        backend._llm = mock_llm

        with pytest.raises(LLMBackendError) as exc_info:
            backend.generate('test')
        assert 'Ollama error' in str(exc_info.value)

    def test_google_error_wrapped(self):
        """Test that Google errors are wrapped in LLMBackendError."""
        backend = GoogleBackend('gemini-1.5-flash')

        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")
        backend._llm = mock_llm

        with pytest.raises(LLMBackendError) as exc_info:
            backend.generate('test')
        assert 'Google API error' in str(exc_info.value)

    def test_langchain_error_wrapped(self):
        """Test that Langchain errors are wrapped in LLMBackendError."""
        backend = LangchainBackend('openai:gpt-4o-mini')

        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Model Error")
        backend._llm = mock_llm

        with pytest.raises(LLMBackendError) as exc_info:
            backend.generate('test')
        assert 'Langchain error' in str(exc_info.value)
