"""
Tests for custom exceptions.
"""

import pytest

from tabullam.exceptions import (
    TabularLLMError,
    NotFittedError,
    EmbeddingError,
    LLMBackendError,
    ParseError,
    ConfigurationError,
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all exceptions inherit from TabularLLMError."""
        assert issubclass(NotFittedError, TabularLLMError)
        assert issubclass(EmbeddingError, TabularLLMError)
        assert issubclass(LLMBackendError, TabularLLMError)
        assert issubclass(ParseError, TabularLLMError)
        assert issubclass(ConfigurationError, TabularLLMError)

    def test_base_exception_inherits_from_exception(self):
        """Test that TabularLLMError inherits from Exception."""
        assert issubclass(TabularLLMError, Exception)


class TestExceptionMessages:
    """Tests for exception messages."""

    def test_tabular_llm_error_message(self):
        """Test TabularLLMError can be raised with message."""
        with pytest.raises(TabularLLMError) as exc_info:
            raise TabularLLMError("Test error message")
        assert "Test error message" in str(exc_info.value)

    def test_not_fitted_error_message(self):
        """Test NotFittedError can be raised with message."""
        with pytest.raises(NotFittedError) as exc_info:
            raise NotFittedError("Classifier not fitted")
        assert "Classifier not fitted" in str(exc_info.value)

    def test_embedding_error_message(self):
        """Test EmbeddingError can be raised with message."""
        with pytest.raises(EmbeddingError) as exc_info:
            raise EmbeddingError("Failed to compute embeddings")
        assert "Failed to compute embeddings" in str(exc_info.value)

    def test_llm_backend_error_message(self):
        """Test LLMBackendError can be raised with message."""
        with pytest.raises(LLMBackendError) as exc_info:
            raise LLMBackendError("API call failed")
        assert "API call failed" in str(exc_info.value)

    def test_parse_error_message(self):
        """Test ParseError can be raised with message."""
        with pytest.raises(ParseError) as exc_info:
            raise ParseError("Failed to parse response")
        assert "Failed to parse response" in str(exc_info.value)

    def test_configuration_error_message(self):
        """Test ConfigurationError can be raised with message."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Invalid configuration")
        assert "Invalid configuration" in str(exc_info.value)


class TestExceptionCatching:
    """Tests for catching exceptions at different levels."""

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        try:
            raise EmbeddingError("Embedding failed")
        except EmbeddingError as e:
            assert "Embedding failed" in str(e)

    def test_catch_base_exception(self):
        """Test catching all library exceptions via base class."""
        exceptions_to_test = [
            NotFittedError("test"),
            EmbeddingError("test"),
            LLMBackendError("test"),
            ParseError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions_to_test:
            try:
                raise exc
            except TabularLLMError as e:
                # All should be catchable as TabularLLMError
                assert str(e) == "test"

    def test_not_caught_by_wrong_type(self):
        """Test that specific exceptions aren't caught by wrong type."""
        with pytest.raises(EmbeddingError):
            try:
                raise EmbeddingError("test")
            except LLMBackendError:
                pytest.fail("Should not catch EmbeddingError as LLMBackendError")
