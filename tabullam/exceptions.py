"""Custom exceptions for the TabularLLM library."""


class TabularLLMError(Exception):
    """Base exception for TabularLLM library."""
    pass


class NotFittedError(TabularLLMError):
    """Exception raised when predict is called before fit."""
    pass


class EmbeddingError(TabularLLMError):
    """Exception raised when embedding computation fails."""
    pass


class LLMBackendError(TabularLLMError):
    """Exception raised when LLM backend fails."""
    pass


class ParseError(TabularLLMError):
    """Exception raised when response parsing fails."""
    pass


class ConfigurationError(TabularLLMError):
    """Exception raised when configuration is invalid."""
    pass
