"""LLM backend implementations."""

from ..base.llm_backend import BaseLLMBackend, LLMResponse, LLMResponseDuration
from .langchain import LangchainBackend, parse_langchain_response
from .ollama import OllamaBackend
from .openai import OpenAIBackend
from .google import GoogleBackend


def create_llm_backend(llm_spec: str, **kwargs) -> BaseLLMBackend:
    """
    Factory function to create LLM backend from specification string.

    Parameters
    ----------
    llm_spec : str
        Format: 'provider:model_name'
        Examples:
        - 'openai:gpt-4o-mini' -> OpenAIBackend
        - 'ollama:llama3.1:8b' -> OllamaBackend
        - 'google:gemini-1.5-flash' -> GoogleBackend
        - 'langchain:openai:gpt-4o' -> LangchainBackend (generic)

    **kwargs
        Additional arguments passed to backend constructor

    Returns
    -------
    BaseLLMBackend
        Configured LLM backend

    Examples
    --------
    >>> backend = create_llm_backend('openai:gpt-4o-mini')
    >>> response = backend.generate('Hello, world!')

    >>> backend = create_llm_backend('ollama:llama3.1:8b')
    >>> response = backend.generate('Hello, world!')

    >>> # Use langchain for any supported model
    >>> backend = create_llm_backend('langchain:anthropic:claude-3-sonnet')
    """
    if ':' not in llm_spec:
        raise ValueError(
            f"Invalid llm format: '{llm_spec}'. Expected format: 'provider:model_name'"
        )

    provider, model = llm_spec.split(':', 1)
    provider = provider.lower()

    if provider == 'openai':
        return OpenAIBackend(model=model, **kwargs)
    elif provider == 'ollama':
        return OllamaBackend(model=model, **kwargs)
    elif provider == 'google':
        return GoogleBackend(model=model, **kwargs)
    elif provider == 'langchain':
        # For langchain, the model is the full langchain model spec
        return LangchainBackend(model=model, **kwargs)
    else:
        # Default to langchain for unknown providers
        # This allows using any langchain-supported model
        return LangchainBackend(model=llm_spec, **kwargs)


__all__ = [
    # Base classes
    'BaseLLMBackend',
    'LLMResponse',
    'LLMResponseDuration',
    # Implementations
    'LangchainBackend',
    'OllamaBackend',
    'OpenAIBackend',
    'GoogleBackend',
    # Utilities
    'parse_langchain_response',
    'create_llm_backend',
]
