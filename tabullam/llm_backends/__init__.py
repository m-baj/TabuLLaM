"""LLM backend implementations."""

from ..base.llm_backend import BaseLLMBackend, LLMResponse, LLMResponseDuration
from .langchain import LangchainBackend, parse_langchain_response
from .ollama import OllamaBackend
from .google import GoogleBackend


def create_llm_backend(llm_spec: str, **kwargs) -> BaseLLMBackend:
    """
    Factory function to create LLM backend from specification string.

    Parameters
    ----------
    llm_spec : str
        Format: 'provider:model_name'
        Examples:
        - 'openai:gpt-4o-mini' -> LangchainBackend (via OpenAI)
        - 'ollama:llama3.1:8b' -> OllamaBackend
        - 'google:gemini-1.5-flash' -> GoogleBackend
        - 'langchain:anthropic:claude-3-sonnet' -> LangchainBackend (generic)

    **kwargs
        Additional arguments passed to backend constructor

    Returns
    -------
    BaseLLMBackend
        Configured LLM backend

    Examples
    --------
    >>> backend = create_llm_backend('openai:gpt-4o-mini')  # Uses LangChain
    >>> response = backend.generate('Hello, world!')

    >>> backend = create_llm_backend('ollama:llama3.1:8b')
    >>> response = backend.generate('Hello, world!')

    >>> backend = create_llm_backend('langchain:anthropic:claude-3-sonnet')
    >>> response = backend.generate('Hello, world!')
    """
    if ':' not in llm_spec:
        raise ValueError(
            f"Invalid llm format: '{llm_spec}'. Expected format: 'provider:model_name'"
        )

    provider, model = llm_spec.split(':', 1)
    provider = provider.lower()

    if provider == 'openai':
        # Use LangChain for OpenAI models
        return LangchainBackend(model=f'openai:{model}', **kwargs)
    elif provider == 'ollama':
        return OllamaBackend(model=model, **kwargs)
    elif provider == 'google':
        return GoogleBackend(model=model, **kwargs)
    elif provider == 'langchain':
        return LangchainBackend(model=model, **kwargs)
    else:
        return LangchainBackend(model=llm_spec, **kwargs)


__all__ = [
    'BaseLLMBackend',
    'LLMResponse',
    'LLMResponseDuration',
    'LangchainBackend',
    'OllamaBackend',
    'GoogleBackend',
    'parse_langchain_response',
    'create_llm_backend',
]
