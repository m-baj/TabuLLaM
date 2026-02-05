"""Ollama LLM backend."""

from ..base.llm_backend import BaseLLMBackend, LLMResponse
from ..exceptions import LLMBackendError
from .langchain import parse_langchain_response


class OllamaBackend(BaseLLMBackend):
    """
    Ollama local model backend.

    Uses langchain-ollama's ChatOllama for local model inference.

    Parameters
    ----------
    model : str, default='llama3.1:8b'
        Ollama model name

    base_url : str, default='http://localhost:11434'
        Ollama API base URL

    **kwargs
        Additional arguments passed to ChatOllama

    Examples
    --------
    >>> backend = OllamaBackend('llama3.1:8b')
    >>> response = backend.generate('Hello, world!')
    """

    def __init__(
        self,
        model: str = 'llama3.1:8b',
        base_url: str = 'http://localhost:11434',
        **kwargs
    ):
        self._model = model
        self.base_url = base_url
        self.kwargs = kwargs
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of ChatOllama."""
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=self._model,
                    base_url=self.base_url,
                    **self.kwargs
                )
            except ImportError:
                raise LLMBackendError(
                    "langchain-ollama package is required for OllamaBackend. "
                    "Install it with: pip install langchain-ollama"
                )
        return self._llm

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using Ollama."""
        try:
            response = self.llm.invoke(prompt)
            return parse_langchain_response(response)
        except Exception as e:
            raise LLMBackendError(f"Ollama error: {e}") from e

    @property
    def model_name(self) -> str:
        return f"ollama:{self._model}"
