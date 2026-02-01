"""Google Gemini LLM backend."""

from typing import Optional, Dict, Any

from ..base.llm_backend import BaseLLMBackend, LLMResponse, LLMResponseDuration
from ..exceptions import LLMBackendError
from .langchain import parse_langchain_response


class GoogleBackend(BaseLLMBackend):
    """
    Google Gemini backend using langchain-google-genai.

    Parameters
    ----------
    model : str, default='gemini-1.5-flash'
        Google Gemini model name

    api_key : str, optional
        Google API key. If None, uses GOOGLE_API_KEY env variable.

    **kwargs
        Additional arguments passed to ChatGoogleGenerativeAI

    Examples
    --------
    >>> backend = GoogleBackend('gemini-1.5-flash')
    >>> response = backend.generate('Hello, world!')
    """

    def __init__(
        self,
        model: str = 'gemini-1.5-flash',
        api_key: Optional[str] = None,
        **kwargs
    ):
        self._model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of Google Gemini model."""
        if self._llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model=self._model,
                    google_api_key=self.api_key,
                    **self.kwargs
                )
            except ImportError:
                raise LLMBackendError(
                    "langchain-google-genai package is required for GoogleBackend. "
                    "Install it with: pip install langchain-google-genai"
                )
        return self._llm

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using Google Gemini."""
        try:
            response = self.llm.invoke(prompt)
            return parse_langchain_response(response)
        except Exception as e:
            raise LLMBackendError(f"Google API error: {e}") from e

    @property
    def model_name(self) -> str:
        return f"google:{self._model}"
