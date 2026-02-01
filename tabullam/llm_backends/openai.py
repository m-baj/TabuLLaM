"""OpenAI LLM backend."""

import time
from typing import Optional, Dict, Any

from ..base.llm_backend import BaseLLMBackend, LLMResponse, LLMResponseDuration
from ..exceptions import LLMBackendError


class OpenAIBackend(BaseLLMBackend):
    """
    OpenAI API backend using the official openai package.

    Parameters
    ----------
    model : str, default='gpt-4o-mini'
        OpenAI model name

    api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY env variable.

    **kwargs
        Additional arguments passed to ChatCompletion.create

    Examples
    --------
    >>> backend = OpenAIBackend('gpt-4o-mini')
    >>> response = backend.generate('Hello, world!')
    """

    def __init__(
        self,
        model: str = 'gpt-4o-mini',
        api_key: Optional[str] = None,
        **kwargs
    ):
        self._model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise LLMBackendError(
                    "openai package is required for OpenAIBackend. "
                    "Install it with: pip install openai"
                )
        return self._client

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using OpenAI API."""
        start = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                **self.kwargs
            )
        except Exception as e:
            raise LLMBackendError(f"OpenAI API error: {e}") from e

        duration_seconds = time.time() - start

        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            },
            duration=LLMResponseDuration(
                prompt_eval_duration=0,  # Not available from OpenAI API
                eval_duration=0,
                total_duration=duration_seconds * 1000  # Convert to ms
            ),
            raw_response=response
        )

    @property
    def model_name(self) -> str:
        return f"openai:{self._model}"
