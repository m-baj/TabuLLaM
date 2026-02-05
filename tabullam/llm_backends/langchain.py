"""Langchain-based LLM backend."""

import time

from ..base.llm_backend import BaseLLMBackend, LLMResponse, LLMResponseDuration
from ..exceptions import LLMBackendError


def parse_langchain_response(response) -> LLMResponse:
    """
    Parse a Langchain response into LLMResponse format.

    Handles both simple string responses and complex responses
    with reasoning blocks (e.g., from Claude with extended thinking).

    Parameters
    ----------
    response : langchain response object
        Response from langchain model.invoke()

    Returns
    -------
    LLMResponse
        Parsed response with content, usage, duration, and reasoning
    """
    meta = getattr(response, 'response_metadata', {}) or {}
    to_ms = 1e-6

    duration = LLMResponseDuration(
        prompt_eval_duration=meta.get('prompt_eval_duration', 0) * to_ms,
        eval_duration=meta.get('eval_duration', 0) * to_ms,
        total_duration=meta.get('total_duration', 0) * to_ms
    )

    reasoning_data = []
    final_response = None

    if isinstance(response.content, str):
        final_response = response.content
    else:
        for block in response.content:
            if isinstance(block, dict):
                if block.get("type") == "reasoning":
                    for summary in block.get("summary", []):
                        reasoning_data.append(summary.get("text", ""))
                elif block.get("type") == "text":
                    final_response = block.get("text", "")
            elif hasattr(block, 'type'):
                if block.type == "reasoning":
                    for summary in getattr(block, 'summary', []):
                        reasoning_data.append(getattr(summary, 'text', ''))
                elif block.type == "text":
                    final_response = getattr(block, 'text', '')

    usage = {}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage_meta = response.usage_metadata
        if isinstance(usage_meta, dict):
            usage = {
                'input_tokens': usage_meta.get('input_tokens', 0),
                'output_tokens': usage_meta.get('output_tokens', 0),
                'total_tokens': usage_meta.get('total_tokens', 0),
            }
        else:
            usage = {
                'input_tokens': getattr(usage_meta, 'input_tokens', 0),
                'output_tokens': getattr(usage_meta, 'output_tokens', 0),
                'total_tokens': getattr(usage_meta, 'total_tokens', 0),
            }

    return LLMResponse(
        content=final_response or "",
        usage=usage,
        duration=duration,
        reasoning="\n".join(reasoning_data) if reasoning_data else None,
        raw_response=response
    )


class LangchainBackend(BaseLLMBackend):
    """
    Generic Langchain backend using init_chat_model.

    This backend uses Langchain's init_chat_model which automatically
    selects the appropriate chat model based on the model name format.

    Parameters
    ----------
    model : str
        Model identifier in Langchain format, e.g.:
        - 'openai:gpt-4o-mini'
        - 'google_genai:gemini-1.5-flash'
        - 'anthropic:claude-3-sonnet'

    **kwargs
        Additional arguments passed to init_chat_model

    Examples
    --------
    >>> backend = LangchainBackend('openai:gpt-4o-mini')
    >>> response = backend.generate('Hello, world!')
    """

    def __init__(self, model: str, **kwargs):
        self._model_name = model
        self.kwargs = kwargs
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of Langchain chat model."""
        if self._llm is None:
            try:
                from langchain.chat_models import init_chat_model
                self._llm = init_chat_model(self._model_name, **self.kwargs)
            except ImportError:
                raise LLMBackendError(
                    "langchain package is required for LangchainBackend. "
                    "Install it with: pip install langchain"
                )
        return self._llm

    def generate(self, prompt: str) -> LLMResponse:
        """Generate response using Langchain."""
        
        start_time = time.time()

        try:
            response = self.llm.invoke(prompt)
            elapsed_ms = (time.time() - start_time) * 1000  # Convert to ms

            llm_response = parse_langchain_response(response)

            # If duration is all zeros (provider doesn't return timing info),
            # use client-side measurement
            if llm_response.duration and llm_response.duration.total_duration == 0:
                from ..base.llm_backend import LLMResponseDuration
                llm_response.duration = LLMResponseDuration(
                    prompt_eval_duration=0,
                    eval_duration=0,
                    total_duration=elapsed_ms
                )

            return llm_response
        except Exception as e:
            raise LLMBackendError(f"Langchain error: {e}") from e

    @property
    def model_name(self) -> str:
        return self._model_name
