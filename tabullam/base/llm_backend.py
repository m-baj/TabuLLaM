"""Base classes for LLM backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LLMResponseDuration:
    """Duration metrics for LLM response."""
    prompt_eval_duration: float  # Time to process prompt (ms)
    eval_duration: float         # Time to generate response (ms)
    total_duration: float        # Total time (ms)


@dataclass
class LLMResponse:
    """Response from LLM inference."""
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    duration: Optional[LLMResponseDuration] = None
    reasoning: Optional[str] = None
    raw_response: Any = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Return total duration in seconds (for convenience)."""
        if self.duration is None:
            return None
        return self.duration.total_duration / 1000.0


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    All LLM backends must implement the `generate` method.
    """

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """
        Generate a response for the given prompt.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM

        Returns
        -------
        LLMResponse
            Response containing content, usage stats, and metadata
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass
