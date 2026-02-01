"""Prompt builders for classification tasks."""

from typing import Optional

from ..base.prompt_builder import BasePromptBuilder, TaskMetadata
from .standard import StandardPromptBuilder
from .confidence import ConfidencePromptBuilder
from .probabilities import ProbabilitiesPromptBuilder


def create_prompt_builder(
    prompt_type: str,
    task_metadata: TaskMetadata,
    instruction: Optional[str] = None
) -> BasePromptBuilder:
    """
    Factory function to create prompt builder.

    Parameters
    ----------
    prompt_type : str
        Type of prompt: 'standard', 'confidence', or 'probabilities'
    task_metadata : TaskMetadata
        Metadata about the classification task
    instruction : str, optional
        Custom instruction. If None, uses default for the prompt type.

    Returns
    -------
    BasePromptBuilder
        Configured prompt builder
    """
    builders = {
        'standard': StandardPromptBuilder,
        'confidence': ConfidencePromptBuilder,
        'probabilities': ProbabilitiesPromptBuilder,
    }

    if prompt_type not in builders:
        raise ValueError(
            f"Unknown prompt_type: '{prompt_type}'. "
            f"Available: {list(builders.keys())}"
        )

    return builders[prompt_type](task_metadata, instruction)


__all__ = [
    'BasePromptBuilder',
    'TaskMetadata',
    'StandardPromptBuilder',
    'ConfidencePromptBuilder',
    'ProbabilitiesPromptBuilder',
    'create_prompt_builder',
]
