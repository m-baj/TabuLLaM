"""Prompt builders for classification tasks."""

from typing import Optional

from ..base.prompt_builder import BasePromptBuilder, TaskMetadata
from .standard import StandardPromptBuilder
from .binary_confidence import BinaryConfidencePromptBuilder
from .multiclass_confidence import MulticlassConfidencePromptBuilder


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
        Type of prompt: 'standard', 'binary_confidence', or 'multiclass_confidence'
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
        'binary_confidence': BinaryConfidencePromptBuilder,
        'multiclass_confidence': MulticlassConfidencePromptBuilder,
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
    'BinaryConfidencePromptBuilder',
    'MulticlassConfidencePromptBuilder',
    'create_prompt_builder',
]
