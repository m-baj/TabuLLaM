"""Base class for prompt builders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ..utils.serialization import key_value_serialize


@dataclass
class TaskMetadata:
    """
    Metadata about the classification task.

    Parameters
    ----------
    feature_names : list of str
        Names of input features
    class_labels : list of str
        Possible class labels
    target_name : str
        Name of the target column
    task_description : str, optional
        Human-readable description of the task
    """
    feature_names: List[str]
    class_labels: List[str]
    target_name: str
    task_description: Optional[str] = None


class BasePromptBuilder(ABC):
    """Abstract base class for prompt builders."""

    def __init__(
        self,
        task_metadata: TaskMetadata,
        instruction: Optional[str] = None
    ):
        self.task_metadata = task_metadata
        self.instruction = instruction or self._default_instruction()

    @abstractmethod
    def _default_instruction(self) -> str:
        """Return default instruction for this prompt type."""
        pass

    @abstractmethod
    def _format_output_section(self) -> str:
        """Define expected output format."""
        pass

    def build(
        self,
        query_features: Dict[str, Any],
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build the complete prompt.

        Parameters
        ----------
        query_features : dict
            Features of the sample to classify
        examples : list of dict, optional
            Few-shot examples, each with 'features' and 'label' keys

        Returns
        -------
        str
            Complete prompt
        """
        sections = [
            self._format_instruction_section(),
            self._format_metadata_section(),
        ]

        if examples:
            sections.append(self._format_examples_section(examples))

        sections.append(self._format_query_section(query_features))
        sections.append(self._format_output_section())

        return "\n\n".join(sections)

    def _format_instruction_section(self) -> str:
        """Format the instruction section."""
        return f"## Instruction\n{self.instruction}"

    def _format_metadata_section(self) -> str:
        """Format the task metadata section."""
        lines = ["## Task Information"]

        if self.task_metadata.task_description:
            lines.append(f"Description: {self.task_metadata.task_description}")

        lines.append(f"Features: {', '.join(self.task_metadata.feature_names)}")
        lines.append(f"Possible classes: {', '.join(self.task_metadata.class_labels)}")

        return "\n".join(lines)

    def _format_examples_section(self, examples: List[Dict[str, Any]]) -> str:
        """Format the few-shot examples section."""
        lines = ["## Examples"]

        for i, ex in enumerate(examples, 1):
            label_dict = {self.task_metadata.target_name: ex['label']}
            serialized = key_value_serialize(ex['features'], label_dict)
            lines.append(f"{i}. {serialized}")

        return "\n".join(lines)

    def _format_query_section(self, query_features: Dict[str, Any]) -> str:
        """Format the query section."""
        serialized = key_value_serialize(query_features)
        return f"## Query\nClassify the following instance:\n{serialized}"
