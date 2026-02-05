"""Base class for prompt builders."""

import textwrap
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
        parts = []
        parts.append(self.instruction)
        parts.append(self._format_metadata_section())

        if examples:
            parts.append(self._format_examples_section(examples))

        parts.append(self._format_query_section(query_features))
        parts.append(self._format_output_section())

        return "\n".join(parts)

    def _format_metadata_section(self) -> str:
        """Format the task metadata section."""
        return textwrap.dedent(f"""
            Task description: {self.task_metadata.task_description}
            Features: {', '.join(self.task_metadata.feature_names)}
            Target label classes: {', '.join(self.task_metadata.class_labels)}
        """)

    def _format_examples_section(self, examples: List[Dict[str, Any]]) -> str:
        """Format the few-shot examples section."""
        section = ["Labeled instances:"]
        for ex in examples:
            label_dict = {self.task_metadata.target_name: ex['label']}
            serialized = key_value_serialize(ex['features'], label_dict)
            section.append(serialized)
        return "\n".join(section)

    def _format_query_section(self, query_features: Dict[str, Any]) -> str:
        """Format the query section."""
        serialized = key_value_serialize(query_features)
        return textwrap.dedent(f"""
            Now use the provided metadata and instances to infer by analogy about the label of this new instance:
            {serialized}
        """)
