"""Multiclass confidence prompt builder for JSON with per-class confidence scores."""

import textwrap

from ..base.prompt_builder import BasePromptBuilder


class MulticlassConfidencePromptBuilder(BasePromptBuilder):
    """Prompt builder expecting JSON with confidence score for each class."""

    def _default_instruction(self) -> str:
        return (
            "You are a classification assistant. Based on the provided features, "
            "estimate the confidence score for each possible class."
        )

    def _format_output_section(self) -> str:
        classes = self.task_metadata.class_labels
        classes_str = ", ".join([f'"{c}"' for c in classes])

        example_c1 = classes[0] if len(classes) > 0 else "Class_A"
        example_c2 = classes[1] if len(classes) > 1 else "Class_B"

        return textwrap.dedent(f"""
            *** INSTRUCTIONS ***
            Assign a confidence score from 0 (impossible) to 10 (absolute certainty) for EACH possible target class.
            Higher number = higher probability.
            DO NOT provide any explanations, DO NOT analyse, only return the JSON object as specified below.

            RETURN FORMAT (JSON ONLY):
            {{
              // The JSON keys must contain exactly these classes: {classes_str}
              "{example_c1}": INTEGER_0_TO_10,
              "{example_c2}": INTEGER_0_TO_10,
              ... (and so on for all classes)
            }}
        """)
