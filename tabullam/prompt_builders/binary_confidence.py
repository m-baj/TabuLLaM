"""Binary confidence prompt builder for JSON with prediction and confidence score."""

import textwrap

from ..base.prompt_builder import BasePromptBuilder


class BinaryConfidencePromptBuilder(BasePromptBuilder):
    """Prompt builder expecting JSON with prediction and confidence for binary classification."""

    def _default_instruction(self) -> str:
        return (
            "You are a classification assistant. Based on the provided features, "
            "predict the most likely class and rate your confidence in this prediction."
        )

    def _format_output_section(self) -> str:
        return textwrap.dedent("""
            *** INSTRUCTIONS ***
            1. Predict the class label.
            2. Rate your confidence on a scale from 0 (guessing) to 10 (absolute certainty).
            3. DO NOT provide any explanations, DO NOT analyse, only return the JSON object as specified below.

            RETURN FORMAT (JSON ONLY):
            {
              "prediction": "PREDICTED_CLASS_LABEL",
              "confidence_score": INTEGER_0_TO_10
            }
        """)
