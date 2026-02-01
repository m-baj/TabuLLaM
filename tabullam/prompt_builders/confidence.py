"""Confidence prompt builder for JSON with prediction and confidence score."""

from ..base.prompt_builder import BasePromptBuilder


class ConfidencePromptBuilder(BasePromptBuilder):
    """Prompt builder expecting JSON with prediction and confidence."""

    def _default_instruction(self) -> str:
        return (
            "You are a classification assistant. Based on the provided features, "
            "predict the most likely class and rate your confidence in this prediction."
        )

    def _format_output_section(self) -> str:
        return (
            "## Output Format\n"
            "Respond with a JSON object containing:\n"
            '- "prediction": one of the possible classes\n'
            '- "confidence_score": an integer from 0 to 10 (10 = most confident)\n\n'
            'Example: {"prediction": "class_name", "confidence_score": 8}\n\n'
            "DO NOT add any explanations. Return ONLY the JSON object."
        )
