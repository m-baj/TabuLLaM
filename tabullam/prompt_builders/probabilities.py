"""Probabilities prompt builder for JSON with probability scores for each class."""

from ..base.prompt_builder import BasePromptBuilder


class ProbabilitiesPromptBuilder(BasePromptBuilder):
    """Prompt builder expecting JSON with probability score for each class."""

    def _default_instruction(self) -> str:
        return (
            "You are a classification assistant. Based on the provided features, "
            "estimate the probability score for each possible class."
        )

    def _format_output_section(self) -> str:
        classes = self.task_metadata.class_labels
        classes_str = ", ".join([f'"{c}"' for c in classes])

        example_scores = {label: 5 for label in classes[:2]}
        if len(classes) > 2:
            example_scores["..."] = "..."

        return (
            "## Output Format\n"
            "Respond with a JSON object containing a score (0-10) for EACH class.\n"
            "Higher scores indicate higher probability.\n"
            f"The JSON keys must be exactly: {classes_str}\n\n"
            f"Example format: {example_scores}\n\n"
            "DO NOT add any explanations. Return ONLY the JSON object."
        )
