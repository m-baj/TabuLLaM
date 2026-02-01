"""Standard prompt builder for plain text class prediction."""

from ..base.prompt_builder import BasePromptBuilder


class StandardPromptBuilder(BasePromptBuilder):
    """Prompt builder expecting plain text class prediction."""

    def _default_instruction(self) -> str:
        return (
            "You are a classification assistant. Based on the provided features, "
            "predict the most likely class. Respond with ONLY the class name, "
            "nothing else."
        )

    def _format_output_section(self) -> str:
        return (
            "## Output\n"
            f"Respond with exactly one of: {', '.join(self.task_metadata.class_labels)}\n"
            "DO NOT add any explanations or analysis."
        )
