"""Standard prompt builder for plain text class prediction."""

import textwrap

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
        return textwrap.dedent("""
            DO NOT analyze, directly give the prediction answer as a plain class value. Do NOT add any explanations.
        """)
