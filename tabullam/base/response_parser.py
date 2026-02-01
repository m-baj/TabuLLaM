"""Base class for response parsers."""

from abc import ABC, abstractmethod
import re
import json
from typing import Dict, Any, List, Optional

from ..exceptions import ParseError


class BaseResponseParser(ABC):
    """Abstract base class for response parsers."""

    def __init__(
        self,
        class_labels: List[str],
        positive_class: Optional[str] = None
    ):
        self.class_labels = class_labels
        self.positive_class = positive_class

    @abstractmethod
    def parse(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured prediction.

        Parameters
        ----------
        raw_response : str
            Raw text from LLM

        Returns
        -------
        dict
            Must contain 'prediction' key with predicted class.
            May contain additional keys like 'confidence', 'probabilities'.
            Contains 'error' key if parsing failed.
        """
        pass

    def _extract_json(self, raw_response: str) -> Dict[str, Any]:
        """Extract JSON object from response text."""
        match = re.search(r'\{[^{}]*\}', raw_response, re.DOTALL)
        if not match:
            raise ParseError("No JSON object found in response")

        json_str = match.group()

        # Try parsing as-is first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Try fixing common LLM JSON issues:
        # 1. Single quotes -> double quotes
        fixed = json_str.replace("'", '"')

        # 2. Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

        # 3. Try again
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # 4. Try extracting key-value pairs manually for simple cases
        # Pattern: "key": value or 'key': value or key: value
        pairs = re.findall(r'["\']?(\w+)["\']?\s*:\s*([0-9.]+|"[^"]*"|\'[^\']*\')', json_str)
        if pairs:
            result = {}
            for key, value in pairs:
                # Clean up value
                value = value.strip('"\'')
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
            if result:
                return result

        raise ParseError(f"Could not parse JSON: {json_str[:100]}")
