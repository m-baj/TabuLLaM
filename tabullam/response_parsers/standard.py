"""Standard parser for plain text class predictions."""

import re
import logging
from typing import Dict, Any

from ..base.response_parser import BaseResponseParser


logger = logging.getLogger(__name__)


class StandardParser(BaseResponseParser):
    """Parser for plain text class predictions."""

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """Parse plain text response to extract class label."""
        response_lower = raw_response.lower().strip()

        # Try exact match first
        for label in self.class_labels:
            if response_lower == label.lower():
                return {'prediction': label}

        # Try to find class name in response (longest first to avoid partial matches)
        sorted_labels = sorted(self.class_labels, key=len, reverse=True)
        pattern = r"(" + "|".join(map(re.escape, sorted_labels)) + r")"
        matches = re.findall(pattern, raw_response, re.IGNORECASE)

        if matches:
            # Return the last match (often the final answer)
            found = matches[-1]
            for label in self.class_labels:
                if found.lower() == label.lower():
                    return {'prediction': label}

        logger.warning(f"StandardParser: No valid class found in '{raw_response[:100]}...'")
        return {'prediction': None, 'error': 'No valid class found in response'}
