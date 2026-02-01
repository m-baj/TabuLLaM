"""Probabilities parser for JSON with probability scores for each class."""

import json
import logging
from typing import Dict, Any

import numpy as np

from ..base.response_parser import BaseResponseParser
from ..exceptions import ParseError


logger = logging.getLogger(__name__)


class ProbabilitiesParser(BaseResponseParser):
    """Parser for JSON with probability scores for each class."""

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON response with probability scores."""
        try:
            data = self._extract_json(raw_response)

            # Extract scores for each class
            raw_scores = {}
            for label in self.class_labels:
                # Try exact match first, then case-insensitive
                if label in data:
                    raw_scores[label] = float(data[label])
                else:
                    # Try case-insensitive match
                    found = False
                    for key, value in data.items():
                        if key.lower() == label.lower():
                            raw_scores[label] = float(value)
                            found = True
                            break
                    if not found:
                        raw_scores[label] = 0.0

            # Normalize scores to probabilities
            scores_array = np.array([raw_scores[label] for label in self.class_labels])
            total = scores_array.sum()

            if total > 0:
                probs_array = scores_array / total
            else:
                probs_array = np.full(len(self.class_labels), 1.0 / len(self.class_labels))

            probabilities = dict(zip(self.class_labels, probs_array.tolist()))

            # Get prediction (highest probability)
            prediction = self.class_labels[np.argmax(probs_array)]

            result = {
                'prediction': prediction,
                'probabilities': probabilities,
                'raw_scores': raw_scores,
            }

            # Add positive class probability if specified
            if self.positive_class and self.positive_class in probabilities:
                result['positive_class_prob'] = probabilities[self.positive_class]

            return result

        except (json.JSONDecodeError, ParseError) as e:
            logger.warning(f"ProbabilitiesParser: {e}")
            return {'prediction': None, 'error': str(e)}
        except Exception as e:
            logger.warning(f"ProbabilitiesParser unexpected error: {e}")
            return {'prediction': None, 'error': str(e)}
