"""Multiclass confidence parser for JSON with per-class confidence scores."""

import json
import logging
from typing import Dict, Any

import numpy as np

from ..base.response_parser import BaseResponseParser
from ..exceptions import ParseError


logger = logging.getLogger(__name__)


class MulticlassConfidenceParser(BaseResponseParser):
    """Parser for JSON with confidence scores for each class (multiclass classification)."""

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON response with per-class confidence scores."""
        try:
            data = self._extract_json(raw_response)

            raw_scores = {}
            for label in self.class_labels:
                if label in data:
                    raw_scores[label] = float(data[label])
                else:
                    found = False
                    for key, value in data.items():
                        if key.lower() == label.lower():
                            raw_scores[label] = float(value)
                            found = True
                            break
                    if not found:
                        raw_scores[label] = 0.0

            scores_array = np.array([raw_scores[label] for label in self.class_labels])
            total = scores_array.sum()

            if total > 0:
                probs_array = scores_array / total
            else:
                probs_array = np.full(len(self.class_labels), 1.0 / len(self.class_labels))

            probabilities = dict(zip(self.class_labels, probs_array.tolist()))

            prediction = self.class_labels[np.argmax(probs_array)]

            result = {
                'prediction': prediction,
                'probabilities': probabilities,
                'raw_scores': raw_scores,
            }

            if self.positive_class and self.positive_class in probabilities:
                result['positive_class_prob'] = probabilities[self.positive_class]

            return result

        except (json.JSONDecodeError, ParseError) as e:
            logger.warning(f"MulticlassConfidenceParser: {e}")
            return {'prediction': None, 'error': str(e)}
        except Exception as e:
            logger.warning(f"MulticlassConfidenceParser unexpected error: {e}")
            return {'prediction': None, 'error': str(e)}
