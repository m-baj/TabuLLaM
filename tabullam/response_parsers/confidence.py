"""Confidence parser for JSON with prediction and confidence score."""

import json
import logging
from typing import Dict, Any

import numpy as np

from ..base.response_parser import BaseResponseParser
from ..exceptions import ParseError


logger = logging.getLogger(__name__)


class ConfidenceParser(BaseResponseParser):
    """Parser for JSON with prediction and confidence score."""

    def parse(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON response with prediction and confidence."""
        try:
            data = self._extract_json(raw_response)

            prediction = str(data.get('prediction', 'unknown'))
            raw_score = int(data.get('confidence_score', 5))

            # Normalize confidence to [0, 1]
            confidence = float(np.clip(raw_score, 0, 10)) / 10.0

            result = {
                'prediction': prediction,
                'confidence': confidence,
                'raw_confidence_score': raw_score,
            }

            # Calculate positive class probability for binary classification
            if self.positive_class:
                if prediction == self.positive_class:
                    result['positive_class_prob'] = confidence
                else:
                    result['positive_class_prob'] = 1.0 - confidence

            return result

        except (json.JSONDecodeError, ParseError) as e:
            logger.warning(f"ConfidenceParser: {e}")
            return {'prediction': None, 'error': str(e)}
        except Exception as e:
            logger.warning(f"ConfidenceParser unexpected error: {e}")
            return {'prediction': None, 'error': str(e)}
