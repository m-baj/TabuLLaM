"""Response parsers for LLM outputs."""

from typing import Optional, List

from ..base.response_parser import BaseResponseParser
from .standard import StandardParser
from .binary_confidence import BinaryConfidenceParser
from .multiclass_confidence import MulticlassConfidenceParser


def create_response_parser(
    prompt_type: str,
    class_labels: List[str],
    positive_class: Optional[str] = None
) -> BaseResponseParser:
    """
    Factory function to create response parser.

    Parameters
    ----------
    prompt_type : str
        Type of prompt: 'standard', 'binary_confidence', or 'multiclass_confidence'
    class_labels : list of str
        Possible class labels
    positive_class : str, optional
        Positive class for binary classification

    Returns
    -------
    BaseResponseParser
        Configured response parser
    """
    parsers = {
        'standard': StandardParser,
        'binary_confidence': BinaryConfidenceParser,
        'multiclass_confidence': MulticlassConfidenceParser,
    }

    if prompt_type not in parsers:
        raise ValueError(
            f"Unknown prompt_type: '{prompt_type}'. "
            f"Available: {list(parsers.keys())}"
        )

    return parsers[prompt_type](class_labels, positive_class)


__all__ = [
    'BaseResponseParser',
    'StandardParser',
    'BinaryConfidenceParser',
    'MulticlassConfidenceParser',
    'create_response_parser',
]
