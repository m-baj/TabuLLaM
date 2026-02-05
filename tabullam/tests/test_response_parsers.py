"""
Tests for response parsers.
"""

import pytest
import numpy as np

from tabullam.base.response_parser import BaseResponseParser
from tabullam.response_parsers import (
    StandardParser,
    BinaryConfidenceParser,
    MulticlassConfidenceParser,
    create_response_parser,
)


class TestStandardParser:
    """Tests for StandardParser."""

    @pytest.fixture
    def parser(self):
        """Create a StandardParser instance."""
        return StandardParser(
            class_labels=['low', 'medium', 'high'],
            positive_class=None
        )

    def test_parse_exact_match(self, parser):
        """Test parsing exact class match."""
        result = parser.parse('medium')
        assert result['prediction'] == 'medium'

    def test_parse_exact_match_case_insensitive(self, parser):
        """Test parsing is case-insensitive for exact match."""
        result = parser.parse('MEDIUM')
        assert result['prediction'] == 'medium'

    def test_parse_class_in_sentence(self, parser):
        """Test parsing class name within a sentence."""
        result = parser.parse('I predict the class is high.')
        assert result['prediction'] == 'high'

    def test_parse_last_match_wins(self, parser):
        """Test that last match is returned when multiple classes mentioned."""
        result = parser.parse('It could be low but I think high')
        assert result['prediction'] == 'high'

    def test_parse_no_match_returns_error(self, parser):
        """Test that no match returns error."""
        result = parser.parse('This is completely unrelated text')
        assert result['prediction'] is None
        assert 'error' in result

    def test_parse_with_whitespace(self, parser):
        """Test parsing with extra whitespace."""
        result = parser.parse('  low  ')
        assert result['prediction'] == 'low'

    def test_parse_longest_match_first(self):
        """Test that longer class names are matched before shorter ones."""
        parser = StandardParser(
            class_labels=['cat', 'category'],
            positive_class=None
        )
        result = parser.parse('The category is clear')
        assert result['prediction'] == 'category'


class TestBinaryConfidenceParser:
    """Tests for BinaryConfidenceParser."""

    @pytest.fixture
    def parser(self):
        """Create a BinaryConfidenceParser instance."""
        return BinaryConfidenceParser(
            class_labels=['no', 'yes'],
            positive_class='yes'
        )

    def test_parse_valid_json(self, parser):
        """Test parsing valid JSON response."""
        response = '{"prediction": "yes", "confidence_score": 8}'
        result = parser.parse(response)
        assert result['prediction'] == 'yes'
        assert result['confidence'] == 0.8
        assert result['raw_confidence_score'] == 8

    def test_parse_json_with_surrounding_text(self, parser):
        """Test parsing JSON embedded in text."""
        response = 'Here is my answer: {"prediction": "no", "confidence_score": 6}'
        result = parser.parse(response)
        assert result['prediction'] == 'no'
        assert result['confidence'] == 0.6

    def test_parse_confidence_clamped_to_max(self, parser):
        """Test that confidence above 10 is clamped."""
        response = '{"prediction": "yes", "confidence_score": 15}'
        result = parser.parse(response)
        assert result['confidence'] == 1.0

    def test_parse_confidence_clamped_to_min(self, parser):
        """Test that confidence below 0 is clamped."""
        response = '{"prediction": "yes", "confidence_score": -5}'
        result = parser.parse(response)
        assert result['confidence'] == 0.0

    def test_parse_positive_class_prob_when_predicted(self, parser):
        """Test positive class probability when positive class is predicted."""
        response = '{"prediction": "yes", "confidence_score": 8}'
        result = parser.parse(response)
        assert result['positive_class_prob'] == 0.8

    def test_parse_positive_class_prob_when_not_predicted(self, parser):
        """Test positive class probability when negative class is predicted."""
        response = '{"prediction": "no", "confidence_score": 8}'
        result = parser.parse(response)
        assert abs(result['positive_class_prob'] - 0.2) < 0.001  # 1 - 0.8

    def test_parse_invalid_json_returns_error(self, parser):
        """Test that invalid JSON returns error."""
        response = 'This is not JSON at all'
        result = parser.parse(response)
        assert result['prediction'] is None
        assert 'error' in result

    def test_parse_missing_confidence_uses_default(self, parser):
        """Test that missing confidence uses default value."""
        response = '{"prediction": "yes"}'
        result = parser.parse(response)
        assert result['prediction'] == 'yes'
        assert result['confidence'] == 0.5  # default is 5 / 10


class TestMulticlassConfidenceParser:
    """Tests for MulticlassConfidenceParser."""

    @pytest.fixture
    def parser(self):
        """Create a MulticlassConfidenceParser instance."""
        return MulticlassConfidenceParser(
            class_labels=['low', 'medium', 'high'],
            positive_class='high'
        )

    def test_parse_valid_json(self, parser):
        """Test parsing valid JSON with confidence scores."""
        response = '{"low": 2, "medium": 3, "high": 5}'
        result = parser.parse(response)
        assert result['prediction'] == 'high'
        assert 'probabilities' in result
        # Check normalization: 5/(2+3+5) = 0.5
        assert abs(result['probabilities']['high'] - 0.5) < 0.01

    def test_parse_normalizes_to_sum_one(self, parser):
        """Test that probabilities are normalized to sum to 1."""
        response = '{"low": 1, "medium": 1, "high": 1}'
        result = parser.parse(response)
        probs = result['probabilities']
        assert abs(sum(probs.values()) - 1.0) < 0.001

    def test_parse_highest_score_is_prediction(self, parser):
        """Test that highest score class is the prediction."""
        response = '{"low": 8, "medium": 1, "high": 1}'
        result = parser.parse(response)
        assert result['prediction'] == 'low'

    def test_parse_json_with_surrounding_text(self, parser):
        """Test parsing JSON embedded in text."""
        response = 'My scores are: {"low": 2, "medium": 3, "high": 5}'
        result = parser.parse(response)
        assert result['prediction'] == 'high'

    def test_parse_case_insensitive_keys(self, parser):
        """Test that class keys are matched case-insensitively."""
        response = '{"LOW": 2, "MEDIUM": 3, "HIGH": 5}'
        result = parser.parse(response)
        assert result['prediction'] == 'high'

    def test_parse_missing_class_uses_zero(self, parser):
        """Test that missing class gets zero score."""
        response = '{"low": 5, "medium": 5}'  # 'high' missing
        result = parser.parse(response)
        assert result['raw_scores']['high'] == 0.0

    def test_parse_all_zeros_uniform_distribution(self, parser):
        """Test that all zeros results in uniform distribution."""
        response = '{"low": 0, "medium": 0, "high": 0}'
        result = parser.parse(response)
        probs = result['probabilities']
        expected = 1.0 / 3
        for class_name in ['low', 'medium', 'high']:
            assert abs(probs[class_name] - expected) < 0.01

    def test_parse_invalid_json_returns_error(self, parser):
        """Test that invalid JSON returns error."""
        response = 'Not valid JSON'
        result = parser.parse(response)
        assert result['prediction'] is None
        assert 'error' in result

    def test_parse_positive_class_prob(self, parser):
        """Test positive class probability is included."""
        response = '{"low": 2, "medium": 3, "high": 5}'
        result = parser.parse(response)
        assert 'positive_class_prob' in result
        assert result['positive_class_prob'] == result['probabilities']['high']


class TestCreateResponseParser:
    """Tests for create_response_parser factory function."""

    def test_create_standard_parser(self):
        """Test creating StandardParser."""
        parser = create_response_parser('standard', ['a', 'b', 'c'])
        assert isinstance(parser, StandardParser)

    def test_create_binary_confidence_parser(self):
        """Test creating BinaryConfidenceParser."""
        parser = create_response_parser('binary_confidence', ['no', 'yes'], 'yes')
        assert isinstance(parser, BinaryConfidenceParser)
        assert parser.positive_class == 'yes'

    def test_create_multiclass_confidence_parser(self):
        """Test creating MulticlassConfidenceParser."""
        parser = create_response_parser('multiclass_confidence', ['a', 'b', 'c'])
        assert isinstance(parser, MulticlassConfidenceParser)

    def test_create_invalid_type_raises(self):
        """Test that invalid prompt type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_response_parser('invalid_type', ['a', 'b'])
        assert 'Unknown prompt_type' in str(exc_info.value)

    def test_parser_has_class_labels(self):
        """Test that created parser has correct class labels."""
        labels = ['low', 'medium', 'high']
        parser = create_response_parser('standard', labels)
        assert parser.class_labels == labels
