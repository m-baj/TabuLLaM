"""
Tests for utility functions.
"""

import pytest
import pandas as pd

from tabullam.utils.serialization import key_value_serialize
from tabullam.utils.validation import validate_mode, validate_llm_spec, VALID_MODES, VALID_PROVIDERS
from tabullam.exceptions import ConfigurationError


class TestKeyValueSerialize:
    """Tests for key_value_serialize function."""

    def test_serialize_simple_features(self):
        """Test serialization of simple features."""
        features = {'age': 25, 'income': 50000}
        result = key_value_serialize(features)
        assert 'age: 25' in result
        assert 'income: 50000' in result
        assert ' | ' in result

    def test_serialize_with_string_values(self):
        """Test serialization with string values."""
        features = {'name': 'John', 'city': 'NYC'}
        result = key_value_serialize(features)
        assert 'name: John' in result
        assert 'city: NYC' in result

    def test_serialize_with_label(self):
        """Test serialization with label included."""
        features = {'age': 25}
        label = {'target': 'yes'}
        result = key_value_serialize(features, label)
        assert 'age: 25' in result
        assert '=>' in result
        assert 'target: yes' in result

    def test_serialize_skips_nan_values(self):
        """Test that NaN values are skipped."""
        features = {'age': 25, 'income': float('nan'), 'city': 'NYC'}
        result = key_value_serialize(features)
        assert 'age: 25' in result
        assert 'city: NYC' in result
        assert 'income' not in result

    def test_serialize_with_none_using_pandas(self):
        """Test that None values are handled via pd.notna."""
        features = {'age': 25, 'income': None, 'city': 'NYC'}
        result = key_value_serialize(features)
        assert 'age: 25' in result
        assert 'city: NYC' in result
        # None should be skipped by pd.notna
        assert 'income' not in result

    def test_serialize_empty_features(self):
        """Test serialization of empty features dict."""
        features = {}
        result = key_value_serialize(features)
        assert result == ''

    def test_serialize_single_feature(self):
        """Test serialization of single feature."""
        features = {'age': 25}
        result = key_value_serialize(features)
        assert result == 'age: 25'
        assert ' | ' not in result


class TestValidateMode:
    """Tests for validate_mode function."""

    def test_valid_modes(self):
        """Test that all valid modes pass validation."""
        for mode in VALID_MODES:
            # Should not raise
            validate_mode(mode)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_mode('invalid_mode')
        assert 'invalid_mode' in str(exc_info.value)
        assert 'Must be one of' in str(exc_info.value)

    def test_zero_shot_mode(self):
        """Test zero_shot mode validation."""
        validate_mode('zero_shot')  # Should not raise

    def test_random_few_shot_mode(self):
        """Test random_few_shot mode validation."""
        validate_mode('random_few_shot')  # Should not raise

    def test_semantic_few_shot_mode(self):
        """Test semantic_few_shot mode validation."""
        validate_mode('semantic_few_shot')  # Should not raise


class TestValidateLLMSpec:
    """Tests for validate_llm_spec function."""

    def test_valid_openai_spec(self):
        """Test valid OpenAI specification."""
        provider, model = validate_llm_spec('openai:gpt-4o-mini')
        assert provider == 'openai'
        assert model == 'gpt-4o-mini'

    def test_valid_ollama_spec(self):
        """Test valid Ollama specification."""
        provider, model = validate_llm_spec('ollama:llama3.1:8b')
        assert provider == 'ollama'
        assert model == 'llama3.1:8b'

    def test_valid_google_spec(self):
        """Test valid Google specification."""
        provider, model = validate_llm_spec('google:gemini-1.5-flash')
        assert provider == 'google'
        assert model == 'gemini-1.5-flash'

    def test_missing_colon_raises(self):
        """Test that missing colon raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_llm_spec('invalid_spec')
        assert 'Expected format' in str(exc_info.value)

    def test_invalid_provider_raises(self):
        """Test that invalid provider raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            validate_llm_spec('unknown:model')
        assert 'Invalid provider' in str(exc_info.value)
        assert 'unknown' in str(exc_info.value)

    def test_model_with_multiple_colons(self):
        """Test model name with multiple colons (e.g., ollama:model:tag)."""
        provider, model = validate_llm_spec('ollama:llama3.1:8b-instruct')
        assert provider == 'ollama'
        assert model == 'llama3.1:8b-instruct'
