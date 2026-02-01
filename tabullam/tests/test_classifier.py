"""
Tests for TabularLLMClassifier.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.utils.validation import check_is_fitted

from tabullam.classifier import TabularLLMClassifier
from tabullam.base.llm_backend import LLMResponse, LLMResponseDuration
from tabullam.exceptions import ConfigurationError


class TestTabularLLMClassifierInit:
    """Tests for TabularLLMClassifier initialization."""

    def test_default_parameters(self):
        """Test default parameter values."""
        clf = TabularLLMClassifier()

        assert clf.llm == 'openai:gpt-4o-mini'
        assert clf.mode == 'zero_shot'
        assert clf.k_shots == 5
        assert clf.embedder is None
        assert clf.vector_store == 'sklearn'
        assert clf.task_description is None
        assert clf.instruction is None
        assert clf.positive_class is None
        assert clf.random_state is None
        assert clf.verbose is False
        assert clf.show_progress is True

    def test_custom_parameters(self):
        """Test custom parameter values."""
        clf = TabularLLMClassifier(
            llm='ollama:llama3.1:8b',
            mode='semantic_few_shot',
            k_shots=3,
            task_description='Test task',
            random_state=42,
            verbose=True
        )

        assert clf.llm == 'ollama:llama3.1:8b'
        assert clf.mode == 'semantic_few_shot'
        assert clf.k_shots == 3
        assert clf.task_description == 'Test task'
        assert clf.random_state == 42
        assert clf.verbose is True


class TestTabularLLMClassifierFit:
    """Tests for TabularLLMClassifier.fit method."""

    @pytest.fixture
    def mock_llm_backend(self):
        """Create a mock LLM backend."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            backend = Mock()
            backend.generate.return_value = LLMResponse(
                content='medium',
                usage={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}
            )
            mock.return_value = backend
            yield mock

    def test_fit_stores_classes(self, mock_llm_backend, sample_dataframe, sample_labels):
        """Test that fit stores class labels."""
        clf = TabularLLMClassifier()
        clf.fit(sample_dataframe, sample_labels)

        assert hasattr(clf, 'classes_')
        assert set(clf.classes_) == {'low', 'medium', 'high'}

    def test_fit_stores_feature_info(self, mock_llm_backend, sample_dataframe, sample_labels):
        """Test that fit stores feature information."""
        clf = TabularLLMClassifier()
        clf.fit(sample_dataframe, sample_labels)

        assert clf.n_features_in_ == 3
        assert list(clf.feature_names_in_) == ['age', 'income', 'education']

    def test_fit_with_numpy_array(self, mock_llm_backend, sample_labels):
        """Test fit with numpy array instead of DataFrame."""
        X = np.array([[25, 30000], [35, 50000], [45, 70000], [55, 90000], [30, 40000]])
        clf = TabularLLMClassifier()
        clf.fit(X, sample_labels)

        assert clf.n_features_in_ == 2

    def test_fit_returns_self(self, mock_llm_backend, sample_dataframe, sample_labels):
        """Test that fit returns self for chaining."""
        clf = TabularLLMClassifier()
        result = clf.fit(sample_dataframe, sample_labels)

        assert result is clf

    def test_fit_invalid_mode_raises(self, sample_dataframe, sample_labels):
        """Test that invalid mode raises ConfigurationError."""
        clf = TabularLLMClassifier(mode='invalid_mode')

        with pytest.raises(ConfigurationError):
            clf.fit(sample_dataframe, sample_labels)

    def test_fit_sets_positive_class_for_binary(self, mock_llm_backend, sample_dataframe, binary_labels):
        """Test that positive class is set automatically for binary classification."""
        clf = TabularLLMClassifier()
        clf.fit(sample_dataframe, binary_labels)

        assert clf.positive_class is not None


class TestTabularLLMClassifierZeroShot:
    """Tests for zero-shot classification mode."""

    @pytest.fixture
    def fitted_clf(self, sample_dataframe, sample_labels):
        """Create a fitted classifier with mocked backend."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            backend = Mock()
            backend.generate.return_value = LLMResponse(
                content='medium',
                usage={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}
            )
            mock.return_value = backend

            clf = TabularLLMClassifier(mode='zero_shot', show_progress=False)
            clf.fit(sample_dataframe, sample_labels)
            return clf

    def test_predict_returns_array(self, fitted_clf, sample_dataframe):
        """Test that predict returns numpy array."""
        predictions = fitted_clf.predict(sample_dataframe.iloc[:2])

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2

    def test_predict_proba_returns_array(self, fitted_clf, sample_dataframe):
        """Test that predict_proba returns numpy array."""
        # Mock the LLM to return probabilities format
        fitted_clf._llm_backend.generate.return_value = LLMResponse(
            content='{"low": 2, "medium": 5, "high": 3}',
            usage={}
        )

        probas = fitted_clf.predict_proba(sample_dataframe.iloc[:2])

        assert isinstance(probas, np.ndarray)
        assert probas.shape == (2, 3)  # 2 samples, 3 classes

    def test_predict_proba_sums_to_one(self, fitted_clf, sample_dataframe):
        """Test that probabilities sum to 1."""
        fitted_clf._llm_backend.generate.return_value = LLMResponse(
            content='{"low": 2, "medium": 5, "high": 3}',
            usage={}
        )

        probas = fitted_clf.predict_proba(sample_dataframe.iloc[:1])

        assert abs(probas[0].sum() - 1.0) < 0.001


class TestTabularLLMClassifierStaticFewShot:
    """Tests for static few-shot classification mode."""

    @pytest.fixture
    def fitted_clf(self, sample_dataframe, sample_labels):
        """Create a fitted classifier with static few-shot mode."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            backend = Mock()
            backend.generate.return_value = LLMResponse(
                content='medium',
                usage={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}
            )
            mock.return_value = backend

            clf = TabularLLMClassifier(
                mode='random_few_shot',
                k_shots=3,
                random_state=42,
                show_progress=False
            )
            clf.fit(sample_dataframe, sample_labels)
            return clf

    def test_fit_samples_examples(self, fitted_clf):
        """Test that fit samples random examples."""
        assert hasattr(fitted_clf, '_static_indices')
        assert len(fitted_clf._static_indices) == 3

    def test_static_indices_reproducible(self, sample_dataframe, sample_labels):
        """Test that static indices are reproducible with random_state."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            mock.return_value = Mock()

            clf1 = TabularLLMClassifier(mode='random_few_shot', k_shots=3, random_state=42)
            clf1.fit(sample_dataframe, sample_labels)

            clf2 = TabularLLMClassifier(mode='random_few_shot', k_shots=3, random_state=42)
            clf2.fit(sample_dataframe, sample_labels)

            np.testing.assert_array_equal(clf1._static_indices, clf2._static_indices)

    def test_get_examples_returns_examples(self, fitted_clf, sample_features):
        """Test that _get_examples returns examples."""
        examples = fitted_clf._get_examples(sample_features)

        assert examples is not None
        assert len(examples) == 3
        assert all('features' in ex and 'label' in ex for ex in examples)


class TestTabularLLMClassifierDynamicFewShot:
    """Tests for dynamic few-shot classification mode."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder."""
        embedder = Mock()
        embedder.embed.return_value = np.random.randn(5, 384)
        embedder.embed_dataframe.return_value = np.random.randn(5, 384)
        return embedder

    @pytest.fixture
    def fitted_clf(self, sample_dataframe, sample_labels, mock_embedder):
        """Create a fitted classifier with dynamic few-shot mode."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            backend = Mock()
            backend.generate.return_value = LLMResponse(
                content='medium',
                usage={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110}
            )
            mock.return_value = backend

            clf = TabularLLMClassifier(
                mode='semantic_few_shot',
                k_shots=3,
                embedder=mock_embedder,
                show_progress=False
            )
            clf.fit(sample_dataframe, sample_labels)
            return clf

    def test_fit_computes_embeddings(self, fitted_clf, mock_embedder):
        """Test that fit computes embeddings."""
        mock_embedder.embed_dataframe.assert_called_once()

    def test_fit_builds_vector_store(self, fitted_clf):
        """Test that fit builds vector store."""
        assert hasattr(fitted_clf, '_vector_store')
        assert len(fitted_clf._vector_store) == 5

    def test_get_examples_queries_vector_store(self, fitted_clf, sample_features, mock_embedder):
        """Test that _get_examples queries vector store."""
        mock_embedder.embed.return_value = np.random.randn(1, 384)

        examples = fitted_clf._get_examples(sample_features)

        assert examples is not None
        assert len(examples) == 3


class TestTabularLLMClassifierPredictWithMetadata:
    """Tests for predict_with_metadata method."""

    @pytest.fixture
    def fitted_clf(self, sample_dataframe, sample_labels):
        """Create a fitted classifier."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            backend = Mock()
            backend.generate.return_value = LLMResponse(
                content='{"low": 2, "medium": 5, "high": 3}',
                usage={'input_tokens': 100, 'output_tokens': 10, 'total_tokens': 110},
                duration=LLMResponseDuration(50.0, 30.0, 80.0),
                reasoning='Step by step reasoning'
            )
            mock.return_value = backend

            clf = TabularLLMClassifier(mode='zero_shot', show_progress=False)
            clf.fit(sample_dataframe, sample_labels)
            return clf

    def test_returns_list_of_dicts(self, fitted_clf, sample_dataframe):
        """Test that predict_with_metadata returns list of dicts."""
        results = fitted_clf.predict_with_metadata(sample_dataframe.iloc[:2])

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_results_contain_metadata(self, fitted_clf, sample_dataframe):
        """Test that results contain metadata fields."""
        results = fitted_clf.predict_with_metadata(sample_dataframe.iloc[:1])

        result = results[0]
        assert 'prediction' in result
        assert 'prompt' in result
        assert 'raw_response' in result
        assert 'usage' in result
        assert 'duration' in result


class TestTabularLLMClassifierSklearnInterface:
    """Tests for sklearn interface compatibility."""

    @pytest.fixture
    def fitted_clf(self, sample_dataframe, sample_labels):
        """Create a fitted classifier."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            backend = Mock()
            backend.generate.return_value = LLMResponse(content='medium', usage={})
            mock.return_value = backend

            clf = TabularLLMClassifier(show_progress=False)
            clf.fit(sample_dataframe, sample_labels)
            return clf

    def test_get_params(self):
        """Test get_params returns all parameters."""
        clf = TabularLLMClassifier(llm='test:model', k_shots=10)
        params = clf.get_params()

        assert params['llm'] == 'test:model'
        assert params['k_shots'] == 10

    def test_set_params(self):
        """Test set_params updates parameters."""
        clf = TabularLLMClassifier()
        clf.set_params(llm='new:model', k_shots=7)

        assert clf.llm == 'new:model'
        assert clf.k_shots == 7

    def test_set_params_returns_self(self):
        """Test set_params returns self for chaining."""
        clf = TabularLLMClassifier()
        result = clf.set_params(k_shots=7)

        assert result is clf

    def test_is_classifier(self):
        """Test that classifier is recognized as sklearn classifier."""
        from sklearn.base import is_classifier
        clf = TabularLLMClassifier()
        assert is_classifier(clf)

    def test_check_is_fitted_raises_before_fit(self):
        """Test that check_is_fitted raises before fit."""
        clf = TabularLLMClassifier()

        with pytest.raises(Exception):  # sklearn raises NotFittedError
            check_is_fitted(clf)

    def test_check_is_fitted_passes_after_fit(self, fitted_clf):
        """Test that check_is_fitted passes after fit."""
        # Should not raise
        check_is_fitted(fitted_clf)

    def test_predict_before_fit_raises(self, sample_dataframe):
        """Test that predict before fit raises error."""
        clf = TabularLLMClassifier()

        with pytest.raises(Exception):
            clf.predict(sample_dataframe)

    def test_predict_proba_before_fit_raises(self, sample_dataframe):
        """Test that predict_proba before fit raises error."""
        clf = TabularLLMClassifier()

        with pytest.raises(Exception):
            clf.predict_proba(sample_dataframe)


class TestTabularLLMClassifierEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock LLM backend."""
        with patch('tabullam.classifier.create_llm_backend') as mock:
            backend = Mock()
            mock.return_value = backend
            yield backend

    def test_predict_with_numpy_array(self, mock_backend, sample_dataframe, sample_labels):
        """Test predict with numpy array input."""
        mock_backend.generate.return_value = LLMResponse(content='medium', usage={})

        clf = TabularLLMClassifier(show_progress=False)
        clf.fit(sample_dataframe, sample_labels)

        X_test = sample_dataframe.iloc[:2].values
        predictions = clf.predict(X_test)

        assert len(predictions) == 2

    def test_handles_parse_error_gracefully(self, mock_backend, sample_dataframe, sample_labels):
        """Test that parse errors are handled gracefully."""
        mock_backend.generate.return_value = LLMResponse(
            content='This is not a valid class',
            usage={}
        )

        clf = TabularLLMClassifier(show_progress=False)
        clf.fit(sample_dataframe, sample_labels)

        predictions = clf.predict(sample_dataframe.iloc[:1])

        # Should return default class instead of raising
        assert len(predictions) == 1

    def test_k_shots_larger_than_training_data(self, mock_backend, sample_labels):
        """Test k_shots larger than training data is handled."""
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y = sample_labels[:3]

        clf = TabularLLMClassifier(mode='random_few_shot', k_shots=100, show_progress=False)
        clf.fit(X, y)

        # Should use all available samples
        assert len(clf._static_indices) == 3

    def test_single_class_classification(self, mock_backend):
        """Test classification with single class."""
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = np.array(['only_class', 'only_class', 'only_class'])

        mock_backend.generate.return_value = LLMResponse(content='only_class', usage={})

        clf = TabularLLMClassifier(show_progress=False)
        clf.fit(X, y)

        assert len(clf.classes_) == 1
