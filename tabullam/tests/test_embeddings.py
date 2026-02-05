"""
Tests for embeddings.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from tabullam.base.embedder import BaseEmbedder
from tabullam.embeddings.ollama import OllamaEmbedder
from tabullam.embeddings.openai import OpenAIEmbedder
from tabullam.embeddings.sentence_transformers import SentenceTransformerEmbedder
from tabullam.exceptions import EmbeddingError


class TestBaseEmbedder:
    """Tests for BaseEmbedder abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseEmbedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_subclass_must_implement_embed(self):
        """Test that subclass must implement embed method."""
        class IncompleteEmbedder(BaseEmbedder):
            @property
            def embedding_dim(self):
                return 384

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_subclass_must_implement_embedding_dim(self):
        """Test that subclass must implement embedding_dim property."""
        class IncompleteEmbedder(BaseEmbedder):
            def embed(self, texts):
                return np.zeros((len(texts), 384))

        with pytest.raises(TypeError):
            IncompleteEmbedder()


class TestBaseEmbedderEmbedDataframe:
    """Tests for BaseEmbedder.embed_dataframe method."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder that returns fixed embeddings."""
        class MockEmbedder(BaseEmbedder):
            def embed(self, texts):
                return np.random.randn(len(texts), 384)

            @property
            def embedding_dim(self):
                return 384

        return MockEmbedder()

    def test_embed_dataframe(self, mock_embedder, sample_dataframe):
        """Test embedding a DataFrame."""
        embeddings = mock_embedder.embed_dataframe(sample_dataframe)

        assert embeddings.shape[0] == len(sample_dataframe)
        assert embeddings.shape[1] == 384

    def test_embed_dataframe_with_columns(self, mock_embedder, sample_dataframe):
        """Test embedding DataFrame with specific columns."""
        embeddings = mock_embedder.embed_dataframe(
            sample_dataframe,
            columns=['age', 'income']
        )

        assert embeddings.shape[0] == len(sample_dataframe)

    def test_embed_dataframe_with_custom_serialization(self, mock_embedder, sample_dataframe):
        """Test embedding DataFrame with custom serialization function."""
        def custom_serialize(row):
            return f"AGE={row.get('age')}"

        embeddings = mock_embedder.embed_dataframe(
            sample_dataframe,
            serialization_fn=custom_serialize
        )

        assert embeddings.shape[0] == len(sample_dataframe)


class TestOllamaEmbedder:
    """Tests for OllamaEmbedder."""

    def test_default_parameters(self):
        """Test default parameters."""
        embedder = OllamaEmbedder()
        assert embedder.model == 'nomic-embed-text'
        assert embedder.base_url == 'http://localhost:11434'
        assert embedder.batch_size == 32

    def test_custom_parameters(self):
        """Test custom parameters."""
        embedder = OllamaEmbedder(
            model='mxbai-embed-large',
            base_url='http://custom:11434',
            batch_size=16
        )
        assert embedder.model == 'mxbai-embed-large'
        assert embedder.base_url == 'http://custom:11434'
        assert embedder.batch_size == 16

    def test_embed_single_text(self):
        """Test embedding a single text."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {'embedding': list(np.random.randn(768))}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            embedder = OllamaEmbedder(show_progress=False)
            result = embedder.embed(['Hello world'])

            assert result.shape == (1, 768)
            mock_post.assert_called_once()

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {'embedding': list(np.random.randn(768))}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            embedder = OllamaEmbedder(batch_size=2, show_progress=False)
            result = embedder.embed(['text1', 'text2', 'text3'])

            assert result.shape == (3, 768)
            assert mock_post.call_count == 3

    def test_embed_error_raises_embedding_error(self):
        """Test that request error raises EmbeddingError."""
        import requests as requests_module
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests_module.RequestException("Connection error")

            embedder = OllamaEmbedder(show_progress=False)

            with pytest.raises(EmbeddingError) as exc_info:
                embedder.embed(['test'])
            assert 'Ollama' in str(exc_info.value)

    def test_embedding_dim_property(self):
        """Test embedding_dim property queries model."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {'embedding': list(np.random.randn(768))}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            embedder = OllamaEmbedder(show_progress=False)
            dim = embedder.embedding_dim

            assert dim == 768


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder."""

    def test_default_parameters(self):
        """Test default parameters."""
        embedder = OpenAIEmbedder()
        assert embedder.model == 'text-embedding-3-small'
        assert embedder.batch_size == 100

    def test_embedding_dim_known_models(self):
        """Test embedding_dim for known models."""
        embedder1 = OpenAIEmbedder(model='text-embedding-3-small')
        assert embedder1.embedding_dim == 1536

        embedder2 = OpenAIEmbedder(model='text-embedding-3-large')
        assert embedder2.embedding_dim == 3072

        embedder3 = OpenAIEmbedder(model='text-embedding-ada-002')
        assert embedder3.embedding_dim == 1536

    def test_embedding_dim_unknown_model_defaults(self):
        """Test embedding_dim defaults to 1536 for unknown models."""
        embedder = OpenAIEmbedder(model='unknown-model')
        assert embedder.embedding_dim == 1536

    def test_embed_texts(self, mock_openai_embedding_response):
        """Test embedding texts using OpenAI API."""
        embedder = OpenAIEmbedder(show_progress=False)

        # Manually set mock client
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_openai_embedding_response
        embedder._client = mock_client

        result = embedder.embed(['Hello world'])

        assert result.shape[0] == 1
        mock_client.embeddings.create.assert_called_once()

    def test_embed_error_raises_embedding_error(self):
        """Test that API error raises EmbeddingError."""
        embedder = OpenAIEmbedder(show_progress=False)

        # Manually set mock client that raises error
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        embedder._client = mock_client

        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(['test'])
        assert 'OpenAI' in str(exc_info.value)

    def test_lazy_initialization(self):
        """Test that client is lazily initialized."""
        embedder = OpenAIEmbedder()
        assert embedder._client is None


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder."""

    def test_default_parameters(self):
        """Test default parameters."""
        embedder = SentenceTransformerEmbedder()
        assert embedder.model_name == 'all-MiniLM-L6-v2'
        assert embedder.device == 'auto'
        assert embedder.batch_size == 32

    def test_custom_parameters(self):
        """Test custom parameters."""
        embedder = SentenceTransformerEmbedder(
            model='all-mpnet-base-v2',
            device='cpu',
            batch_size=16
        )
        assert embedder.model_name == 'all-mpnet-base-v2'
        assert embedder.device == 'cpu'
        assert embedder.batch_size == 16

    def test_embed_texts(self):
        """Test embedding texts using SentenceTransformer."""
        embedder = SentenceTransformerEmbedder()

        # Manually set mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.random.randn(2, 384)
        embedder._model = mock_model

        result = embedder.embed(['text1', 'text2'])

        assert result.shape == (2, 384)
        mock_model.encode.assert_called_once()

    def test_embedding_dim_property(self):
        """Test embedding_dim property calls get_sentence_embedding_dimension."""
        embedder = SentenceTransformerEmbedder()

        # Manually set mock model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        embedder._model = mock_model

        dim = embedder.embedding_dim

        assert dim == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    def test_lazy_initialization(self):
        """Test that model is lazily initialized."""
        embedder = SentenceTransformerEmbedder()
        assert embedder._model is None


class TestEmbedderOutputShapes:
    """Tests for embedding output shapes."""

    @pytest.fixture
    def mock_ollama_embedder(self):
        """Create OllamaEmbedder with mocked API."""
        embedder = OllamaEmbedder(show_progress=False)
        return embedder

    def _mock_requests_post(self, mock_post, dim=768):
        """Helper to set up mock requests.post."""
        mock_response = Mock()
        mock_response.json.return_value = {'embedding': list(np.random.randn(dim))}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

    def test_single_text_returns_2d_array(self, mock_ollama_embedder):
        """Test that single text returns 2D array."""
        with patch('requests.post') as mock_post:
            self._mock_requests_post(mock_post)
            result = mock_ollama_embedder.embed(['single text'])
            assert result.ndim == 2
            assert result.shape[0] == 1

    def test_multiple_texts_return_correct_shape(self, mock_ollama_embedder):
        """Test that multiple texts return correct shape."""
        with patch('requests.post') as mock_post:
            self._mock_requests_post(mock_post)
            texts = ['text1', 'text2', 'text3', 'text4', 'text5']
            result = mock_ollama_embedder.embed(texts)
            assert result.shape[0] == 5

    def test_empty_list_returns_empty_array(self, mock_ollama_embedder):
        """Test that empty list returns empty array."""
        with patch('requests.post') as mock_post:
            self._mock_requests_post(mock_post)
            result = mock_ollama_embedder.embed([])
            assert result.shape[0] == 0
