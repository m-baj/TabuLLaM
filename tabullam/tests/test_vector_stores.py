"""
Tests for vector stores.
"""

import pytest
import numpy as np

from tabullam.base.vector_store import BaseVectorStore
from tabullam.vector_stores.sklearn_store import SklearnVectorStore


class TestBaseVectorStore:
    """Tests for BaseVectorStore abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseVectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_subclass_must_implement_methods(self):
        """Test that subclass must implement all abstract methods."""
        class IncompleteStore(BaseVectorStore):
            def add(self, embeddings, metadata=None):
                pass

        with pytest.raises(TypeError):
            IncompleteStore()


class TestSklearnVectorStore:
    """Tests for SklearnVectorStore."""

    @pytest.fixture
    def store(self):
        """Create a SklearnVectorStore instance."""
        return SklearnVectorStore()

    @pytest.fixture
    def store_with_data(self, sample_embeddings):
        """Create a store with pre-loaded data."""
        store = SklearnVectorStore()
        metadata = [{'label': f'class_{i}'} for i in range(len(sample_embeddings))]
        store.add(sample_embeddings, metadata)
        return store

    def test_default_parameters(self, store):
        """Test default parameters."""
        assert store.metric == 'cosine'
        assert store.algorithm == 'auto'

    def test_custom_parameters(self):
        """Test custom parameters."""
        store = SklearnVectorStore(metric='euclidean', algorithm='brute')
        assert store.metric == 'euclidean'
        assert store.algorithm == 'brute'

    def test_add_embeddings(self, store, sample_embeddings):
        """Test adding embeddings to store."""
        metadata = [{'label': f'class_{i}'} for i in range(len(sample_embeddings))]
        store.add(sample_embeddings, metadata)

        assert len(store) == len(sample_embeddings)

    def test_add_embeddings_without_metadata(self, store, sample_embeddings):
        """Test adding embeddings without metadata."""
        store.add(sample_embeddings)

        assert len(store) == len(sample_embeddings)

    def test_query_returns_indices_and_distances(self, store_with_data, sample_embeddings):
        """Test that query returns indices and distances."""
        query = sample_embeddings[0]
        indices, distances = store_with_data.query(query, k=3)

        assert len(indices) == 3
        assert len(distances) == 3
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(d, float) for d in distances)

    def test_query_1d_embedding(self, store_with_data, sample_embeddings):
        """Test query with 1D embedding."""
        query = sample_embeddings[0]  # 1D array
        indices, distances = store_with_data.query(query, k=2)

        assert len(indices) == 2

    def test_query_2d_embedding(self, store_with_data, sample_embeddings):
        """Test query with 2D embedding (1, dim)."""
        query = sample_embeddings[0].reshape(1, -1)  # 2D array
        indices, distances = store_with_data.query(query, k=2)

        assert len(indices) == 2

    def test_query_k_larger_than_store(self, store_with_data, sample_embeddings):
        """Test query with k larger than number of stored vectors."""
        query = sample_embeddings[0]
        indices, distances = store_with_data.query(query, k=100)

        assert len(indices) == len(sample_embeddings)  # Should return all

    def test_query_empty_store_raises(self, store):
        """Test that querying empty store raises error."""
        with pytest.raises(ValueError) as exc_info:
            store.query(np.random.randn(384), k=3)
        assert 'empty' in str(exc_info.value).lower()

    def test_get_metadata(self, store_with_data):
        """Test getting metadata by indices."""
        metadata = store_with_data.get_metadata([0, 1, 2])

        assert len(metadata) == 3
        assert metadata[0]['label'] == 'class_0'
        assert metadata[1]['label'] == 'class_1'
        assert metadata[2]['label'] == 'class_2'

    def test_get_metadata_without_metadata(self, store, sample_embeddings):
        """Test getting metadata when none was provided."""
        store.add(sample_embeddings)
        metadata = store.get_metadata([0, 1])

        assert len(metadata) == 2
        assert all(m == {} for m in metadata)

    def test_get_embeddings(self, store_with_data, sample_embeddings):
        """Test getting embeddings by indices."""
        embeddings = store_with_data.get_embeddings([0, 2])

        assert embeddings.shape[0] == 2
        np.testing.assert_array_equal(embeddings[0], sample_embeddings[0])
        np.testing.assert_array_equal(embeddings[1], sample_embeddings[2])

    def test_len_empty_store(self, store):
        """Test length of empty store."""
        assert len(store) == 0

    def test_len_after_add(self, store, sample_embeddings):
        """Test length after adding embeddings."""
        store.add(sample_embeddings)
        assert len(store) == len(sample_embeddings)


class TestSklearnVectorStoreMetrics:
    """Tests for different distance metrics."""

    @pytest.fixture
    def embeddings(self):
        """Create simple embeddings for metric testing."""
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ])

    def test_cosine_metric(self, embeddings):
        """Test cosine metric finds similar vectors."""
        store = SklearnVectorStore(metric='cosine')
        store.add(embeddings)

        # Query with vector similar to first
        query = np.array([0.9, 0.1, 0.0])
        indices, distances = store.query(query, k=2)

        # Should find index 0 (most similar) and index 3
        assert 0 in indices

    def test_euclidean_metric(self, embeddings):
        """Test euclidean metric."""
        store = SklearnVectorStore(metric='euclidean')
        store.add(embeddings)

        # Query with exact match
        query = embeddings[0]
        indices, distances = store.query(query, k=1)

        assert indices[0] == 0
        assert distances[0] < 0.01  # Should be very close to 0


class TestSklearnVectorStoreNearestNeighbor:
    """Tests for nearest neighbor search correctness."""

    def test_finds_exact_match_first(self):
        """Test that exact match is returned first."""
        embeddings = np.random.randn(10, 384)
        store = SklearnVectorStore()
        store.add(embeddings)

        # Query with one of the stored embeddings
        query = embeddings[5]
        indices, distances = store.query(query, k=1)

        assert indices[0] == 5

    def test_distances_are_sorted(self):
        """Test that distances are returned in sorted order."""
        embeddings = np.random.randn(20, 384)
        store = SklearnVectorStore()
        store.add(embeddings)

        query = np.random.randn(384)
        indices, distances = store.query(query, k=10)

        # Distances should be monotonically increasing
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]

    def test_all_indices_unique(self):
        """Test that returned indices are unique."""
        embeddings = np.random.randn(10, 384)
        store = SklearnVectorStore()
        store.add(embeddings)

        query = np.random.randn(384)
        indices, _ = store.query(query, k=5)

        assert len(set(indices)) == len(indices)
