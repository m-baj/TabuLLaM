"""Sklearn-based vector store using NearestNeighbors."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.neighbors import NearestNeighbors

from ..base.vector_store import BaseVectorStore


class SklearnVectorStore(BaseVectorStore):
    """
    Vector store using scikit-learn's NearestNeighbors.

    Good for small to medium datasets (< 100k vectors).

    Parameters
    ----------
    metric : str, default='cosine'
        Distance metric: 'cosine', 'euclidean', 'manhattan'

    algorithm : str, default='auto'
        Algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'

    Examples
    --------
    >>> store = SklearnVectorStore(metric='cosine')
    >>> embeddings = np.random.randn(100, 384)
    >>> metadata = [{'label': 'yes' if i % 2 == 0 else 'no'} for i in range(100)]
    >>> store.add(embeddings, metadata)
    >>> indices, distances = store.query(np.random.randn(384), k=5)
    """

    def __init__(
        self,
        metric: str = 'cosine',
        algorithm: str = 'auto'
    ):
        self.metric = metric
        self.algorithm = algorithm
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: Optional[List[Dict[str, Any]]] = None
        self._index: Optional[NearestNeighbors] = None

    def add(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add embeddings to the store and build index."""
        self._embeddings = embeddings
        self._metadata = metadata or [{} for _ in range(len(embeddings))]

        self._index = NearestNeighbors(
            metric=self.metric,
            algorithm=self.algorithm,
            n_neighbors=min(10, len(embeddings))  # Default max neighbors
        )
        self._index.fit(embeddings)

    def query(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> Tuple[List[int], List[float]]:
        """Find k nearest neighbors."""
        if self._index is None:
            raise ValueError("Vector store is empty. Call add() first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure k doesn't exceed number of stored vectors
        k = min(k, len(self._embeddings))

        distances, indices = self._index.kneighbors(query_embedding, n_neighbors=k)
        return indices[0].tolist(), distances[0].tolist()

    def get_metadata(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get metadata for given indices."""
        if self._metadata is None:
            return [{} for _ in indices]
        return [self._metadata[i] for i in indices]

    def get_embeddings(self, indices: List[int]) -> np.ndarray:
        """Get embeddings for given indices."""
        if self._embeddings is None:
            raise ValueError("Vector store is empty.")
        return self._embeddings[indices]

    def __len__(self) -> int:
        """Return number of vectors in store."""
        return len(self._embeddings) if self._embeddings is not None else 0
