"""Base class for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.

    Vector stores enable efficient similarity search over embeddings
    for dynamic few-shot example selection.
    """

    @abstractmethod
    def add(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add embeddings to the store.

        Parameters
        ----------
        embeddings : ndarray of shape (n_samples, embedding_dim)
            Embeddings to add
        metadata : list of dict, optional
            Metadata for each embedding (e.g., original row data, labels)
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> Tuple[List[int], List[float]]:
        """
        Find k nearest neighbors.

        Parameters
        ----------
        query_embedding : ndarray of shape (embedding_dim,) or (1, embedding_dim)
            Query vector
        k : int
            Number of neighbors to return

        Returns
        -------
        indices : list of int
            Indices of nearest neighbors
        distances : list of float
            Distances to nearest neighbors
        """
        pass

    @abstractmethod
    def get_metadata(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get metadata for given indices.

        Parameters
        ----------
        indices : list of int
            Indices to retrieve metadata for

        Returns
        -------
        metadata : list of dict
            Metadata for each index
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of vectors in store."""
        pass
