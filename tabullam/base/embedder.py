"""Base class for embedders."""

from abc import ABC, abstractmethod
from typing import List, Callable, Optional
import numpy as np
import pandas as pd


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedders.

    All embedders must implement the `embed` method that converts
    text into dense vector representations.

    Examples
    --------
    >>> class MyEmbedder(BaseEmbedder):
    ...     def embed(self, texts):
    ...         return np.random.randn(len(texts), 384)
    ...     @property
    ...     def embedding_dim(self):
    ...         return 384
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into dense vectors.

        Parameters
        ----------
        texts : list of str
            Texts to embed

        Returns
        -------
        embeddings : ndarray of shape (n_texts, embedding_dim)
            Dense vector representations
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        serialization_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Embed rows of a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Data to embed
        columns : list of str, optional
            Columns to include. If None, uses all columns.
        serialization_fn : callable, optional
            Function to serialize rows to text.
            Default: key_value_serialize

        Returns
        -------
        embeddings : ndarray of shape (n_rows, embedding_dim)
        """
        from ..utils.serialization import key_value_serialize

        if serialization_fn is None:
            serialization_fn = lambda row: key_value_serialize(row)

        if columns is not None:
            df = df[columns]

        texts = [serialization_fn(row.to_dict()) for _, row in df.iterrows()]
        return self.embed(texts)
