"""OpenAI embedding backend."""

import numpy as np
from typing import List, Optional
from tqdm import tqdm

from .base import BaseEmbedder
from ..exceptions import EmbeddingError


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder using OpenAI's embedding API.

    Parameters
    ----------
    model : str, default='text-embedding-3-small'
        OpenAI embedding model. Options:
        - 'text-embedding-3-small' (1536 dim, cost-effective)
        - 'text-embedding-3-large' (3072 dim, highest quality)
        - 'text-embedding-ada-002' (1536 dim, legacy)

    api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY env variable.

    batch_size : int, default=100
        Number of texts per API call (max 2048)

    show_progress : bool, default=True
        Whether to show progress bar

    Examples
    --------
    >>> embedder = OpenAIEmbedder(model='text-embedding-3-small')
    >>> embeddings = embedder.embed(['Hello world', 'Goodbye world'])
    >>> embeddings.shape
    (2, 1536)
    """

    DIMENSIONS = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536,
    }

    def __init__(
        self,
        model: str = 'text-embedding-3-small',
        api_key: Optional[str] = None,
        batch_size: int = 100,
        show_progress: bool = True
    ):
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise EmbeddingError(
                    "openai package is required for OpenAIEmbedder. "
                    "Install it with: pip install openai"
                )
        return self._client

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        embeddings = []
        iterator = range(0, len(texts), self.batch_size)

        if self.show_progress and len(texts) > self.batch_size:
            iterator = tqdm(iterator, desc=f"Embedding with {self.model}")

        for i in iterator:
            batch = texts[i:i + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to get embedding from OpenAI: {e}"
                ) from e

        return np.array(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension for the model."""
        return self.DIMENSIONS.get(self.model, 1536)
