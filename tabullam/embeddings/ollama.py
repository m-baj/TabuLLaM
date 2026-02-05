"""Ollama embedding backend."""

import numpy as np
from typing import List
from tqdm import tqdm

from .base import BaseEmbedder
from ..exceptions import EmbeddingError


class OllamaEmbedder(BaseEmbedder):
    """
    Embedder using Ollama's local embedding models.

    Parameters
    ----------
    model : str, default='nomic-embed-text'
        Ollama embedding model name. Popular options:
        - 'nomic-embed-text' (768 dim, good general purpose)
        - 'mxbai-embed-large' (1024 dim, high quality)
        - 'all-minilm' (384 dim, fast)

    base_url : str, default='http://localhost:11434'
        Ollama API base URL

    batch_size : int, default=32
        Number of texts to embed in each iteration (sequential calls)

    show_progress : bool, default=True
        Whether to show progress bar for large batches

    Examples
    --------
    >>> embedder = OllamaEmbedder(model='nomic-embed-text')
    >>> embeddings = embedder.embed(['Hello world', 'Goodbye world'])
    >>> embeddings.shape
    (2, 768)
    """

    def __init__(
        self,
        model: str = 'nomic-embed-text',
        base_url: str = 'http://localhost:11434',
        batch_size: int = 32,
        show_progress: bool = True
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._embedding_dim = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using Ollama API."""

        embeddings = []
        iterator = range(0, len(texts), self.batch_size)

        if self.show_progress and len(texts) > self.batch_size:
            iterator = tqdm(iterator, desc=f"Embedding with {self.model}")

        for i in iterator:
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return np.array(embeddings)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        import requests

        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text}
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)

                if self._embedding_dim is None:
                    self._embedding_dim = len(embedding)

            except requests.RequestException as e:
                raise EmbeddingError(
                    f"Failed to get embedding from Ollama: {e}"
                ) from e

        return embeddings

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension, querying model if needed."""
        if self._embedding_dim is None:
            self.embed(["test"])
        return self._embedding_dim
