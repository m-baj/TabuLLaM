"""SentenceTransformers embedding backend."""

import numpy as np
from typing import List

from .base import BaseEmbedder
from ..exceptions import EmbeddingError


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder using sentence-transformers library (local, no API needed).

    Parameters
    ----------
    model : str, default='all-MiniLM-L6-v2'
        Model name from HuggingFace. Popular options:
        - 'all-MiniLM-L6-v2' (384 dim, fast, good quality)
        - 'all-mpnet-base-v2' (768 dim, best quality)
        - 'paraphrase-multilingual-MiniLM-L12-v2' (384 dim, multilingual)

    device : str, default='auto'
        Device to use: 'cpu', 'cuda', or 'auto'

    batch_size : int, default=32
        Batch size for encoding

    Examples
    --------
    >>> embedder = SentenceTransformerEmbedder(model='all-MiniLM-L6-v2')
    >>> embeddings = embedder.embed(['Hello world', 'Goodbye world'])
    >>> embeddings.shape
    (2, 384)
    """

    def __init__(
        self,
        model: str = 'all-MiniLM-L6-v2',
        device: str = 'auto',
        batch_size: int = 32
    ):
        self.model_name = model
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """Lazy initialization of SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                device = self.device if self.device != 'auto' else None
                self._model = SentenceTransformer(self.model_name, device=device)
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers package is required for SentenceTransformerEmbedder. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using SentenceTransformer."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100
        )

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension for the model."""
        return self.model.get_sentence_embedding_dimension()
