"""Embedding module – encode text chunks into dense vectors for retrieval."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from glqa.config import get_settings

logger = logging.getLogger(__name__)


class LegalEmbedder:
    """Wraps a sentence-transformer model for encoding German legal text."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        settings = get_settings().embedding
        self.model_name = model_name or settings.model_name
        self.device = device or settings.device
        self.batch_size = settings.batch_size
        self.normalize = settings.normalize

        logger.info("Loading embedding model: %s (device=%s)", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Encode a list of texts into normalized embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string (no progress bar)."""
        return self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )[0]

    def save_embeddings(self, embeddings: np.ndarray, path: Path) -> None:
        """Save embeddings array to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        logger.info("Saved %d embeddings (%d dim) to %s", embeddings.shape[0], embeddings.shape[1], path)

    def load_embeddings(self, path: Path) -> np.ndarray:
        """Load embeddings array from disk."""
        return np.load(path)
