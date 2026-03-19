"""
Note Embedder
-------------
Converts note text into dense vector representations using sentence-transformers.
Model: all-MiniLM-L6-v2  (384-dim, fast, completely free — no API key required)
Vectors are L2-normalised so cosine similarity == dot product (faster at query time).
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class NoteEmbedder:
    """Wraps a sentence-transformer model to embed note content into vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Embedder] Loading model: {model_name} …")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim: int = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] Ready — embedding dim: {self.dim}")

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of texts.
        Returns ndarray of shape (N, dim), dtype float32, L2-normalised.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return embeddings.astype("float32")

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dim,), float32, L2-normalised."""
        return self.embed([text])[0]
