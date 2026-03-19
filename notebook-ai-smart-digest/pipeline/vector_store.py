"""
Note Vector Store
-----------------
FAISS-backed in-memory vector store for semantic note retrieval.
Uses IndexFlatIP (inner product) on L2-normalised vectors = cosine similarity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Note:
    id: str
    title: str
    content: str
    created_at: str
    tags: list[str] = field(default_factory=list)

    def to_embed_text(self) -> str:
        """Title + content gives richer semantic signal than content alone."""
        return f"{self.title}. {self.content}"

    @classmethod
    def from_dict(cls, d: dict) -> "Note":
        return cls(
            id=d["id"],
            title=d["title"],
            content=d["content"],
            created_at=d["created_at"],
            tags=d.get("tags", []),
        )


@dataclass
class SearchResult:
    note: Note
    score: float  # cosine similarity in [0, 1]


# ─────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────

class NoteVectorStore:
    """Stores note embeddings in FAISS and supports k-NN queries."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # exact cosine on normalised vecs
        self.notes: list[Note] = []

    # ── Building ──────────────────────────────

    def add(self, notes: list[Note], embeddings: np.ndarray) -> None:
        assert embeddings.shape == (len(notes), self.dim)
        self.index.add(embeddings.astype("float32"))
        self.notes.extend(notes)

    @classmethod
    def from_json(cls, path: str | Path, embedder) -> "NoteVectorStore":
        """Load notes from JSON, embed them, and return a ready-to-use store."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        notes = [Note.from_dict(d) for d in raw]
        embeddings = embedder.embed([n.to_embed_text() for n in notes])
        store = cls(dim=embedder.dim)
        store.add(notes, embeddings)
        print(f"[VectorStore] Indexed {len(notes)} notes.")
        return store

    # ── Querying ──────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Return top-k most similar notes. Optionally exclude a note by id."""
        fetch_k = k + (1 if exclude_id else 0)
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype("float32"), fetch_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.notes):
                continue
            note = self.notes[idx]
            if exclude_id and note.id == exclude_id:
                continue
            if float(score) < min_score:
                continue
            results.append(SearchResult(note=note, score=float(score)))
        return results[:k]

    def get_all(self) -> list[Note]:
        return list(self.notes)

    def __len__(self) -> int:
        return len(self.notes)
