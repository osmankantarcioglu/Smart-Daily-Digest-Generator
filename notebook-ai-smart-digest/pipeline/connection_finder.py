"""
Connection Finder
-----------------
Discovers semantic cross-note connections — the core "second brain" insight
from the Smart Daily Digest feature proposal.

Algorithm:
  For each note, query the vector store for its most similar neighbours.
  Filter by a minimum similarity threshold, deduplicate symmetric pairs,
  and label the connection type based on score strength.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vector_store import Note, NoteVectorStore, SearchResult


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Connection:
    source: Note
    target: Note
    score: float          # cosine similarity
    label: str            # human-readable connection type

    def summary(self) -> str:
        return (
            f"'{self.source.title}' ↔ '{self.target.title}' "
            f"[{self.label}, score={self.score:.2f}]"
        )


# ─────────────────────────────────────────────
# Connection finder
# ─────────────────────────────────────────────

class ConnectionFinder:
    """
    Finds the most surprising cross-note connections in a user's library.

    Threshold guidance:
      ≥ 0.80  → highly related (nearly the same topic)
      0.60–0.79 → conceptually linked (shared ideas, different context)
      0.45–0.59 → loosely connected (worth surfacing as an insight)
      < 0.45  → noise — skip
    """

    def __init__(
        self,
        store: NoteVectorStore,
        embedder,
        threshold: float = 0.50,
        top_k_per_note: int = 3,
    ):
        self.store = store
        self.embedder = embedder
        self.threshold = threshold
        self.top_k_per_note = top_k_per_note

    def find(self, notes: list[Note], max_connections: int = 5) -> list[Connection]:
        """
        Return the top unique cross-note connections sorted by score descending.
        """
        seen: set[frozenset] = set()
        connections: list[Connection] = []

        for note in notes:
            embedding = self.embedder.embed_single(note.to_embed_text())
            results: list[SearchResult] = self.store.search(
                embedding,
                k=self.top_k_per_note,
                exclude_id=note.id,
                min_score=self.threshold,
            )
            for result in results:
                pair_key = frozenset([note.id, result.note.id])
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                connections.append(
                    Connection(
                        source=note,
                        target=result.note,
                        score=result.score,
                        label=self._label(result.score),
                    )
                )

        # Highest-score connections first; return top N
        connections.sort(key=lambda c: c.score, reverse=True)
        return connections[:max_connections]

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.80:
            return "highly related"
        elif score >= 0.60:
            return "conceptually linked"
        else:
            return "loosely connected"
