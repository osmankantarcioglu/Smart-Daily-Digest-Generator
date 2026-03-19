"""
Query Complexity Classifier
----------------------------
Classifies an incoming user query as SIMPLE / MEDIUM / COMPLEX
and maps it to the cheapest model that can handle it well.

This is the technical implementation of the model-routing trade-off
described in Task 3 of the AppNation case study:

  "Route by query complexity. Use a fast, cheap model (GPT-4o-mini,
   Gemini Flash) for simple queries; reserve the premium model for
   complex multi-document generation. This reduces cost 40–60% with
   minimal perceived quality loss for the majority of users."

Scoring logic (0.0 → 1.0):
  • Query length                  → 0.0–0.30
  • Complex-intent keywords        → 0.0–0.40
  • Simple-intent keywords (penalty) → up to −0.20
  • Multi-document indicators      → +0.20
  • Multiple question marks        → +0.05 each (capped 0.10)

Thresholds:
  score < 0.35  → SIMPLE   → gpt-4o-mini   ($0.15 / 1M tokens)
  score < 0.65  → MEDIUM   → gpt-4o-mini   ($0.15 / 1M tokens)
  score ≥ 0.65  → COMPLEX  → gpt-4o        ($5.00 / 1M tokens)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ─────────────────────────────────────────────
# Enums & constants
# ─────────────────────────────────────────────

class QueryComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# Keywords that signal query complexity
COMPLEX_SIGNALS: list[str] = [
    "analyze", "analyse", "compare", "contrast", "synthesize", "evaluate",
    "critique", "implications", "strategy", "across all", "across my",
    "find connections", "why", "how does", "what patterns", "multi",
    "multiple", "cross-reference", "comprehensive", "in-depth", "deep dive",
    "trade-offs", "pros and cons", "recommend", "should i",
]

SIMPLE_SIGNALS: list[str] = [
    "summarize", "summarise", "translate", "what is", "define", "list",
    "rewrite", "fix grammar", "make shorter", "make longer", "rephrase",
    "bullet points", "tldr", "tl;dr",
]

# Model configs  (costs are per 1 000 tokens, USD, as of 2025)
MODEL_CONFIG: dict[QueryComplexity, dict] = {
    QueryComplexity.SIMPLE: {
        "model": "gpt-4o-mini",
        "cost_per_1k": 0.000_150,
        "avg_latency_ms": 400,
        "description": "Fast & cheap — handles most routine queries",
    },
    QueryComplexity.MEDIUM: {
        "model": "gpt-4o-mini",
        "cost_per_1k": 0.000_150,
        "avg_latency_ms": 700,
        "description": "Still efficient — slightly more tokens needed",
    },
    QueryComplexity.COMPLEX: {
        "model": "gpt-4o",
        "cost_per_1k": 0.005_000,
        "avg_latency_ms": 2_000,
        "description": "Full power — reserved for multi-doc reasoning",
    },
}

# Cost of always using the premium model (baseline for savings calc)
PREMIUM_COST_PER_1K = MODEL_CONFIG[QueryComplexity.COMPLEX]["cost_per_1k"]


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class ClassificationResult:
    query: str
    complexity: QueryComplexity
    score: float                          # raw [0, 1]
    confidence: float                     # distance from boundary, [0.5, 1]
    recommended_model: str
    cost_per_1k: float
    avg_latency_ms: int
    reasoning: dict[str, float] = field(default_factory=dict)

    @property
    def cost_saving_vs_premium(self) -> float:
        """Fraction saved vs always calling gpt-4o."""
        return 1.0 - (self.cost_per_1k / PREMIUM_COST_PER_1K)

    def __str__(self) -> str:
        return (
            f"[{self.complexity.value.upper()}] "
            f"model={self.recommended_model} | "
            f"score={self.score:.2f} | "
            f"saving={self.cost_saving_vs_premium:.0%} vs gpt-4o"
        )


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────

class ComplexityClassifier:
    """
    Lightweight heuristic classifier — no model weights, no API calls,
    sub-millisecond latency. Suitable for high-volume routing at the edge.
    """

    def classify(self, query: str) -> ClassificationResult:
        score, reasoning = self._score(query)
        complexity = self._bucket(score)
        config = MODEL_CONFIG[complexity]

        return ClassificationResult(
            query=query,
            complexity=complexity,
            score=score,
            confidence=self._confidence(score),
            recommended_model=config["model"],
            cost_per_1k=config["cost_per_1k"],
            avg_latency_ms=config["avg_latency_ms"],
            reasoning=reasoning,
        )

    # ── Scoring ───────────────────────────────

    def _score(self, query: str) -> tuple[float, dict]:
        q = query.lower()
        words = q.split()

        length_score = min(len(words) / 50, 1.0) * 0.30

        complex_hits = sum(1 for kw in COMPLEX_SIGNALS if kw in q)
        complex_score = min(complex_hits / 3, 1.0) * 0.40

        simple_hits = sum(1 for kw in SIMPLE_SIGNALS if kw in q)
        simple_penalty = min(simple_hits / 2, 1.0) * 0.20

        multi_doc = bool(
            re.search(r"\b(all|every|across|multiple|compare|both|each)\b", q)
        )
        multi_doc_score = 0.20 if multi_doc else 0.0

        question_score = min(query.count("?") * 0.05, 0.10)

        raw = length_score + complex_score - simple_penalty + multi_doc_score + question_score
        final = max(0.0, min(1.0, raw))

        reasoning = {
            "length_score": round(length_score, 3),
            "complex_hits": complex_hits,
            "complex_score": round(complex_score, 3),
            "simple_hits": simple_hits,
            "simple_penalty": round(simple_penalty, 3),
            "multi_doc": multi_doc,
            "multi_doc_score": round(multi_doc_score, 3),
            "question_score": round(question_score, 3),
            "final_score": round(final, 3),
        }
        return final, reasoning

    @staticmethod
    def _bucket(score: float) -> QueryComplexity:
        if score < 0.35:
            return QueryComplexity.SIMPLE
        elif score < 0.65:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.COMPLEX

    @staticmethod
    def _confidence(score: float) -> float:
        """Higher confidence when score is far from a threshold boundary."""
        return round(0.50 + abs(score - 0.50), 3)
