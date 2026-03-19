"""
Model Router
------------
Combines the ComplexityClassifier with runtime stats tracking.

For each query it:
  1. Classifies complexity
  2. Records which model was selected
  3. Accumulates cost savings vs always-premium baseline
  4. Optionally executes the LLM call (when OPENAI_API_KEY is set)

This makes the cost-reduction impact quantifiable — directly supporting
the Task 3 argument: "reduces cost 40–60% with minimal perceived quality loss."
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from .complexity_classifier import (
    ClassificationResult,
    ComplexityClassifier,
    PREMIUM_COST_PER_1K,
    QueryComplexity,
)


# ─────────────────────────────────────────────
# Router stats
# ─────────────────────────────────────────────

@dataclass
class RouterStats:
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0   # if we always used gpt-4o

    @property
    def total_queries(self) -> int:
        return sum(self.counts.values())

    @property
    def cost_reduction_pct(self) -> float:
        if self.baseline_cost_usd == 0:
            return 0.0
        return (1 - self.total_cost_usd / self.baseline_cost_usd) * 100

    def summary(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "distribution": dict(self.counts),
            "actual_cost_usd": round(self.total_cost_usd, 6),
            "baseline_cost_usd": round(self.baseline_cost_usd, 6),
            "cost_reduction_pct": round(self.cost_reduction_pct, 1),
        }


# ─────────────────────────────────────────────
# Router result
# ─────────────────────────────────────────────

@dataclass
class RouterResult:
    classification: ClassificationResult
    response: str | None          # LLM output (None if execute=False)
    latency_ms: float
    executed: bool

    def to_display(self) -> dict:
        c = self.classification
        return {
            "query": c.query[:120] + ("…" if len(c.query) > 120 else ""),
            "complexity": c.complexity.value,
            "model": c.recommended_model,
            "score": c.score,
            "confidence": c.confidence,
            "cost_per_1k_usd": c.cost_per_1k,
            "saving_vs_gpt4o": f"{c.cost_saving_vs_premium:.0%}",
            "est_latency_ms": c.avg_latency_ms,
            "reasoning": c.reasoning,
        }


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────

# Assume ~500 tokens per query (input + output average) for cost calc
AVG_TOKENS_PER_QUERY = 0.5  # in thousands


class ModelRouter:
    """
    Routes queries to the cheapest capable model.

    Usage:
        router = ModelRouter()
        result = router.route(query, execute=False)   # classify only
        result = router.route(query, execute=True)    # classify + call LLM
        print(router.stats.summary())
    """

    def __init__(self):
        self.classifier = ComplexityClassifier()
        self.stats = RouterStats()
        self._client = None

    def route(
        self,
        query: str,
        context: str = "",
        execute: bool = False,
    ) -> RouterResult:
        """
        Classify the query and optionally call the selected model.

        Args:
            query:   The user's natural-language query.
            context: Optional additional context (note content, etc.)
            execute: If True and OPENAI_API_KEY is set, calls the LLM.
        """
        t0 = time.perf_counter()
        classification = self.classifier.classify(query)
        self._record(classification)

        response = None
        if execute and os.getenv("OPENAI_API_KEY"):
            response = self._call(classification.recommended_model, query, context)

        latency_ms = (time.perf_counter() - t0) * 1000

        return RouterResult(
            classification=classification,
            response=response,
            latency_ms=round(latency_ms, 2),
            executed=execute and response is not None,
        )

    def route_batch(self, queries: list[str]) -> list[RouterResult]:
        """Classify a list of queries (no LLM calls). Useful for demo/analysis."""
        return [self.route(q, execute=False) for q in queries]

    # ── Stats tracking ────────────────────────

    def _record(self, c: ClassificationResult) -> None:
        self.stats.counts[c.complexity.value] += 1
        self.stats.total_cost_usd += c.cost_per_1k * AVG_TOKENS_PER_QUERY
        self.stats.baseline_cost_usd += PREMIUM_COST_PER_1K * AVG_TOKENS_PER_QUERY

    # ── LLM call ─────────────────────────────

    def _call(self, model: str, query: str, context: str) -> str:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI()

        messages = []
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": query})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=600,
        )
        return response.choices[0].message.content
