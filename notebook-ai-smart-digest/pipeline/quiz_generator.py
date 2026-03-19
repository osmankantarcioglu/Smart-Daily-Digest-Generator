"""
Quiz Generator
--------------
Generates a single spaced-repetition quiz card from a note.
Returns structured JSON so the UI can render it as interactive buttons.

Falls back to a rule-based mock when OPENAI_API_KEY is not set.
"""

from __future__ import annotations

import json
import os
import re

from .vector_store import Note


class QuizGenerator:
    """Produces one 4-option multiple-choice question per note."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None

    def generate(self, note: Note) -> dict:
        """
        Returns:
            {
                "question": str,
                "options": ["A) …", "B) …", "C) …", "D) …"],
                "correct": "B",
                "explanation": str,
                "source_note": str   # note title
            }
        """
        if not os.getenv("OPENAI_API_KEY"):
            return self._mock_quiz(note)

        prompt = f"""\
Generate a single multiple-choice quiz question for spaced-repetition learning.
Base it ONLY on the note content below.

Return valid JSON with this exact schema:
{{
  "question": "...",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct": "A",
  "explanation": "..."
}}

Note title: {note.title}
Note content:
{note.content[:1200]}
"""
        result = self._call_llm(prompt)
        result["source_note"] = note.title
        return result

    # ── LLM call ─────────────────────────────

    def _call_llm(self, prompt: str) -> dict:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=350,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    # ── Mock fallback ─────────────────────────

    @staticmethod
    def _mock_quiz(note: Note) -> dict:
        """Produce a plausible quiz card without calling any API."""
        # Extract first sentence as the "key fact"
        first_sentence = re.split(r"(?<=[.!?])\s", note.content)[0][:120]

        return {
            "question": f"Based on your note '{note.title}': {first_sentence[:80]}… What does this primarily illustrate?",
            "options": [
                "A) A common misconception in the field",
                "B) The core principle described in this note ✓",
                "C) An unrelated historical example",
                "D) A counter-argument to the main idea",
            ],
            "correct": "B",
            "explanation": (
                f"The note '{note.title}' focuses on the concept captured in the first sentence. "
                "Recognising this helps anchor the broader content through spaced repetition."
            ),
            "source_note": note.title,
        }
