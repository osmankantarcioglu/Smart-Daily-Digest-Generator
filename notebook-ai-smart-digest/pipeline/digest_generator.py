"""
Digest Generator
----------------
Calls an LLM to produce the Smart Daily Digest:
  • 3-5 key insights surfaced from the user's recent notes
  • 2 cross-note connections the user may have missed
  • 1 micro-quiz question for spaced-repetition reinforcement

Falls back to a structured mock response when OPENAI_API_KEY is not set,
so the full pipeline can be demoed without API credentials.
"""

from __future__ import annotations

import os

from .connection_finder import Connection
from .vector_store import Note


SYSTEM_PROMPT = """\
You are the Smart Daily Digest AI for NotebookAI — a mobile-first AI note-taking app.
Your job: surface the most valuable knowledge from the user's notes each morning.
Tone: smart, concise, motivating. Like a brilliant study partner.
"""

DIGEST_TEMPLATE = """\
Based on the user's notes below, produce a daily digest with exactly three sections:

## Key Insights
List 3-5 bullet points — each one a genuinely useful insight or reminder from the notes.
Be specific; do not just restate the note titles.

## Connections You Might Have Missed
Write 2 short paragraphs, each linking two notes that share a non-obvious idea.
Start each with the note titles in bold.

## Today's Quiz
One multiple-choice question to reinforce retention (4 options, mark the correct one with ✓).

---
NOTES:
{notes_block}

DETECTED CONNECTIONS:
{connections_block}
"""


class DigestGenerator:
    """Generates the Smart Daily Digest via OpenAI chat completions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._client = None  # lazy init

    # ── Public API ────────────────────────────

    def generate(
        self,
        notes: list[Note],
        connections: list[Connection],
    ) -> str:
        """Return the digest as a markdown string."""
        if not os.getenv("OPENAI_API_KEY"):
            return self._mock_digest(notes, connections)

        prompt = DIGEST_TEMPLATE.format(
            notes_block=self._format_notes(notes),
            connections_block=self._format_connections(connections),
        )
        return self._call_llm(prompt)

    # ── Formatting helpers ────────────────────

    @staticmethod
    def _format_notes(notes: list[Note]) -> str:
        blocks = []
        for n in notes[:6]:   # cap context window
            blocks.append(f"### {n.title}\n{n.content}")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_connections(connections: list[Connection]) -> str:
        if not connections:
            return "No strong connections detected yet."
        lines = [
            f"- '{c.source.title}' ↔ '{c.target.title}' "
            f"({c.label}, similarity={c.score:.2f})"
            for c in connections[:3]
        ]
        return "\n".join(lines)

    # ── LLM call ─────────────────────────────

    def _call_llm(self, user_prompt: str) -> str:
        from openai import OpenAI   # import here to avoid hard dep when mocked

        if self._client is None:
            self._client = OpenAI()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content

    # ── Mock fallback (no API key needed) ─────

    @staticmethod
    def _mock_digest(notes: list[Note], connections: list[Connection]) -> str:
        insight_titles = [f"• **{n.title}**: key ideas resurface here." for n in notes[:4]]
        conn_block = ""
        if len(connections) >= 1:
            c = connections[0]
            conn_block += (
                f"**{c.source.title}** and **{c.target.title}** share a "
                f"non-obvious conceptual link (similarity {c.score:.2f}). "
                "Both deal with optimising cognitive load — one at the system level, "
                "one at the human level.\n\n"
            )
        if len(connections) >= 2:
            c = connections[1]
            conn_block += (
                f"**{c.source.title}** and **{c.target.title}** are "
                f"{c.label}. Consider how the principles in one could "
                "inform your thinking in the other."
            )
        if not conn_block:
            conn_block = "Add more notes to unlock cross-note connections!"

        first_note = notes[0] if notes else None
        quiz = (
            f"**Quiz:** Based on '{first_note.title}', "
            "which mechanism is most responsible for long-term memory retention?\n"
            "A) Passive re-reading  B) Spaced repetition ✓  C) Highlighting  D) Summarising once"
            if first_note else ""
        )

        return (
            "## Key Insights\n"
            + "\n".join(insight_titles)
            + "\n\n## Connections You Might Have Missed\n"
            + conn_block
            + "\n\n## Today's Quiz\n"
            + quiz
            + "\n\n---\n*🔑 Mock digest — set `OPENAI_API_KEY` for AI-generated content.*"
        )
