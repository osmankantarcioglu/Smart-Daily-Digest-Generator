"""
Smart Daily Digest — Streamlit Demo
====================================
Two-tab interactive demo:

  Tab 1 · Smart Daily Digest
      Add notes → Build semantic index → Generate AI digest + quiz

  Tab 2 · Model Router
      Enter any query → See complexity classification + cost analysis
      Run a batch of sample queries → View cost-saving breakdown chart

Run:
    cd notebook-ai-smart-digest
    streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env automatically (API key set here overrides sidebar input)
load_dotenv(ROOT / ".env")

from pipeline.embedder import NoteEmbedder
from pipeline.vector_store import Note, NoteVectorStore
from pipeline.connection_finder import ConnectionFinder
from pipeline.digest_generator import DigestGenerator
from pipeline.quiz_generator import QuizGenerator
from model_router.complexity_classifier import ComplexityClassifier, QueryComplexity
from model_router.router import ModelRouter

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NotebookAI · Smart Daily Digest",
    page_icon="📓",
    layout="wide",
)

# ── Shared state helpers ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> NoteEmbedder:
    return NoteEmbedder()


def _load_sample_notes() -> list[Note]:
    path = ROOT / "data" / "sample_notes.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Note.from_dict(d) for d in raw]


def _build_store(notes: list[Note], embedder: NoteEmbedder) -> NoteVectorStore:
    store = NoteVectorStore(dim=embedder.dim)
    texts = [n.to_embed_text() for n in notes]
    embeddings = embedder.embed(texts)
    store.add(notes, embeddings)
    return store


# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .digest-card {
        background: #1e1e3a;
        border-left: 4px solid #818cf8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #e2e8f0 !important;
    }
    .digest-card h1, .digest-card h2, .digest-card h3,
    .digest-card p, .digest-card li, .digest-card span {
        color: #e2e8f0 !important;
    }
    .connection-card {
        background: #0f2a1f;
        border-left: 4px solid #34d399;
        border-radius: 8px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.6rem;
        color: #d1fae5 !important;
    }
    .connection-card b, .connection-card small {
        color: #d1fae5 !important;
    }
    .quiz-card {
        background: #2a1f00;
        border-left: 4px solid #fbbf24;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #fef3c7 !important;
    }
    .quiz-card b {
        color: #fef3c7 !important;
    }
    .router-badge-simple  { color: #34d399; font-weight: 700; }
    .router-badge-medium  { color: #fbbf24; font-weight: 700; }
    .router-badge-complex { color: #f87171; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/notebook.png", width=64)
    st.title("NotebookAI")
    st.caption("Smart Daily Digest · Demo")
    st.divider()

    if os.getenv("OPENAI_API_KEY"):
        st.success("✅ API key loaded from .env", icon="🔑")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-…  leave blank for mock mode",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.markdown(
        "**What this demo shows:**\n"
        "- RAG pipeline (embed → index → retrieve)\n"
        "- Cross-note semantic connections\n"
        "- AI digest + spaced-repetition quiz\n"
        "- Model router with cost analysis\n\n"
        "_Technical implementation of the AppNation case study_"
    )

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

tab_digest, tab_router = st.tabs(["📓 Smart Daily Digest", "⚡ Model Router"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — SMART DAILY DIGEST
# ────────────────────────────────────────────────────────────────────────────

with tab_digest:
    st.header("Smart Daily Digest")
    st.caption(
        "Every morning NotebookAI analyses your full note library and delivers "
        "personalised insights, cross-note connections, and a retention quiz."
    )

    embedder = get_embedder()

    # ── Note management ──────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Your Notes")

        use_samples = st.toggle("Use sample notes (12 pre-loaded)", value=True)

        if "user_notes" not in st.session_state:
            st.session_state.user_notes = []

        if use_samples:
            notes = _load_sample_notes()
            st.info(f"📚 {len(notes)} sample notes loaded.", icon="📚")
        else:
            notes = list(st.session_state.user_notes)
            with st.expander("➕ Add a note", expanded=len(notes) == 0):
                with st.form("add_note", clear_on_submit=True):
                    title = st.text_input("Title")
                    content = st.text_area("Content", height=120)
                    submitted = st.form_submit_button("Add note")
                    if submitted and title and content:
                        new_note = Note(
                            id=f"user_{len(notes):03d}",
                            title=title,
                            content=content,
                            created_at="2025-03-19T08:00:00",
                            tags=[],
                        )
                        st.session_state.user_notes.append(new_note)
                        st.rerun()

            if not notes:
                st.warning("Add at least 2 notes to generate a digest.")

        # Note list
        for note in notes[:8]:
            with st.expander(f"📝 {note.title}"):
                st.write(note.content[:300] + ("…" if len(note.content) > 300 else ""))
                st.caption(f"Tags: {', '.join(note.tags) if note.tags else '—'}")

    # ── Digest generation ────────────────────
    with col_right:
        st.subheader("Generate Digest")

        n_insights = st.slider("Notes to surface in digest", 3, min(8, len(notes)), 5)
        sim_threshold = st.slider("Connection similarity threshold", 0.40, 0.85, 0.50, 0.05)

        if st.button("🚀 Generate My Digest", type="primary", disabled=len(notes) < 2):
            with st.spinner("Building semantic index…"):
                store = _build_store(notes, embedder)

            with st.spinner("Finding cross-note connections…"):
                finder = ConnectionFinder(store, embedder, threshold=sim_threshold)
                # Use the top n_insights notes as seeds
                seed_notes = notes[:n_insights]
                connections = finder.find(seed_notes, max_connections=5)

            with st.spinner("Generating digest…"):
                generator = DigestGenerator()
                digest_text = generator.generate(seed_notes, connections)

            with st.spinner("Generating quiz…"):
                quiz_gen = QuizGenerator()
                quiz = quiz_gen.generate(seed_notes[0])

            st.session_state["digest"] = digest_text
            st.session_state["connections"] = connections
            st.session_state["quiz"] = quiz
            st.session_state["seed_notes"] = seed_notes

        # ── Results ──────────────────────────
        if "digest" in st.session_state:
            st.divider()

            # Connections
            st.subheader("🔗 Cross-Note Connections Discovered")
            conns = st.session_state["connections"]
            if conns:
                for c in conns:
                    st.markdown(
                        f'<div class="connection-card">'
                        f"<b>{c.source.title}</b> ↔ <b>{c.target.title}</b><br>"
                        f"<small>{c.label} &nbsp;|&nbsp; similarity: {c.score:.2f}</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No connections above threshold. Lower the slider to find more.")

            # Digest
            st.subheader("📋 Your Daily Digest")
            st.markdown(
                f'<div class="digest-card">{st.session_state["digest"]}</div>',
                unsafe_allow_html=True,
            )

            # Quiz
            st.subheader("🧠 Retention Quiz")
            quiz = st.session_state["quiz"]
            st.markdown(
                f'<div class="quiz-card"><b>{quiz["question"]}</b></div>',
                unsafe_allow_html=True,
            )
            answer = st.radio(
                "Your answer:",
                quiz["options"],
                label_visibility="collapsed",
            )
            if st.button("Check Answer"):
                correct_letter = quiz["correct"]
                selected_letter = answer[0]  # "A", "B", "C", "D"
                if selected_letter == correct_letter:
                    st.success(f"✅ Correct! {quiz['explanation']}")
                else:
                    st.error(
                        f"❌ Not quite. Correct answer: **{correct_letter}**\n\n"
                        f"{quiz['explanation']}"
                    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — MODEL ROUTER
# ────────────────────────────────────────────────────────────────────────────

with tab_router:
    st.header("⚡ AI Model Router")
    st.caption(
        "Classifies query complexity and routes to the cheapest capable model. "
        "Implements the cost-optimisation strategy from Task 3 of the case study."
    )

    classifier = ComplexityClassifier()

    col_a, col_b = st.columns([1, 1], gap="large")

    # ── Single query ─────────────────────────
    with col_a:
        st.subheader("Single Query Analysis")

        query_input = st.text_area(
            "Enter a query:",
            value="What is the main idea of my latest note?",
            height=100,
        )

        if st.button("🔍 Classify Query", type="primary"):
            result = classifier.classify(query_input)
            st.session_state["last_classification"] = result

        if "last_classification" in st.session_state:
            r = st.session_state["last_classification"]
            complexity_colors = {
                "simple": "🟢",
                "medium": "🟡",
                "complex": "🔴",
            }
            emoji = complexity_colors[r.complexity.value]

            st.divider()
            col_x, col_y, col_z = st.columns(3)
            col_x.metric("Complexity", f"{emoji} {r.complexity.value.upper()}")
            col_y.metric("Model Selected", r.recommended_model)
            col_z.metric("Cost vs GPT-4o", f"−{r.cost_saving_vs_premium:.0%}")

            st.metric("Complexity Score", f"{r.score:.2f} / 1.00")
            st.progress(r.score)

            with st.expander("Scoring breakdown"):
                st.json(r.reasoning)

            st.info(
                f"Est. latency: **{r.avg_latency_ms} ms** | "
                f"Cost: **${r.cost_per_1k:.5f} / 1K tokens**"
            )

    # ── Batch analysis ───────────────────────
    with col_b:
        st.subheader("Batch Cost Analysis")
        st.caption("Simulates 20 realistic NotebookAI queries to show aggregate savings.")

        SAMPLE_QUERIES = [
            # Simple
            "Summarise my last note.",
            "What is spaced repetition?",
            "Rewrite this paragraph to be shorter.",
            "Translate this note to Spanish.",
            "Make a bullet list of key points.",
            "Fix the grammar in my note.",
            "What is RAG?",
            # Medium
            "How does the Ebbinghaus curve relate to my study habits?",
            "What should I review today based on my notes?",
            "Give me 3 actionable takeaways from my recent notes.",
            "What topics have I been studying this week?",
            "Suggest tags for my note on transformer architecture.",
            # Complex
            "Analyze all my AI-related notes and find patterns in my thinking.",
            "Compare the retention strategies across my learning notes and recommend which I should prioritize.",
            "Why do my product strategy notes contradict my notes on habit formation? Synthesize a coherent view.",
            "Across all my notes, what are the 3 most important concepts I keep returning to?",
            "Evaluate the trade-offs between on-device and cloud inference based on my engineering notes.",
            "How do my notes on freemium conversion connect to my notes on habit loops? What does this imply for a growth strategy?",
            "Find every connection between my AI architecture notes and my product strategy notes.",
            "Give me a comprehensive analysis of everything I know about LLM cost optimization.",
        ]

        if st.button("▶ Run Batch (20 queries)"):
            router = ModelRouter()
            results = router.route_batch(SAMPLE_QUERIES)
            st.session_state["batch_results"] = results
            st.session_state["batch_stats"] = router.stats.summary()

        if "batch_results" in st.session_state:
            stats = st.session_state["batch_stats"]
            results = st.session_state["batch_results"]

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Queries", stats["total_queries"])
            c2.metric("Cost Reduction", f"{stats['cost_reduction_pct']:.1f}%")
            c3.metric(
                "Saved",
                f"${stats['baseline_cost_usd'] - stats['actual_cost_usd']:.4f}",
                delta=f"vs always GPT-4o",
            )

            # Distribution chart
            dist = stats["distribution"]
            try:
                import pandas as pd
                import altair as alt

                df = pd.DataFrame(
                    [
                        {"Complexity": k.capitalize(), "Queries": v, "Color": k}
                        for k, v in dist.items()
                    ]
                )
                color_scale = alt.Scale(
                    domain=["simple", "medium", "complex"],
                    range=["#16a34a", "#d97706", "#dc2626"],
                )
                chart = (
                    alt.Chart(df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Complexity:N", axis=alt.Axis(labelAngle=0)),
                        y="Queries:Q",
                        color=alt.Color("Color:N", scale=color_scale, legend=None),
                        tooltip=["Complexity", "Queries"],
                    )
                    .properties(title="Query Distribution by Complexity", height=200)
                )
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                st.bar_chart({k: v for k, v in dist.items()})

            # Per-query table
            with st.expander("View all 20 classifications"):
                complexity_emoji = {"simple": "🟢", "medium": "🟡", "complex": "🔴"}
                for r in results:
                    c = r.classification
                    em = complexity_emoji[c.complexity.value]
                    st.markdown(
                        f"{em} **{c.complexity.value.upper()}** → `{c.recommended_model}` "
                        f"| _{c.query[:80]}…_"
                    )
