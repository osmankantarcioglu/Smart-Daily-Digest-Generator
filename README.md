# Smart Daily Digest — AI-Powered Note Resurfacing

> **Technical supplement** to the AppNation New Grad Product Specialist case study.
>
> The case proposed *Smart Daily Digest* as a high-impact feature for NotebookAI.
> This repository is the **working proof** of that proposal — every hypothesis made
> in the case is validated here with real code, real embeddings, and real LLM calls.

---

## What Is This?

NotebookAI users write notes, then forget them. The **post-capture engagement gap**
is the biggest white space in the note-taking category — and no competitor has solved it.

**Smart Daily Digest** fixes this: every morning, an AI analyses the user's entire
note library and delivers:

- **3–5 key insights** surfaced from recent notes
- **2 cross-note connections** the user may have missed
- **1 micro-quiz** for spaced-repetition reinforcement

This repo implements the full technical pipeline behind that feature.

---

## Live Demo

```bash
cd notebook-ai-smart-digest
pip install -r requirements.txt
streamlit run demo/app.py
```

Opens at `http://localhost:8501`

> Works without an API key — LLM sections use mock responses.
> Add `OPENAI_API_KEY` to `.env` for real AI-generated content.

---

## Architecture

```
User Notes
    │
    ▼
NoteEmbedder          sentence-transformers (all-MiniLM-L6-v2, free, local)
    │
    ▼
NoteVectorStore       FAISS IndexFlatIP — cosine similarity on normalised vectors
    │
    ├──► ConnectionFinder    Cross-note semantic discovery
    │
    └──► DigestGenerator     RAG: retrieved notes → LLM prompt → daily digest
              │
              └──► QuizGenerator   Spaced-repetition quiz card


ModelRouter
    └──► ComplexityClassifier   Scores query 0 → 1
              │
              ├── score < 0.35  →  SIMPLE  →  gpt-4o-mini  ($0.15 / 1M tokens)
              ├── score < 0.65  →  MEDIUM  →  gpt-4o-mini  ($0.15 / 1M tokens)
              └── score ≥ 0.65  →  COMPLEX →  gpt-4o       ($5.00 / 1M tokens)
```

---

## Project Structure

```
notebook-ai-smart-digest/
├── pipeline/
│   ├── embedder.py            Note → 384-dim vector (sentence-transformers)
│   ├── vector_store.py        FAISS index + Note dataclass + k-NN search
│   ├── connection_finder.py   Pairwise cosine similarity → cross-note links
│   ├── digest_generator.py    RAG prompt builder + OpenAI call (mock fallback)
│   └── quiz_generator.py      Spaced-repetition quiz generator (mock fallback)
├── model_router/
│   ├── complexity_classifier.py   Heuristic query scorer (sub-ms, no GPU)
│   └── router.py                  Routing + per-session cost tracking
├── demo/
│   └── app.py                 Two-tab Streamlit UI
├── data/
│   └── sample_notes.json      12 realistic notes (AI, product, learning)
├── notebook.ipynb             Analysis: t-SNE, heatmap, RAG eval, cost sim
├── requirements.txt
└── .env.example
```

---

## Demo Walkthrough

### Tab 1 — Smart Daily Digest

1. Toggle **Use sample notes** ON (12 pre-loaded notes)
2. Click **Generate My Digest**
3. The pipeline runs automatically:
   - Notes are embedded into 384-dim vectors
   - FAISS index is built in memory
   - Cross-note connections are found via cosine similarity
   - GPT-4o-mini generates the digest via RAG

**Expected output:**

```
🔗 Cross-Note Connections Discovered
  RAG Architecture ↔ Vector Embeddings          [highly related    | 0.84]
  Mobile App Retention ↔ Habit Formation         [conceptually linked | 0.71]
  LLM Cost Optimization ↔ On-Device AI          [conceptually linked | 0.66]

📋 Your Daily Digest
  ## Key Insights
  • 70% of information is forgotten within 24h without reinforcement (Ebbinghaus)
  • RAG grounds LLM responses in your own documents, cutting hallucinations
  • Model routing reduces inference costs by up to 70%
  ...

  ## Connections You Might Have Missed
  Mobile App Retention & Freemium Conversion share a key insight:
  daily value delivery drives both habit formation and upgrade motivation...

🧠 Today's Quiz
  What is the primary benefit of spaced repetition systems (SRS)?
  A) Reduces the need for notes
  B) Increases long-term retention by scheduling reviews ✓
  C) Enhances data visualization
  D) Decreases time spent studying
```

---

### Tab 2 — Model Router

Type any query and click **Classify Query** to see:

| Query | Complexity | Model | Cost vs GPT-4o |
|---|---|---|---|
| `"Summarize my last note"` | 🟢 SIMPLE | gpt-4o-mini | −97% |
| `"What should I review today?"` | 🟡 MEDIUM | gpt-4o-mini | −97% |
| `"Analyze all my notes and find patterns"` | 🔴 COMPLEX | gpt-4o | baseline |

Click **Run Batch (20 queries)** to see aggregate cost savings with a distribution chart.

---

## Notebook Analysis

`notebook.ipynb` runs five analyses and saves charts:

| Output | What It Shows |
|---|---|
| `tsne_notes.png` | 2D embedding space — topic clusters visible |
| `similarity_heatmap.png` | Pairwise cosine similarity (12×12) |
| `model_router_cost.png` | Cost: always-GPT-4o vs routed (~64% saving) |
| `complexity_distribution.png` | Query score histogram by class |
| `retention_curves.png` | Simulated D30/D60/D90: baseline vs digest users (+8pp) |

Also includes a **RAG retrieval evaluation** — Precision@3 on 4 test queries.

---

## Cost Analysis

Simulated over 100 realistic NotebookAI queries:

| Scenario | Cost |
|---|---|| Always GPT-4o (baseline) | $0.250 |
| With model routing | ~$0.090 |
| **Saving** | **~64%** |

---

## Relation to Case Study

| Case Study | Implementation |
|---|---|
| Task 2A — Smart Daily Digest feature | `pipeline/` — full RAG pipeline |
| Task 2A — Validation plan | `notebook.ipynb` Section 5 — RAG Precision@3 |
| Task 2B — Retention impact | `notebook.ipynb` Section 7 — retention curve sim |
| Task 3 — Model routing trade-off | `model_router/` + `notebook.ipynb` Section 6 |
| Task 3 — Semantic caching | Groundwork in `vector_store.py` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Note embedding |
| `faiss-cpu` | Vector similarity search |
| `openai` | GPT-4o / GPT-4o-mini |
| `streamlit` | Demo UI |
| `scikit-learn` | t-SNE visualisation |
| `matplotlib / seaborn` | Notebook charts |

---

*Author: Osman Kantarcıoğlu*
*AppNation New Grad Product Specialist — Case Study Technical Supplement*
