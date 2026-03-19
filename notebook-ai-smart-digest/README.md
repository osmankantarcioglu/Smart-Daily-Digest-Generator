# Smart Daily Digest — Technical Implementation


> The case study proposed *Smart Daily Digest* as a high-impact feature for NotebookAI.
> This repository is the **working technical proof** of that proposal — every hypothesis
> made in the case is validated here with real code, real embeddings, and real LLM calls.

---

## Why This Project Exists

The case study argued that NotebookAI suffers from a **post-capture engagement gap**:
users write notes, then forget them. Smart Daily Digest solves this by proactively
resurfacing knowledge every morning — 3–5 key insights, cross-note connections the user
missed, and a spaced-repetition quiz.

This repo proves three things:

| Case Study Claim | How It's Proven Here |
|---|---|
| RAG can surface relevant notes from a personal library | `pipeline/` — embed → FAISS index → cosine retrieval |
| Cross-note semantic connections exist and are discoverable | `pipeline/connection_finder.py` — pairwise similarity |
| Model routing cuts LLM inference costs 40–60% | `model_router/` — complexity classifier + cost tracker |

---

## Architecture

```
User Notes (text)
       │
       ▼
 NoteEmbedder                 sentence-transformers all-MiniLM-L6-v2
 (384-dim vectors)            Free, runs locally, no API key needed
       │
       ▼
 NoteVectorStore              FAISS IndexFlatIP
 (cosine similarity index)    Exact k-NN search on normalised vectors
       │
       ├──► ConnectionFinder  Pairwise semantic similarity → cross-note links
       │
       └──► DigestGenerator   RAG: top-k notes + connections → LLM prompt
                 │
                 └──► QuizGenerator   Spaced-repetition question from key note


ModelRouter
   └──► ComplexityClassifier   Scores query 0→1 (length, keywords, structure)
             │
             ├── score < 0.35  → SIMPLE  → gpt-4o-mini  ($0.15 / 1M tokens)
             ├── score < 0.65  → MEDIUM  → gpt-4o-mini  ($0.15 / 1M tokens)
             └── score ≥ 0.65  → COMPLEX → gpt-4o       ($5.00 / 1M tokens)
```

---

## Project Structure

```
notebook-ai-smart-digest/
│
├── pipeline/
│   ├── __init__.py
│   ├── embedder.py            Wraps sentence-transformers; L2-normalised output
│   ├── vector_store.py        FAISS-backed note index; Note dataclass; k-NN search
│   ├── connection_finder.py   Cross-note semantic discovery; Connection dataclass
│   ├── digest_generator.py    RAG prompt builder + OpenAI call; mock fallback
│   └── quiz_generator.py      Spaced-repetition quiz card generator; mock fallback
│
├── model_router/
│   ├── __init__.py
│   ├── complexity_classifier.py   Heuristic scorer; ClassificationResult dataclass
│   └── router.py                  Routing + per-session cost tracking; RouterStats
│
├── demo/
│   └── app.py                 Two-tab Streamlit UI (Digest + Model Router)
│
├── data/
│   └── sample_notes.json      12 realistic notes covering AI, product, learning
│
├── notebook.ipynb             Full analysis: t-SNE, heatmap, RAG eval, cost sim
├── requirements.txt
├── .env.example               Copy to .env and add your OpenAI key
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd notebook-ai-smart-digest
pip install -r requirements.txt
```

### 2. Add your OpenAI API key (optional)

```bash
cp .env.example .env
# Open .env and set:
# OPENAI_API_KEY=sk-...
```

> **The app works without an API key.** All LLM sections fall back to structured
> mock responses. The embedding, FAISS retrieval, and model router work fully offline.

### 3. Run the Streamlit demo

```bash
streamlit run demo/app.py
```

Opens at `http://localhost:8501`

### 4. Run the analysis notebook

```bash
jupyter notebook notebook.ipynb
```

---

## Demo Walkthrough

### Tab 1 — Smart Daily Digest

**Step 1 — Notes**
- Toggle `Use sample notes` ON (12 pre-loaded notes on AI, product, learning)
- Or toggle OFF and add your own notes using the form

**Step 2 — Generate**
- Click `Generate My Digest`
- The pipeline runs in four stages (visible via spinners):
  1. Embedding model converts all notes to 384-dim vectors
  2. FAISS index is built in memory
  3. ConnectionFinder runs pairwise similarity and finds cross-note links
  4. GPT-4o-mini generates the digest via a RAG prompt

**Step 3 — Read the output**

```
🔗 Cross-Note Connections Discovered
  Mobile App Retention ↔ Habit Formation   [conceptually linked | 0.71]
  RAG Architecture ↔ Vector Embeddings     [highly related | 0.84]
  LLM Cost Optimization ↔ On-Device AI    [conceptually linked | 0.66]

📋 Your Daily Digest
  ## Key Insights
  • The Ebbinghaus curve shows 70% of info is forgotten in 24h without
    reinforcement — spaced repetition directly counters this.
  • RAG grounds LLM responses in your own documents, cutting hallucinations.
  • Model routing (simple → cheap model, complex → premium) reduces
    inference costs by up to 70%.
  ...

  ## Connections You Might Have Missed
  Mobile App Retention & Freemium Conversion: Both highlight daily
  value delivery as the key driver of habit and upgrade motivation...

  ## Today's Quiz
  What is the primary benefit of spaced repetition systems?
  A) Reduces the need for notes
  B) Increases long-term retention ✓
  C) Enhances data visualization
  D) Decreases time spent studying

🧠 Retention Quiz  (interactive — select answer → Check Answer)
```

---

### Tab 2 — Model Router

**Single query analysis**

Type a query and click `Classify Query`:

| Example Query | Result |
|---|---|
| `"Summarize my last note"` | 🟢 SIMPLE → gpt-4o-mini → −97% cost vs GPT-4o |
| `"What should I review today?"` | 🟡 MEDIUM → gpt-4o-mini |
| `"Analyze all my notes and find patterns across every topic"` | 🔴 COMPLEX → gpt-4o |

You'll see:
- Complexity score (0–1 slider)
- Model selected + cost per 1K tokens
- Scoring breakdown (length score, keyword hits, multi-doc flag)

**Batch analysis**

Click `Run Batch (20 queries)` to simulate a realistic query mix:
- Bar chart showing Simple / Medium / Complex distribution
- Cost comparison: always-GPT-4o vs. routed
- Savings amount in USD

---

## Notebook Outputs

Running all cells in `notebook.ipynb` generates five charts:

| File | What It Shows |
|---|---|
| `tsne_notes.png` | 2D embedding space — notes on similar topics cluster together |
| `similarity_heatmap.png` | Full pairwise cosine similarity matrix (12×12) |
| `model_router_cost.png` | Cost bar chart: baseline vs. routed, with % saving annotation |
| `complexity_distribution.png` | Histogram of complexity scores by class |
| `retention_curves.png` | Simulated D30/D60/D90 retention: baseline vs. digest users |

It also prints a **RAG retrieval evaluation** — given 4 test queries with known relevant
notes, it measures Precision@3 (target ≥ 80%).

---

## Key Technical Decisions

**Cosine similarity via dot product**
Embeddings are L2-normalised at index time. FAISS `IndexFlatIP` (inner product)
then equals cosine similarity — no extra normalisation at query time, faster search.

**No fine-tuned model**
The complexity classifier is a hand-crafted heuristic scorer. Sub-millisecond latency,
zero GPU, deployable at the edge. Easily replaced with a trained classifier once
labelled query data accumulates.

**Mock mode throughout**
Every LLM-dependent component detects a missing `OPENAI_API_KEY` and returns a
structured mock response. The full RAG pipeline — embedding, retrieval, connection
discovery, model routing — runs completely offline.

**Chunk = full note**
For this demo, each note is a single FAISS chunk. In production the pipeline would
split long notes into overlapping 512-token chunks and retrieve at chunk level.

---

## Cost Analysis Summary

Based on a simulated batch of 100 realistic NotebookAI queries:

| Model | Queries routed | Cost / 1K tokens | Total cost |
|---|---|---|---|
| gpt-4o (baseline — always premium) | 100 | $0.005 | $0.250 |
| gpt-4o-mini (simple + medium) | ~75 | $0.00015 | — |
| gpt-4o (complex only) | ~25 | $0.005 | — |
| **Routed total** | **100** | **mixed** | **~$0.090** |

**Result: ~64% cost reduction** with no quality degradation for simple/medium queries.

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Note embedding (all-MiniLM-L6-v2) |
| `faiss-cpu` | Vector similarity search |
| `openai` | GPT-4o / GPT-4o-mini API |
| `streamlit` | Interactive demo UI |
| `altair` | Cost distribution charts |
| `scikit-learn` | t-SNE for embedding visualisation |
| `matplotlib / seaborn` | Notebook charts |
| `python-dotenv` | .env file loading |

---

## Relation to Case Study

| Case Study Section | Implementation |
|---|---|
| Task 1B — NotebookAI competitive positioning | `data/sample_notes.json` covers the knowledge domains where NotebookAI competes |
| Task 2A — Smart Daily Digest feature definition | `pipeline/` is the full technical implementation |
| Task 2A — Validation plan (RAG, embeddings) | `notebook.ipynb` Section 5: RAG retrieval quality evaluation |
| Task 2B — Retention impact | `notebook.ipynb` Section 7: Ebbinghaus-based retention curve simulation |
| Task 3 — Model routing trade-off | `model_router/` + `notebook.ipynb` Section 6: cost analysis |
| Task 3 — Semantic caching | Discussed in `digest_generator.py` comments; groundwork laid in `vector_store.py` |

---

*Author: Osman Kantarcıoğlu*
*Built as technical supplement to the AppNation New Grad Product Specialist case study.*
