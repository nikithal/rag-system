# rag-system

A production-grade retrieval-augmented generation (RAG) system for domain-specific document Q&A.
Built to bridge the gap between a working demo and an enterprise-ready system.

---

## Overview

This system ingests documents, retrieves relevant context using hybrid search, and generates
answers grounded strictly in retrieved evidence. The model declines to answer when supporting
evidence is absent.

Tested on the "Attention Is All You Need" paper (Vaswani et al., 2017).

---

## Architecture

```
Query
  ├── BM25 keyword search
  └── Vector semantic search
          │
          └── Reciprocal Rank Fusion (RRF)
                    │
                    └── Cross-encoder re-ranking
                                │
                                └── LLM generation with citation enforcement
```

---

## Implementation

### Phase 1 — Core pipeline

- PDF ingestion with pypdf
- Chunking: 500–800 tokens, 100-token overlap to avoid splitting sentences mid-context
- Embeddings: all-MiniLM-L6-v2 via Sentence Transformers (local, no API)
- Vector store: ChromaDB (persistent local store)
- Prompt enforces citation and declines response when evidence is missing

### Phase 2 — Retrieval quality

- Hybrid retrieval: BM25 (keyword) + vector (semantic) merged with Reciprocal Rank Fusion
- Re-ranking: cross-encoder/ms-marco-MiniLM-L-6-v2 re-scores top 20 candidates
- Result: final top 5 chunks selected by cross-encoder, not vector similarity alone
- Prompts versioned in prompts/rag_prompt.yaml — separated from application logic

### Phase 3 — Evaluation and CI/CD

- Golden dataset: 10 manually verified question-answer pairs
- RAGAS metrics: faithfulness, answer relevancy, context recall
- GitHub Actions: evaluation runs on every push, build fails if metrics drop below threshold

---

## Why hybrid search?

Pure vector search retrieves by semantic similarity, which works well for conceptual queries
but misses exact technical terms and specific numbers. BM25 handles exact matches but lacks
semantic understanding. Combining both with RRF consistently outperforms either alone.

The cross-encoder re-ranker reads the query and each candidate chunk together, which produces
more accurate relevance scores than the independent scoring used in vector search.

---

## Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Keyword search | rank-bm25 |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.3 70B via Groq API |
| Evaluation | RAGAS |
| CI/CD | GitHub Actions |

All models run locally except the LLM inference (Groq free tier).

---

## Setup

```bash
git clone https://github.com/nikithal/rag-system.git
cd rag-system

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Get a free Groq API key at https://console.groq.com

```bash
# Place PDFs in data/
python -m src.ingest
streamlit run app.py
```

---

## Evaluation

```bash
python -m evals.run_evals
```

Metrics are checked against thresholds defined in run_evals.py.
Exit code 1 if any metric falls below threshold — used by the CI pipeline.

---

## Project structure

```
src/
  ingest.py        — chunking, embedding, ChromaDB ingestion
  retrieval.py     — hybrid search, RRF, cross-encoder re-ranking
  pipeline.py      — retrieval + LLM + citation enforcement
prompts/
  rag_prompt.yaml  — versioned prompt config
evals/
  golden_dataset.json  — ground truth Q&A pairs
  run_evals.py         — RAGAS evaluation script
.github/workflows/
  eval.yml         — CI/CD pipeline
app.py             — Streamlit interface
```

---

## Limitations

- Chunking is word-based, not token-based. Actual token counts may vary by model tokenizer.
- PDF tables are not parsed structurally. Tabular data may be garbled after extraction.
- Re-ranking adds latency (~1-2s) proportional to the number of candidates scored.
- RAGAS evaluation consumes LLM tokens. With Groq free tier, limit dataset to 5-10 examples per run.
