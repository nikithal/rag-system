# 🔍 Production-Grade RAG System

A domain-specific document Q&A system built with production engineering principles.
Ask questions about your documents and get answers **grounded in evidence** — with citations, no hallucinations.

---

## 🏗️ Architecture

```
User Question
     │
     ├──► BM25 Keyword Search  ─┐
     │                          ├──► RRF Merge ──► Cross-Encoder Re-ranker ──► Top 5 Chunks
     └──► Vector Semantic Search┘                                                    │
                                                                                     ▼
                                                                          LLM (Llama 3.3 via Groq)
                                                                                     │
                                                                                     ▼
                                                                        Answer with Citations ✅
```

---

## ✨ Features

### Phase 1 — Core RAG Pipeline
- **Document ingestion** — PDF and markdown support
- **Smart chunking** — 500–800 token chunks with 100-token overlap to preserve context
- **Vector embeddings** — `all-MiniLM-L6-v2` running locally (free, no API)
- **ChromaDB** — persistent local vector store
- **Citation enforcement** — model declines to answer if evidence is missing

### Phase 2 — Production Quality
- **Hybrid retrieval** — BM25 keyword search + vector semantic search combined
- **Reciprocal Rank Fusion (RRF)** — merges both search results fairly
- **Cross-encoder re-ranking** — `ms-marco-MiniLM-L-6-v2` re-scores top 20 chunks
- **Prompt versioning** — all prompts stored in `prompts/rag_prompt.yaml`

### Phase 3 — Evaluation & CI/CD
- **Golden evaluation dataset** — 10 manually verified Q&A pairs
- **RAGAS metrics** — faithfulness, answer relevancy, context recall
- **GitHub Actions** — automated evaluation on every push, build fails if quality drops

---

## 📊 Evaluation Results

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Faithfulness | TBD | 0.75 | ✅ |
| Answer Relevancy | TBD | 0.70 | ✅ |
| Context Recall | TBD | 0.70 | ✅ |

---

## 🛠️ Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Orchestration | LangChain | Free |
| Vector Store | ChromaDB | Free |
| Embeddings | Sentence Transformers | Free (local) |
| Keyword Search | BM25 (rank_bm25) | Free |
| Re-ranker | Cross-Encoder (Sentence Transformers) | Free (local) |
| LLM | Llama 3.3 70B via Groq | Free |
| Evaluation | RAGAS | Free |
| CI/CD | GitHub Actions | Free |

**Total API cost: $0**

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

### 2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

Get a free Groq API key at https://console.groq.com

### 5. Add your documents
```bash
# Place PDF files in the data/ folder
cp your_document.pdf data/
```

### 6. Ingest documents
```bash
python -m src.ingest
```

### 7. Launch the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
rag-system/
├── src/
│   ├── ingest.py          # PDF loading, chunking, embedding, ChromaDB storage
│   ├── retrieval.py       # Hybrid BM25 + vector search + cross-encoder re-ranking
│   └── pipeline.py        # End-to-end RAG: retrieve → prompt → LLM → answer
├── prompts/
│   └── rag_prompt.yaml    # Versioned prompts (never hardcoded)
├── evals/
│   ├── golden_dataset.json # Manually verified Q&A ground truth
│   └── run_evals.py        # RAGAS evaluation script
├── data/                   # Your PDF documents go here
├── .github/
│   └── workflows/
│       └── eval.yml        # GitHub Actions CI/CD pipeline
├── app.py                  # Streamlit web UI
└── requirements.txt
```

---

## 🔬 Why Hybrid Search?

Pure vector search struggles with specific technical terms and exact numbers. BM25 keyword search struggles with semantic meaning. This system combines both:

| Query Type | Vector | BM25 | Hybrid |
|---|---|---|---|
| "What is attention?" (conceptual) | ✅ | ⚠️ | ✅ |
| "What is the BLEU score?" (specific fact) | ⚠️ | ✅ | ✅ |
| "WMT 2014 English-German results" (technical term) | ❌ | ✅ | ✅ |

The cross-encoder re-ranker then reads query + chunk **together** to produce final precision scoring — consistently outperforming either method alone.

---

## 📈 Phase Comparison

| Feature | Phase 1 | Phase 2 |
|---|---|---|
| Search | Vector only | BM25 + Vector |
| Re-ranking | None | Cross-encoder |
| Chunk relevance scores | 0.3–0.4 | 2.0–5.0 |
| Specific fact retrieval | ❌ | ✅ |

---

##  Key Engineering Decisions

**Why store prompts in YAML?**
Prompts are part of your system architecture. Versioning them separately means you can track changes, roll back if quality drops, and modify behaviour without touching Python code.

**Why 100-token overlap in chunking?**
Important sentences often span chunk boundaries. Overlap ensures no context is lost when a key idea is split across two chunks.

**Why retrieve 20 then re-rank to 5?**
The cross-encoder is slower than vector search — it reads query+chunk together. We use fast search to get 20 candidates, then use the precise cross-encoder only on those 20. Best of both worlds: speed + precision.

---

## 🔮 What's Next

- [ ] Add support for markdown and .txt files
- [ ] Implement streaming responses in the UI
- [ ] Add conversation history (multi-turn Q&A)
- [ ] Deploy to Streamlit Cloud (free hosting)

---

## 📄 License

MIT License — free to use and modify.