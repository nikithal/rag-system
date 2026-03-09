"""
setup_files.py
Run this once to populate all project files automatically.
Usage: python setup_files.py
"""

import os

files = {}

# ── 1. prompts/rag_prompt.yaml ────────────────────────────────────────────────
files["prompts/rag_prompt.yaml"] = '''\
version: "1.0"

rag_prompt:
  system: |
    You are a precise assistant that answers questions based ONLY on the provided context.

    Rules you must follow:
    1. Only use information from the context below to answer.
    2. Always cite your source by referencing the chunk number, e.g. [Chunk 1], [Chunk 3].
    3. If the context does not contain enough information to answer, respond with:
       "I don\'t have enough information in the provided documents to answer this question."
    4. Never make up information or use knowledge outside the provided context.
    5. Keep answers concise and factual.

  user: |
    Context:
    {context}

    Question: {question}

    Answer (with citations):
'''

# ── 2. src/__init__.py ────────────────────────────────────────────────────────
files["src/__init__.py"] = ""

# ── 3. src/ingest.py ─────────────────────────────────────────────────────────
files["src/ingest.py"] = '''\
"""
ingest.py - Phase 1
Reads PDFs, chunks them, embeds with Sentence Transformers, stores in ChromaDB.
"""

import os
from pathlib import Path
from typing import List

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHROMA_PATH   = "chroma_db"
COLLECTION_NAME = "rag_documents"


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\\n"
    print(f"  Loaded PDF: {Path(path).name} ({len(reader.pages)} pages)")
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    chunk_words   = int(chunk_size * 0.75)
    overlap_words = int(overlap * 0.75)
    start = 0
    while start < len(words):
        end   = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap_words
    print(f"  Chunked into {len(chunks)} pieces")
    return chunks


def embed_and_store(chunks: List[str], source_name: str):
    print(f"  Loading embedding model: {EMBED_MODEL} ...")
    embedder   = SentenceTransformer(EMBED_MODEL)
    print(f"  Embedding {len(chunks)} chunks ...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    ids       = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
    print(f"  Stored {len(chunks)} chunks in ChromaDB.")
    return collection


def ingest_document(pdf_path: str):
    print(f"\\n📄 Ingesting: {pdf_path}")
    source_name = Path(pdf_path).stem
    text        = load_pdf(pdf_path)
    chunks      = chunk_text(text)
    collection  = embed_and_store(chunks, source_name)
    print(f"✅ Done! \'{source_name}\' is now searchable.\\n")
    return collection


def ingest_all(data_dir: str = "data"):
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in \'{data_dir}/\' folder.")
        return
    print(f"Found {len(pdf_files)} PDF(s) to ingest:")
    for pdf_path in pdf_files:
        ingest_document(str(pdf_path))
    print(f"🎉 All documents ingested into ChromaDB at \'{CHROMA_PATH}/\'")


if __name__ == "__main__":
    ingest_all()
'''

# ── 4. src/retrieval.py ───────────────────────────────────────────────────────
files["src/retrieval.py"] = '''\
"""
retrieval.py - Phase 1
Retrieves the most relevant chunks for a query using vector similarity search.
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBED_MODEL     = "all-MiniLM-L6-v2"
CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "rag_documents"
TOP_K           = 5


class Retriever:
    def __init__(self):
        print("Loading retriever...")
        self.embedder   = SentenceTransformer(EMBED_MODEL)
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"✅ Retriever ready. Collection has {self.collection.count()} chunks.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text":        results["documents"][0][i],
                "source":      results["metadatas"][0][i]["source"],
                "chunk_index": results["metadatas"][0][i]["chunk_index"],
                "score":       1 - results["distances"][0][i],
            })
        return chunks


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i}] (Source: {chunk[\'source\']}, relevance: {chunk[\'score\']:.2f})\\n"
            f"{chunk[\'text\']}"
        )
    return "\\n\\n---\\n\\n".join(parts)


if __name__ == "__main__":
    retriever = Retriever()
    query     = "What is the attention mechanism?"
    print(f"\\nQuery: \'{query}\'\\n")
    chunks = retriever.retrieve(query)
    for i, chunk in enumerate(chunks, 1):
        print(f"[Chunk {i}] score={chunk[\'score\']:.3f} | source={chunk[\'source\']}")
        print(f"  {chunk[\'text\'][:200]}...\\n")
'''

# ── 5. src/pipeline.py ────────────────────────────────────────────────────────
files["src/pipeline.py"] = '''\
"""
pipeline.py - Phase 1
Ties everything together: retrieve → prompt → LLM → answer with citations.
"""

import os
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from src.retrieval import Retriever, format_context

load_dotenv()

LLM_MODEL    = "gpt-4o-mini"
PROMPTS_FILE = "prompts/rag_prompt.yaml"


def load_prompts(path: str = PROMPTS_FILE) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.prompts   = load_prompts()
        print("✅ RAG Pipeline ready.\\n")

    def ask(self, question: str, top_k: int = 5) -> dict:
        print(f"🔍 Retrieving chunks for: \'{question}\'")
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "I don\'t have enough information in the provided documents to answer this question.",
                "chunks": [],
                "context": ""
            }

        context       = format_context(chunks)
        prompt_config = self.prompts["rag_prompt"]
        system_msg    = prompt_config["system"]
        user_msg      = prompt_config["user"].format(context=context, question=question)

        print(f"🤖 Calling {LLM_MODEL}...")
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0,
            max_tokens=500,
        )
        answer = response.choices[0].message.content
        return {
            "answer":       answer,
            "chunks":       chunks,
            "context":      context,
            "tokens_used":  response.usage.total_tokens,
        }


def print_result(result: dict):
    print("\\n" + "="*60)
    print("📋 ANSWER:")
    print("="*60)
    print(result["answer"])
    print("\\n" + "="*60)
    print("📚 SOURCE CHUNKS USED:")
    print("="*60)
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"\\n[Chunk {i}] | source: {chunk[\'source\']} | relevance: {chunk[\'score\']:.3f}")
        print(f"  {chunk[\'text\'][:300]}...")
    if "tokens_used" in result:
        cost = result[\'tokens_used\'] * 0.00000015
        print(f"\\n💰 Tokens used: {result[\'tokens_used\']} (~${cost:.5f})")
    print("="*60)


if __name__ == "__main__":
    rag = RAGPipeline()
    questions = [
        "What is the attention mechanism and how does it work?",
        "What were the BLEU scores achieved by the Transformer model?",
        "What is the capital of France?",
    ]
    for question in questions:
        print(f"\\n{\'=\'*60}")
        print(f"❓ Question: {question}")
        result = rag.ask(question)
        print_result(result)
        input("\\nPress Enter for next question...")
'''

# ── 6. app.py ─────────────────────────────────────────────────────────────────
files["app.py"] = '''\
"""
app.py — Streamlit UI for the RAG system.
Run with: streamlit run app.py
"""

import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")
st.caption("Answers grounded in your documents — no hallucinations.")


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


try:
    rag = load_pipeline()
    st.success("✅ Documents loaded and ready to query!")
except Exception as e:
    st.error(f"❌ Error loading pipeline: {e}")
    st.info("Make sure you have run `python -m src.ingest` first.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 5)
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Your question is embedded into a vector")
    st.markdown("2. ChromaDB finds the most similar document chunks")
    st.markdown("3. Chunks are sent to GPT-4o-mini with a strict citation prompt")
    st.markdown("4. The model answers ONLY from the retrieved evidence")

question = st.text_input("Ask a question about your documents:",
                          placeholder="e.g. What is the attention mechanism?")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating answer..."):
        result = rag.ask(question, top_k=top_k)

    st.markdown("## 💬 Answer")
    st.markdown(result["answer"])

    if "tokens_used" in result:
        cost = result["tokens_used"] * 0.00000015
        st.caption(f"Tokens: {result[\'tokens_used\']} | Cost: ~${cost:.5f}")

    st.markdown("---")
    with st.expander(f"📄 View {len(result[\'chunks\'])} source chunks", expanded=False):
        for i, chunk in enumerate(result["chunks"], 1):
            st.markdown(f"**[Chunk {i}]** | Source: `{chunk[\'source\']}` | Relevance: `{chunk[\'score\']:.3f}`")
            st.text_area("", value=chunk["text"][:600] + "...", height=120,
                         key=f"chunk_{i}", disabled=True)
            st.markdown("---")

with st.expander("💡 Sample questions to try"):
    st.markdown("""
- What is the attention mechanism and how does it work?
- What problem does the Transformer architecture solve?
- What were the BLEU scores achieved by the model?
- How does multi-head attention differ from single-head attention?
- What is positional encoding and why is it needed?
- What is the capital of France? *(should decline — tests citation enforcement)*
    """)
'''

# ── 7. .gitignore ─────────────────────────────────────────────────────────────
files[".gitignore"] = """\
.env
venv/
__pycache__/
*.pyc
chroma_db/
data/*.pdf
"""

# ── 8. requirements.txt ───────────────────────────────────────────────────────
files["requirements.txt"] = """\
langchain
langchain-community
langchain-openai
chromadb
pypdf
sentence-transformers
rank_bm25
python-dotenv
streamlit
ragas
openai
pyyaml
"""

# ── Write all files ───────────────────────────────────────────────────────────
print("\n🚀 Populating project files...\n")
for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  ✅ Written: {filepath}")

print("\n🎉 All files populated! Next steps:")
print("  1. Add your OpenAI key to .env")
print("  2. pip install -r requirements.txt")
print("  3. python -m src.ingest")
print("  4. streamlit run app.py\n")
