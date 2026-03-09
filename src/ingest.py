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
            full_text += text + "\n"
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
    print(f"\n📄 Ingesting: {pdf_path}")
    source_name = Path(pdf_path).stem
    text        = load_pdf(pdf_path)
    chunks      = chunk_text(text)
    collection  = embed_and_store(chunks, source_name)
    print(f"✅ Done! '{source_name}' is now searchable.\n")
    return collection


def ingest_all(data_dir: str = "data"):
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDFs found in '{data_dir}/' folder.")
        return
    print(f"Found {len(pdf_files)} PDF(s) to ingest:")
    for pdf_path in pdf_files:
        ingest_document(str(pdf_path))
    print(f"🎉 All documents ingested into ChromaDB at '{CHROMA_PATH}/'")


if __name__ == "__main__":
    ingest_all()
