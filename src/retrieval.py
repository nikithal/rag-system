"""
retrieval.py - Phase 2
Upgrades Phase 1 with:
  1. BM25 keyword search
  2. Hybrid search (BM25 + vector merged with Reciprocal Rank Fusion)
  3. Cross-encoder re-ranker for precision improvement
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL     = "all-MiniLM-L6-v2"        # same embedding model as Phase 1
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # free, runs locally
CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "rag_documents"

TOP_K_INITIAL   = 20    # how many chunks to fetch before re-ranking
TOP_K_FINAL     = 5     # how many chunks to return after re-ranking
RRF_K           = 60    # RRF constant (standard value, don't change)
# ──────────────────────────────────────────────────────────────────────────────


class Retriever:
    """
    Phase 2 Retriever:
    - BM25 keyword search (finds exact terms)
    - Vector semantic search (finds meaning)
    - Hybrid merge with Reciprocal Rank Fusion
    - Cross-encoder re-ranker for final precision boost
    """

    def __init__(self):
        print("Loading retriever...")

        # ── Embedding model (same as Phase 1) ─────────────────────────────────
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # ── Cross-encoder re-ranker (NEW in Phase 2) ──────────────────────────
        # Downloads ~85MB on first run, cached forever after
        print(f"  Loading re-ranker: {RERANK_MODEL} ...")
        self.reranker = CrossEncoder(RERANK_MODEL)

        # ── ChromaDB connection ───────────────────────────────────────────────
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)

        # ── Load ALL chunks into memory for BM25 ─────────────────────────────
        # BM25 needs all documents loaded upfront to build its index
        # This is fine because our corpus is small (17 chunks)
        print("  Building BM25 index...")
        all_data        = self.collection.get(include=["documents", "metadatas"])
        self.all_chunks = all_data["documents"]          # list of all chunk texts
        self.all_metas  = all_data["metadatas"]          # list of all metadata dicts

        # Tokenise chunks for BM25 (split into words)
        tokenised       = [chunk.lower().split() for chunk in self.all_chunks]
        self.bm25       = BM25Okapi(tokenised)

        total = self.collection.count()
        print(f"✅ Retriever ready. Collection: {total} chunks | BM25 index built | Re-ranker loaded\n")

    # ── Step 1: BM25 keyword search ───────────────────────────────────────────
    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """
        Find chunks containing the exact keywords from the query.
        Returns list of {text, source, chunk_index, bm25_score}.
        """
        tokenised_query = query.lower().split()
        scores          = self.bm25.get_scores(tokenised_query)

        # Pair each chunk with its BM25 score and sort highest first
        scored = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in scored:
            results.append({
                "text":        self.all_chunks[idx],
                "source":      self.all_metas[idx]["source"],
                "chunk_index": self.all_metas[idx]["chunk_index"],
                "bm25_score":  float(score),
            })
        return results

    # ── Step 2: Vector semantic search ───────────────────────────────────────
    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """
        Find chunks with similar meaning to the query using embeddings.
        Same as Phase 1 retrieval.
        Returns list of {text, source, chunk_index, vector_score}.
        """
        query_embedding = self.embedder.encode(query).tolist()
        results         = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text":         results["documents"][0][i],
                "source":       results["metadatas"][0][i]["source"],
                "chunk_index":  results["metadatas"][0][i]["chunk_index"],
                "vector_score": 1 - results["distances"][0][i],
            })
        return chunks

    # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────────────
    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[dict],
        vector_results: list[dict]
    ) -> list[dict]:
        """
        Merge BM25 and vector results into one ranked list.

        RRF formula: score = 1/(rank + K) for each list
        A chunk appearing high in BOTH lists gets a high combined score.

        Example:
          Chunk 7: rank 1 in BM25  → 1/(1+60) = 0.0164
                   rank 2 in vector → 1/(2+60) = 0.0161
                   combined = 0.0325  ← high score, floats to top

          Chunk 5: rank 8 in BM25  → 1/(8+60) = 0.0147
                   rank 15 in vector → 1/(15+60) = 0.0133
                   combined = 0.0280  ← lower score
        """
        rrf_scores = {}   # key: chunk text, value: running RRF score
        chunk_data = {}   # key: chunk text, value: chunk metadata

        # Score from BM25 rankings
        for rank, chunk in enumerate(bm25_results):
            key = chunk["text"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + RRF_K)
            chunk_data[key] = chunk

        # Score from vector rankings
        for rank, chunk in enumerate(vector_results):
            key = chunk["text"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + RRF_K)
            chunk_data[key] = chunk

        # Sort by combined RRF score (highest first)
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        merged = []
        for key in sorted_keys:
            chunk            = dict(chunk_data[key])
            chunk["rrf_score"] = rrf_scores[key]
            merged.append(chunk)

        return merged

    # ── Step 4: Cross-encoder re-ranking ─────────────────────────────────────
    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Re-score chunks by reading the query AND chunk together.

        Unlike vector search (which scores chunks independently),
        the cross-encoder asks: "Does this specific chunk answer
        this specific question?" — much more precise.

        Input:  top 20 chunks from hybrid search
        Output: top 5 chunks re-ordered by actual relevance
        """
        if not chunks:
            return chunks

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs  = [(query, chunk["text"]) for chunk in chunks]

        # Cross-encoder scores each pair together
        scores = self.reranker.predict(pairs)

        # Attach re-ranker scores to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by re-ranker score (highest = most relevant to THIS query)
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:TOP_K_FINAL]   # return only top 5

    # ── Main retrieve method (used by pipeline.py) ────────────────────────────
    def retrieve(self, query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
        """
        Full Phase 2 retrieval pipeline:
          1. BM25 search    → top 20 keyword matches
          2. Vector search  → top 20 semantic matches
          3. RRF merge      → combined ranked list
          4. Re-rank        → cross-encoder re-scores top 20
          5. Return top 5

        The score shown in the UI is the re-ranker score.
        Higher = more relevant to your specific question.
        """
        print(f"  [BM25]   searching for keywords...")
        bm25_results   = self._bm25_search(query, top_k=TOP_K_INITIAL)

        print(f"  [Vector] searching for semantic meaning...")
        vector_results = self._vector_search(query, top_k=TOP_K_INITIAL)

        print(f"  [RRF]    merging {len(bm25_results)} BM25 + {len(vector_results)} vector results...")
        merged         = self._reciprocal_rank_fusion(bm25_results, vector_results)

        print(f"  [Rerank] cross-encoder re-scoring top {min(TOP_K_INITIAL, len(merged))} chunks...")
        final          = self._rerank(query, merged[:TOP_K_INITIAL])

        # Use rerank_score as the main score shown in the UI
        for chunk in final:
            chunk["score"] = chunk["rerank_score"]

        print(f"  ✅ Retrieved {len(final)} chunks after hybrid search + re-ranking")
        return final


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context string for the prompt.
    Same as Phase 1 — pipeline.py uses this unchanged.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i}] (Source: {chunk['source']}, "
            f"relevance: {chunk['score']:.2f})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Quick test: compare Phase 1 vs Phase 2 on a hard query
    retriever = Retriever()

    query = "What BLEU scores did the Transformer achieve on WMT 2014?"
    print(f"\nQuery: '{query}'\n")
    print("=" * 60)

    chunks = retriever.retrieve(query)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {i}]")
        print(f"  Re-rank score: {chunk.get('rerank_score', 0):.4f}")
        print(f"  Source: {chunk['source']} | chunk_index: {chunk['chunk_index']}")
        print(f"  Text preview: {chunk['text'][:200]}...")
