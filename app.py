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
        st.caption(f"Tokens: {result['tokens_used']} | Cost: FREE (Groq) 🎉")

    st.markdown("---")
    with st.expander(f"📄 View {len(result['chunks'])} source chunks", expanded=False):
        for i, chunk in enumerate(result["chunks"], 1):
            st.markdown(f"**[Chunk {i}]** | Source: `{chunk['source']}` | Relevance: `{chunk['score']:.3f}`")
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
