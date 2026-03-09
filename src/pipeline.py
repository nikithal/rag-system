"""
pipeline.py - Phase 1
Ties everything together: retrieve → prompt → LLM → answer with citations.
Using Groq (free) instead of OpenAI.
"""

import os
import yaml
from dotenv import load_dotenv
from groq import Groq
from src.retrieval import Retriever, format_context

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LLM_MODEL    = "llama-3.3-70b-versatile"   # free on Groq, excellent quality
PROMPTS_FILE = "prompts/rag_prompt.yaml"
# ──────────────────────────────────────────────────────────────────────────────


def load_prompts(path: str = PROMPTS_FILE) -> dict:
    """Load prompts from YAML config — never hardcode prompts in Python."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks for the query
    2. Format them into a context string
    3. Build the prompt from our YAML config
    4. Call Groq (free Llama 3.3) and return the answer
    """

    def __init__(self):
        self.retriever = Retriever()
        self.client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.prompts   = load_prompts()
        print(f"✅ RAG Pipeline ready. Using model: {LLM_MODEL}\n")

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        Ask a question. Returns a dict with:
        - answer:      the LLM's response with citations
        - chunks:      the retrieved source chunks
        - context:     the formatted context string sent to the LLM
        - tokens_used: how many tokens were consumed
        """
        print(f"🔍 Retrieving chunks for: '{question}'")
        chunks = self.retriever.retrieve(question, top_k=top_k)

        if not chunks:
            return {
                "answer":  "I don't have enough information in the provided documents to answer this question.",
                "chunks":  [],
                "context": ""
            }

        # Format chunks into a readable context block
        context = format_context(chunks)

        # Build prompt from YAML config
        prompt_config = self.prompts["rag_prompt"]
        system_msg    = prompt_config["system"]
        user_msg      = prompt_config["user"].format(context=context, question=question)

        print(f"🤖 Calling {LLM_MODEL} via Groq...")

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
            "answer":      answer,
            "chunks":      chunks,
            "context":     context,
            "tokens_used": response.usage.total_tokens,
        }


def print_result(result: dict):
    """Pretty-print the result to the terminal."""
    print("\n" + "=" * 60)
    print("📋 ANSWER:")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "=" * 60)
    print("📚 SOURCE CHUNKS USED:")
    print("=" * 60)
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"\n[Chunk {i}] | source: {chunk['source']} | relevance: {chunk['score']:.3f}")
        print(f"  {chunk['text'][:300]}...")
    if "tokens_used" in result:
        print(f"\n💰 Tokens used: {result['tokens_used']} (FREE on Groq!)")
    print("=" * 60)


if __name__ == "__main__":
    rag = RAGPipeline()

    test_questions = [
        "What is the attention mechanism and how does it work?",
        "What were the BLEU scores achieved by the Transformer model?",
        "What is the capital of France?",   # Should trigger 'no evidence' response
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        result = rag.ask(question)
        print_result(result)
        input("\nPress Enter for next question...")
