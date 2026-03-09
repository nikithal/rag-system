"""
run_evals.py - Phase 3
Measures RAG system quality using RAGAS metrics.
Configured to use Groq (free) instead of OpenAI for evaluation.
Run with: python -m evals.run_evals
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GOLDEN_DATASET_PATH     = "evals/golden_dataset.json"
RESULTS_PATH            = "evals/results.json"
FAITHFULNESS_THRESHOLD  = 0.75
RELEVANCY_THRESHOLD     = 0.70
RECALL_THRESHOLD        = 0.70
# ──────────────────────────────────────────────────────────────────────────────


def load_golden_dataset(path: str) -> list[dict]:
    """Load the manually verified Q&A pairs."""
    with open(path, "r") as f:
        return json.load(f)


def run_rag_on_dataset(golden_data: list[dict]) -> dict:
    """
    Run the full RAG pipeline on every question in the golden dataset.
    Collects: questions, answers, retrieved contexts, and ground truths.
    """
    from src.pipeline import RAGPipeline

    rag = RAGPipeline()

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    print(f"\n📋 Running RAG on {len(golden_data)} questions...\n")

    for i, item in enumerate(golden_data, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{len(golden_data)}] {question[:70]}...")

        result = rag.ask(question, top_k=5)
        # Small delay to avoid hitting Groq rate limits
        import time
        time.sleep(2)
        
        retrieved_contexts = [chunk["text"] for chunk in result["chunks"]]

        questions.append(question)
        answers.append(result["answer"])
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    }


def get_ragas_llm_and_embeddings():
    """
    Configure RAGAS to use Groq (free) instead of OpenAI.
    RAGAS needs an LLM to judge answers and an embedding model
    to measure semantic similarity.
    """
    print("  Configuring RAGAS to use Groq + local embeddings...")

    # Use Groq's Llama model as the judge LLM (free)
    groq_llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    # Use local HuggingFace embeddings (free, no API needed)
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Wrap them in RAGAS wrappers
    ragas_llm        = LangchainLLMWrapper(groq_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    return ragas_llm, ragas_embeddings


def evaluate_with_ragas(data: dict) -> dict:
    """
    Run RAGAS evaluation using Groq as the judge LLM.
    """
    print("\n🔍 Running RAGAS evaluation...\n")

    # Get free LLM + embeddings for RAGAS
    ragas_llm, ragas_embeddings = get_ragas_llm_and_embeddings()

    # Configure each metric to use our free models
    faithfulness.llm        = ragas_llm
    faithfulness.embeddings = ragas_embeddings

    answer_relevancy.llm        = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    context_recall.llm        = ragas_llm
    context_recall.embeddings = ragas_embeddings

    # Convert to HuggingFace Dataset format (required by RAGAS)
    dataset = Dataset.from_dict(data)

    # Run evaluation
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return results


def check_thresholds(results) -> tuple[bool, dict]:
    """
    Check if all metrics meet the minimum thresholds.
    Returns (passed: bool, scores: dict)
    """
    scores = {
        "faithfulness":     float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_recall":   float(results["context_recall"]),
    }

    thresholds = {
        "faithfulness":     FAITHFULNESS_THRESHOLD,
        "answer_relevancy": RELEVANCY_THRESHOLD,
        "context_recall":   RECALL_THRESHOLD,
    }

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    all_passed = True
    for metric, score in scores.items():
        threshold = thresholds[metric]
        passed    = score >= threshold
        status    = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"  {metric:<22} {score:.3f}  (threshold: {threshold})  {status}")

    print("=" * 60)
    return all_passed, scores


def save_results(scores: dict, passed: bool):
    """Save results to JSON for historical tracking."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "passed":    passed,
        "scores":    scores,
        "thresholds": {
            "faithfulness":     FAITHFULNESS_THRESHOLD,
            "answer_relevancy": RELEVANCY_THRESHOLD,
            "context_recall":   RECALL_THRESHOLD,
        }
    }

    existing = []
    if Path(RESULTS_PATH).exists():
        with open(RESULTS_PATH, "r") as f:
            existing = json.load(f)

    existing.append(record)

    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n💾 Results saved to {RESULTS_PATH}")


def main():
    print("🚀 Starting RAG Evaluation Pipeline")
    print("=" * 60)

    # Step 1: Load golden dataset
    print(f"\n📂 Loading golden dataset from {GOLDEN_DATASET_PATH}...")
    golden_data = load_golden_dataset(GOLDEN_DATASET_PATH)
    print(f"  Loaded {len(golden_data)} Q&A pairs")

    # Step 2: Run RAG pipeline on all questions
    rag_data = run_rag_on_dataset(golden_data)

    # Step 3: Evaluate with RAGAS using Groq
    results = evaluate_with_ragas(rag_data)

    # Step 4: Check thresholds
    passed, scores = check_thresholds(results)

    # Step 5: Save results
    save_results(scores, passed)

    # Step 6: Exit with correct code for CI/CD
    if passed:
        print("\n✅ All metrics passed! Build is healthy.\n")
        sys.exit(0)
    else:
        print("\n❌ Some metrics failed! Investigate before merging.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()