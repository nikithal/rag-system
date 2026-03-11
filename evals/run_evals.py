"""
run_evals.py - Phase 3
Custom RAG evaluation script using Groq as the judge LLM.

Runs sequentially (one question at a time) to avoid rate limit timeouts.

Metrics:
  - faithfulness:      Is every claim in the answer supported by the retrieved chunks?
  - answer_relevancy:  Does the answer actually address the question asked?
  - context_recall:    Do the retrieved chunks contain the information needed to answer?
"""

import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GOLDEN_DATASET_PATH    = "evals/golden_dataset.json"
RESULTS_PATH           = "evals/results.json"
FAITHFULNESS_THRESHOLD = 0.75
RELEVANCY_THRESHOLD    = 0.70
RECALL_THRESHOLD       = 0.70
SLEEP_BETWEEN_CALLS    = 3     # seconds between Groq calls to avoid rate limits
JUDGE_MODEL            = "llama-3.1-8b-instant"
# ──────────────────────────────────────────────────────────────────────────────

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_judge(prompt: str) -> str:
    """
    Send a prompt to the judge LLM and get a response.
    The judge reads answers and chunks to score quality.
    """
    time.sleep(SLEEP_BETWEEN_CALLS)   # avoid hitting rate limits
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,   # we only need a single number back
    )
    return response.choices[0].message.content.strip()


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: Is every claim in the answer supported by the retrieved chunks?

    The judge reads the answer and all retrieved chunks, then scores 0-10:
      10 = every single claim is directly supported by the chunks
       5 = some claims are supported, some are not
       0 = the answer contains claims not found in any chunk (hallucination)
    """
    context_text = "\n\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(contexts)])

    prompt = f"""You are evaluating whether an AI answer is faithful to its source documents.

Question: {question}

Retrieved chunks:
{context_text}

AI Answer: {answer}

Score the faithfulness of this answer from 0 to 10.
- Score 10: Every claim in the answer is directly supported by the chunks above.
- Score 5: Some claims are supported, others are not.
- Score 0: The answer contains information not found in any chunk.

Reply with only a single integer from 0 to 10. No explanation."""

    try:
        result = ask_judge(prompt)
        # Extract just the number from the response
        number = ''.join(filter(str.isdigit, result.split()[0] if result.split() else '0'))
        score = int(number) / 10.0 if number else 0.0
        return min(max(score, 0.0), 1.0)   # clamp between 0 and 1
    except Exception as e:
        print(f"    Warning: faithfulness scoring failed ({e}), defaulting to 0")
        return 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: Does the answer actually address the question asked?

    The judge reads the question and answer, then scores 0-10:
      10 = the answer directly and completely addresses the question
       5 = the answer is partially relevant
       0 = the answer does not address the question at all
    """
    prompt = f"""You are evaluating whether an AI answer is relevant to the question asked.

Question: {question}

AI Answer: {answer}

Score the relevancy of this answer from 0 to 10.
- Score 10: The answer directly and completely addresses the question.
- Score 5: The answer is partially relevant to the question.
- Score 0: The answer does not address the question at all.

Reply with only a single integer from 0 to 10. No explanation."""

    try:
        result = ask_judge(prompt)
        number = ''.join(filter(str.isdigit, result.split()[0] if result.split() else '0'))
        score = int(number) / 10.0 if number else 0.0
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        print(f"    Warning: relevancy scoring failed ({e}), defaulting to 0")
        return 0.0


def score_context_recall(question: str, ground_truth: str, contexts: list[str]) -> float:
    """
    Context Recall: Do the retrieved chunks contain what's needed to answer correctly?

    The judge reads the ground truth answer and chunks, then scores 0-10:
      10 = the chunks contain all the information in the ground truth
       5 = the chunks contain some of the needed information
       0 = the chunks do not contain the information needed to answer
    """
    context_text = "\n\n".join([f"Chunk {i+1}: {c}" for i, c in enumerate(contexts)])

    prompt = f"""You are evaluating whether retrieved document chunks contain the information needed to answer a question correctly.

Question: {question}

Correct answer: {ground_truth}

Retrieved chunks:
{context_text}

Score from 0 to 10 how well the retrieved chunks support the correct answer.
- Score 10: The chunks contain all the information needed to produce the correct answer.
- Score 5: The chunks contain some of the needed information.
- Score 0: The chunks do not contain the information needed to answer correctly.

Reply with only a single integer from 0 to 10. No explanation."""

    try:
        result = ask_judge(prompt)
        number = ''.join(filter(str.isdigit, result.split()[0] if result.split() else '0'))
        score = int(number) / 10.0 if number else 0.0
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        print(f"    Warning: context recall scoring failed ({e}), defaulting to 0")
        return 0.0


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    index: int,
    total: int
) -> dict:
    """
    Run all three metrics on a single Q&A pair.
    Runs sequentially with a delay between each call.
    """
    print(f"\n  [{index}/{total}] {question[:65]}...")

    faith = score_faithfulness(question, answer, contexts)
    print(f"    faithfulness:     {faith:.2f}")

    relev = score_answer_relevancy(question, answer)
    print(f"    answer_relevancy: {relev:.2f}")

    recall = score_context_recall(question, ground_truth, contexts)
    print(f"    context_recall:   {recall:.2f}")

    return {
        "question":         question,
        "faithfulness":     faith,
        "answer_relevancy": relev,
        "context_recall":   recall,
    }


def run_rag_on_dataset(golden_data: list[dict]) -> list[dict]:
    """
    Run the full RAG pipeline on every question in the golden dataset.
    Returns list of {question, answer, contexts, ground_truth}.
    """
    from src.pipeline import RAGPipeline

    rag     = RAGPipeline()
    results = []

    print(f"\n  Running RAG pipeline on {len(golden_data)} questions...")

    for i, item in enumerate(golden_data, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i}/{len(golden_data)}] Retrieving: {question[:60]}...")
        result = rag.ask(question, top_k=5)

        results.append({
            "question":     question,
            "answer":       result["answer"],
            "contexts":     [chunk["text"] for chunk in result["chunks"]],
            "ground_truth": ground_truth,
        })

        time.sleep(1)   # small pause between RAG calls

    return results


def check_thresholds(avg_scores: dict) -> bool:
    """Check if all average scores meet the minimum thresholds."""
    thresholds = {
        "faithfulness":     FAITHFULNESS_THRESHOLD,
        "answer_relevancy": RELEVANCY_THRESHOLD,
        "context_recall":   RECALL_THRESHOLD,
    }

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)

    all_passed = True
    for metric, score in avg_scores.items():
        threshold = thresholds[metric]
        passed    = score >= threshold
        status    = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {metric:<22} {score:.3f}  (threshold: {threshold})  {status}")

    print("=" * 60)
    return all_passed


def save_results(avg_scores: dict, per_question: list[dict], passed: bool):
    """Save full results to JSON for historical tracking."""
    record = {
        "timestamp":    datetime.now().isoformat(),
        "passed":       passed,
        "avg_scores":   avg_scores,
        "per_question": per_question,
        "thresholds": {
            "faithfulness":     FAITHFULNESS_THRESHOLD,
            "answer_relevancy": RELEVANCY_THRESHOLD,
            "context_recall":   RECALL_THRESHOLD,
        }
    }

    existing = []
    if Path(RESULTS_PATH).exists():
        with open(RESULTS_PATH, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append(record)
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n  Results saved to {RESULTS_PATH}")


def main():
    print("\n" + "=" * 60)
    print("  RAG EVALUATION PIPELINE")
    print("=" * 60)

    # Step 1: Load golden dataset
    print(f"\n  Loading: {GOLDEN_DATASET_PATH}")
    with open(GOLDEN_DATASET_PATH) as f:
        golden_data = json.load(f)
    print(f"  Loaded {len(golden_data)} Q&A pairs")

    # Step 2: Run RAG on all questions
    print("\n  Phase 1: Running RAG pipeline...")
    rag_results = run_rag_on_dataset(golden_data)

    # Step 3: Score each result individually
    print(f"\n  Phase 2: Scoring {len(rag_results)} answers...")
    print(f"  (using {JUDGE_MODEL} as judge, {SLEEP_BETWEEN_CALLS}s delay between calls)")

    per_question_scores = []
    for i, item in enumerate(rag_results, 1):
        scores = evaluate_single(
            question     = item["question"],
            answer       = item["answer"],
            contexts     = item["contexts"],
            ground_truth = item["ground_truth"],
            index        = i,
            total        = len(rag_results),
        )
        per_question_scores.append(scores)

    # Step 4: Average scores across all questions
    avg_scores = {
        "faithfulness":     sum(s["faithfulness"]     for s in per_question_scores) / len(per_question_scores),
        "answer_relevancy": sum(s["answer_relevancy"] for s in per_question_scores) / len(per_question_scores),
        "context_recall":   sum(s["context_recall"]   for s in per_question_scores) / len(per_question_scores),
    }

    # Step 5: Check thresholds and report
    passed = check_thresholds(avg_scores)

    # Step 6: Save results
    save_results(avg_scores, per_question_scores, passed)

    # Step 7: Exit with correct code for CI/CD
    if passed:
        print("\n  All metrics passed. Build is healthy.\n")
        sys.exit(0)
    else:
        print("\n  Some metrics failed. Investigate before merging.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()