

import json
import os
import time
from datetime import datetime, timezone

from tqdm.auto import tqdm

from system import load_corpus, query, CONFIG as SYSTEM_CONFIG
from hallucination import detect_hallucination



def load_ground_truth(path="eval/ground_truth.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



# Accuracy scoring
#
# IMPORTANT, disclose this in the report: ground truth answers are free text,
# not multiple choice, so exact string match is not a meaningful accuracy
# signal (e.g. "Rs. 4,50,000/- per month" vs "Rs. 4,50,000 monthly" are both
# correct but won't string-match). This scorer uses a heuristic — number/date
# overlap plus a keyword check — and explicitly flags borderline cases as
# "needs_manual_review" rather than silently guessing. Report the manual
# review rate as an honest part of the methodology, not something to hide.

import re

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*%?")


def _numbers(text):
    if not text:
        return set()
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def score_answer(model_answer, ground_truth_answer, category):
    """Returns one of: 'correct', 'incorrect', 'needs_manual_review'."""
    if model_answer is None:
        return "incorrect"

    model_answer_norm = model_answer.strip().lower()
    gt_norm = ground_truth_answer.strip().lower()

    if category == "unanswerable":
        # Correct iff the model says it can't find the answer.
        refusal_signals = ["unanswerable", "not stated", "not mentioned",
                            "does not state", "no provision", "cannot find",
                            "not found", "not specified", "no such"]
        if any(sig in model_answer_norm for sig in refusal_signals):
            return "correct"
        return "incorrect"

    gt_numbers = _numbers(ground_truth_answer)
    model_numbers = _numbers(model_answer)

    if gt_numbers:
        if gt_numbers.issubset(model_numbers):
            return "correct"
        elif gt_numbers.isdisjoint(model_numbers):
            return "incorrect"
        else:
            return "needs_manual_review"  # partial numeric overlap — ambiguous

    # Non-numeric fallback: crude keyword overlap on content words.
    gt_words = {w for w in re.findall(r"[a-z]+", gt_norm) if len(w) > 4}
    model_words = {w for w in re.findall(r"[a-z]+", model_answer_norm) if len(w) > 4}
    if not gt_words:
        return "needs_manual_review"
    overlap_ratio = len(gt_words & model_words) / len(gt_words)
    if overlap_ratio >= 0.5:
        return "correct"
    elif overlap_ratio <= 0.15:
        return "incorrect"
    return "needs_manual_review"



def run_baseline(
    corpus_dir="corpus",
    ground_truth_path="eval/ground_truth.jsonl",
    output_path="results/baseline.json",
):
    docs = load_corpus(corpus_dir)
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(docs)} corpus docs, {len(ground_truth)} ground truth questions.")

    results = []
    t0 = time.time()

    for row in tqdm(ground_truth, desc="Running baseline eval"):
        response = query(row["question"], docs)
        parsed = response.get("parsed") or {}
        model_answer = parsed.get("answer")

        verdict = score_answer(model_answer, row["answer"], row["category"])
        hallucination = detect_hallucination(response, docs, row["category"])

        results.append({
            "id": row["id"],
            "category": row["category"],
            "question": row["question"],
            "ground_truth_answer": row["answer"],
            "model_answer": model_answer,
            "model_supporting_quote": parsed.get("supporting_quote"),
            "model_source_doc_id": parsed.get("source_doc_id"),
            "model_confidence": parsed.get("confidence"),
            "verdict": verdict,               # correct / incorrect / needs_manual_review
            "passed": verdict == "correct",   # boolean used by regression.py
            "hallucination": hallucination,    # None, or dict with layer/type/detail
            "retries_needed": response.get("retries_needed"),
            "raw_output": response.get("raw"),
        })

    elapsed = time.time() - t0

    #Summary stats
    n = len(results)
    n_correct = sum(1 for r in results if r["verdict"] == "correct")
    n_review = sum(1 for r in results if r["verdict"] == "needs_manual_review")
    n_hallucinated = sum(1 for r in results if r["hallucination"] is not None)

    by_category = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"n": 0, "correct": 0, "hallucinated": 0})
        by_category[cat]["n"] += 1
        by_category[cat]["correct"] += int(r["verdict"] == "correct")
        by_category[cat]["hallucinated"] += int(r["hallucination"] is not None)

    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "model": SYSTEM_CONFIG["model_name"],
        "model_revision": SYSTEM_CONFIG["model_revision"],
        "n_questions": n,
        "overall_accuracy": round(n_correct / n, 4) if n else None,
        "manual_review_rate": round(n_review / n, 4) if n else None,
        "hallucination_rate": round(n_hallucinated / n, 4) if n else None,
        "avg_retries_per_question": round(
            sum(r["retries_needed"] or 0 for r in results) / n, 3
        ) if n else None,
        "elapsed_seconds": round(elapsed, 1),
        "accuracy_by_category": {
            cat: round(v["correct"] / v["n"], 4) for cat, v in by_category.items()
        },
        "hallucination_rate_by_category": {
            cat: round(v["hallucinated"] / v["n"], 4) for cat, v in by_category.items()
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== BASELINE SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {output_path}")
    print(f"NOTE: {n_review} question(s) flagged 'needs_manual_review' — "
          f"review these by hand before reporting final accuracy.")

    return summary, results


if __name__ == "__main__":
    run_baseline()