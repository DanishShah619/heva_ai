# adversarial.py
# Generates and runs the 4 required adversarial categories against the system:
#   1. paraphrase              — hand-authored, manually verified (loaded from JSONL)
#   2. injected_irrelevant     — programmatic: appends a fabricated unrelated doc to context
#   3. subtle_factual_error    — hand-authored, manually verified (loaded from JSONL)
#   4. distribution_edge       — programmatic: removes the correct source doc from context
#
# Categories 1 and 3 are loaded from hand-written files, NOT LLM-auto-generated-and-shipped,
# per the assignment's rule that any LLM-generated eval content must be manually verified.
#
# RELIABILITY NOTES (added after an earlier run stalled for 35+ minutes with
# no visible progress and no partial results to recover):
#   - Results are now written incrementally, one line at a time, to a .jsonl
#     file as each variant completes — NOT only at the very end. If this
#     script is interrupted, everything up to that point is already saved.
#   - Explicit print() with per-call timing after every single variant,
#     instead of relying solely on tqdm (whose live-redraw can fail to
#     render visibly when a script is run via `%run` in Colab).
#   - See system.py's optimization notes for the stopping-criterion fix
#     that addresses the most likely root cause of the original stall.

import json
import os
import time

from system import load_corpus, query
from hallucination import detect_hallucination
from run_eval import score_answer, load_ground_truth


# ---------------------------------------------------------------------------
# Category 2: injected irrelevant context (fully programmatic)
# ---------------------------------------------------------------------------
DEFAULT_INJECTION_DOC_ID = "doc_99_injected_irrelevant"
DEFAULT_INJECTION_TEXT = (
    "Doc ID: doc_99_injected_irrelevant\n"
    "Title: Meridian Corporate Towers — Building Cafeteria Menu Notice\n\n"
    "Effective 1st June 2007, the ground-floor cafeteria at Meridian Corporate "
    "Towers will offer a revised lunch menu. Vegetarian thali is priced at "
    "Rs. 120/-, and non-vegetarian thali at Rs. 150/-. The cafeteria will be "
    "closed on all public holidays declared by the Government of Maharashtra. "
    "For catering queries, contact the building facilities desk on the ground "
    "floor. This notice is issued by the Building Management Committee and is "
    "unrelated to any license or lease terms between individual tenants."
)


def inject_irrelevant_context(docs, injection_text=DEFAULT_INJECTION_TEXT,
                               injection_doc_id=DEFAULT_INJECTION_DOC_ID):
    """Returns a NEW docs dict with one fabricated, topically-unrelated
    document added. Does not mutate the original corpus on disk — this is
    a transient, per-query addition."""
    new_docs = dict(docs)
    new_docs[injection_doc_id] = injection_text
    return new_docs


# ---------------------------------------------------------------------------
# Category 4: distribution edge / simulated retrieval miss (fully programmatic)
# ---------------------------------------------------------------------------
def make_distribution_edge_variants(ground_truth):
    """For every answerable ground truth question, creates a variant where
    its correct source_doc_id is withheld from context. Skips 'unanswerable'
    questions (nothing meaningful to withhold — they have no source doc)."""
    variants = []
    for row in ground_truth:
        if row["category"] == "unanswerable" or row.get("source_doc_id") is None:
            continue
        variants.append({
            "id": f"{row['id']}_dist_edge",
            "based_on_id": row["id"],
            "adversarial_type": "distribution_edge",
            "question": row["question"],
            "answer": row["answer"],
            "source_doc_id": row["source_doc_id"],
            "supporting_span": row.get("supporting_span"),
            "category": row["category"],
            "excluded_doc_id": row["source_doc_id"],
        })
    return variants


# ---------------------------------------------------------------------------
# Categories 1 and 3: hand-authored, manually-verified variant files
# ---------------------------------------------------------------------------
def load_hand_authored_variants(path):
    """Loads a JSONL file of hand-written adversarial variants.
    Raises loudly if the file is missing — these are NOT auto-generated,
    they must exist because a human wrote and verified them."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Categories 'paraphrase' and 'subtle_factual_error' "
            f"must be hand-authored and manually verified — they are not "
            f"auto-generated at runtime."
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_adversarial(
    corpus_dir="corpus",
    ground_truth_path="eval/ground_truth.jsonl",
    paraphrase_path="eval/adversarial/paraphrase.jsonl",
    factual_error_path="eval/adversarial/factual_error.jsonl",
    output_path="results/adversarial_results.json",
    resume=True,
):
    docs = load_corpus(corpus_dir)
    ground_truth = load_ground_truth(ground_truth_path)

    paraphrase_variants = [{**v, "adversarial_type": "paraphrase"}
                            for v in load_hand_authored_variants(paraphrase_path)]
    factual_error_variants = [{**v, "adversarial_type": "subtle_factual_error"}
                               for v in load_hand_authored_variants(factual_error_path)]
    distribution_edge_variants = make_distribution_edge_variants(ground_truth)
    injected_variants = [{
        "id": f"{row['id']}_inject",
        "based_on_id": row["id"],
        "adversarial_type": "injected_irrelevant",
        "question": row["question"],
        "answer": row["answer"],
        "source_doc_id": row.get("source_doc_id"),
        "supporting_span": row.get("supporting_span"),
        "category": row["category"],
    } for row in ground_truth]

    all_variants = (paraphrase_variants + factual_error_variants
                     + distribution_edge_variants + injected_variants)

    print(f"Running {len(all_variants)} adversarial variants "
          f"({len(paraphrase_variants)} paraphrase, "
          f"{len(factual_error_variants)} factual_error, "
          f"{len(distribution_edge_variants)} distribution_edge, "
          f"{len(injected_variants)} injected_irrelevant)")

    # --- Resume support: skip variants already completed in a partial file ---
    partial_path = output_path.replace(".json", "_partial.jsonl")
    os.makedirs(os.path.dirname(partial_path), exist_ok=True)

    already_done_ids = set()
    results = []
    if resume and os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    results.append(row)
                    already_done_ids.add(row["id"])
        print(f"Resuming: {len(already_done_ids)} variant(s) already completed "
              f"in {partial_path}, will skip those.")

    remaining = [v for v in all_variants if v["id"] not in already_done_ids]
    print(f"{len(remaining)} variant(s) left to run.\n")

    partial_f = open(partial_path, "a", encoding="utf-8")
    run_t0 = time.time()

    try:
        for i, v in enumerate(remaining):
            adv_type = v["adversarial_type"]
            call_t0 = time.time()
            print(f"[{i+1}/{len(remaining)}] {v['id']} ({adv_type})...", end=" ", flush=True)

            if adv_type == "distribution_edge":
                response = query(v["question"], docs, excluded_doc_id=v["excluded_doc_id"])
            elif adv_type == "injected_irrelevant":
                query_docs = inject_irrelevant_context(docs)
                response = query(v["question"], query_docs)
            else:  # paraphrase, subtle_factual_error
                response = query(v["question"], docs)

            parsed = response.get("parsed") or {}
            model_answer = parsed.get("answer")

            if adv_type == "distribution_edge":
                refusal_signals = ["unanswerable", "not stated", "not mentioned",
                                    "does not state", "cannot find", "not found",
                                    "not specified", "no such", "not available",
                                    "not provided", "not present"]
                verdict = "correct" if any(
                    sig in (model_answer or "").lower() for sig in refusal_signals
                ) else "incorrect"
            else:
                verdict = score_answer(model_answer, v["answer"], v["category"])

            detect_docs = docs if adv_type != "distribution_edge" else {
                k: dv for k, dv in docs.items() if k != v["excluded_doc_id"]
            }
            hallucination = detect_hallucination(response, detect_docs, v["category"])

            result = {
                "id": v["id"],
                "based_on_id": v.get("based_on_id"),
                "adversarial_type": adv_type,
                "category": v["category"],
                "question": v["question"],
                "ground_truth_answer": v["answer"],
                "model_answer": model_answer,
                "model_supporting_quote": parsed.get("supporting_quote"),
                "model_source_doc_id": parsed.get("source_doc_id"),
                "model_confidence": parsed.get("confidence"),
                "verdict": verdict,
                "passed": verdict == "correct",
                "hallucination": hallucination,
                "retries_needed": response.get("retries_needed"),
                "raw_output": response.get("raw"),
            }

            elapsed = time.time() - call_t0
            print(f"done in {elapsed:.1f}s — verdict={verdict}"
                  f"{' — HALLUCINATION: ' + hallucination['type'] if hallucination else ''}")

            partial_f.write(json.dumps(result) + "\n")
            partial_f.flush()
            results.append(result)

    finally:
        partial_f.close()

    total_elapsed = time.time() - run_t0
    print(f"\nCompleted {len(remaining)} variant(s) in {total_elapsed:.1f}s "
          f"({total_elapsed/max(1,len(remaining)):.1f}s/variant average).")

    # --- Summary by adversarial type ---
    by_type = {}
    for r in results:
        t = r["adversarial_type"]
        by_type.setdefault(t, {"n": 0, "correct": 0, "hallucinated": 0})
        by_type[t]["n"] += 1
        by_type[t]["correct"] += int(r["passed"])
        by_type[t]["hallucinated"] += int(r["hallucination"] is not None)

    summary = {
        "n_variants": len(results),
        "accuracy_by_adversarial_type": {
            t: round(v["correct"] / v["n"], 4) for t, v in by_type.items()
        },
        "hallucination_rate_by_adversarial_type": {
            t: round(v["hallucinated"] / v["n"], 4) for t, v in by_type.items()
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print("\n=== ADVERSARIAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {output_path}")
    print(f"(Incremental backup also available at {partial_path})")

    return summary, results


if __name__ == "__main__":
    run_adversarial()