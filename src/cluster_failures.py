

import json
import os


CLUSTER_DEFINITIONS = {
    "temporal_precedence_blindness": {
        "description": (
            "The model answers using an outdated fact from an earlier document "
            "(doc_01) without checking whether a later document (doc_02) "
            "amended it. This suggests the model is not tracking document "
            "recency/precedence relationships — it treats all context as "
            "equally authoritative regardless of effective dates, likely "
            "because nothing in the prompt structurally signals which "
            "document 'wins' when two disagree."
        ),
        "fix_proposal": (
            "Add an explicit 'as-of date' resolution step to the prompt: "
            "before answering, require the model to list every document that "
            "mentions the queried fact along with its effective date, then "
            "select the most recent one. Alternatively, pre-process the corpus "
            "at ingestion time to merge amendments into a single 'current "
            "state' view per fact, separate from the raw historical documents, "
            "so the model isn't required to do temporal reasoning at all."
        ),
    },
    "citation_fabrication": {
        "description": (
            "The model produces a supporting_quote that does not exist "
            "verbatim (or near-verbatim) in the cited source document. This "
            "indicates the model is generating a plausible-sounding citation "
            "from its own paraphrase/summary of the content rather than "
            "actually copying text, which is a distinct failure mode from "
            "getting the underlying fact wrong."
        ),
        "fix_proposal": (
            "Add a mechanical copy-check step: after generation, extract the "
            "supporting_quote and re-prompt the model with 'here is the exact "
            "text of the document you cited — does your quote appear verbatim "
            "in it? If not, correct it or say unanswerable,' as a self-verification "
            "pass before the final answer is accepted."
        ),
    },
    "wrong_clause_attribution": {
        "description": (
            "The model's supporting_quote is real (passes the span-existence "
            "check) but does not actually support the stated answer, OR the "
            "model accepts a wrong clause citation as correct in the "
            "citation-check-style questions. This suggests the model retrieves "
            "topically-adjacent text without verifying it actually addresses "
            "the specific question asked — attention is landing near the "
            "right area of the document but not on the precise supporting "
            "clause."
        ),
        "fix_proposal": (
            "Add an explicit intermediate reasoning field to the output schema "
            "— e.g. 'relevant_clause_number' plus 'why_this_supports_the_answer' "
            "— forcing the model to articulate the logical link between quote "
            "and answer explicitly, rather than allowing quote and answer to be "
            "generated independently and only loosely related."
        ),
    },
    "similar_fact_conflation": {
        "description": (
            "The model confuses two similarly-worded facts within the same "
            "document (e.g. two different interest rates for two different "
            "situations). This points to a retrieval/attention issue rather "
            "than a knowledge gap — the model has 'seen' the correct fact but "
            "attaches it to the wrong condition, likely because the two facts "
            "are lexically similar and appear in nearby or structurally "
            "parallel clauses."
        ),
        "fix_proposal": (
            "Restructure the corpus documents (or add a lightweight "
            "pre-processing pass) to tag numerically similar/adjacent facts "
            "with explicit disambiguating labels close to the number itself "
            "(e.g. '18% p.a. [LATE PAYMENT INTEREST]' rather than relying on "
            "the model to infer the association from surrounding prose)."
        ),
    },
    "overconfident_wrong": {
        "description": (
            "The model reports high confidence (>=0.8) on an answer that is "
            "actually wrong. This is a calibration failure specifically, "
            "distinct from an accuracy failure — the model isn't just making "
            "mistakes, it's making mistakes it doesn't 'know' it's making, "
            "which is more dangerous in a production setting since it would "
            "pass any confidence-based filtering."
        ),
        "fix_proposal": (
            "Recalibrate confidence via a validation-set-based mapping (e.g. "
            "Platt scaling or isotonic regression) fit on the baseline eval "
            "results, applied as a post-hoc adjustment layer rather than "
            "trusting the model's raw self-reported number directly."
        ),
    },
    "unanswerable_overreach": {
        "description": (
            "The model provides a confident-sounding answer and/or citation "
            "for a question that has no support anywhere in the corpus. This "
            "suggests the model defaults to being 'helpful' by answering "
            "rather than recognizing and reporting absence — a common "
            "instruction-following gap in smaller models specifically."
        ),
        "fix_proposal": (
            "Add a small number of explicit few-shot examples in the prompt "
            "showing correctly-refused unanswerable questions, since smaller "
            "instruct models often need in-context demonstration of the "
            "refusal behavior rather than a single abstract instruction to "
            "'say unanswerable if not present.'"
        ),
    },
    "malformed_output": {
        "description": (
            "The model failed to produce parseable JSON even after the "
            "retry loop. This is a pure instruction-following/output-format "
            "reliability issue tied to model capacity (expected to be more "
            "common at 3B parameters than with a larger or hosted model), "
            "not a reasoning failure about the document content."
        ),
        "fix_proposal": (
            "Switch to constrained/structured generation (e.g. a JSON-schema-"
            "constrained decoding library such as outlines or guidance) "
            "instead of relying on prompt instructions plus a text-based "
            "retry loop, which would eliminate this failure category "
            "entirely regardless of model size."
        ),
    },
    "adversarial_robustness_gap": {
        "description": (
            "The model answers a paraphrased, context-injected, or "
            "distribution-edge variant of a question incorrectly, despite "
            "answering the original clean version correctly. This indicates "
            "the model's correct answer on the clean question may reflect "
            "surface pattern-matching to the exact clean phrasing rather than "
            "robust understanding of the underlying fact."
        ),
        "fix_proposal": (
            "Track a 'clean-to-adversarial consistency rate' as a first-class "
            "metric (percentage of clean-correct answers that remain correct "
            "under paraphrase/injection/distribution-edge), and treat any "
            "drop as a signal to expand few-shot examples or increase context "
            "window discipline in the prompt, not just treat each adversarial "
            "failure as an isolated data point."
        ),
    },
}


def classify_failure(row, clean_lookup=None):
    """Assigns one cluster label to a single failing/hallucinated row.
    clean_lookup: optional dict of {based_on_id: clean_row} to check whether
    an adversarial variant's base question was answered correctly in the
    clean baseline (needed for adversarial_robustness_gap classification)."""
    hallucination = row.get("hallucination")
    h_type = hallucination["type"] if hallucination else None
    category = row.get("category")
    confidence = row.get("model_confidence")
    adv_type = row.get("adversarial_type")  # only present in adversarial results

    if h_type == "malformed_output":
        return "malformed_output"

    if category == "unanswerable" and h_type == "cited_on_unanswerable":
        return "unanswerable_overreach"

    if h_type in ("fabricated_citation", "invalid_doc_id"):
        return "citation_fabrication"
    # Add a check BEFORE the wrong_clause_attribution rule:
    if h_type == "answer_not_entailed" and category in ("derived_numeric", "disambiguation_trap", "compound_conditional"):
     return "multi_fact_entailment_scoring_gap"  # scoring artifact, not a model failure
    if h_type == "answer_not_entailed" or category == "citation_check_fail":
     return "wrong_clause_attribution"  # genuine model failure

    if category in ("disambiguation_trap", "direct_disambiguation") and not row["passed"]:
        return "similar_fact_conflation"

    if category == "temporal_conflict" and not row["passed"]:
        return "temporal_precedence_blindness"

    if not row["passed"] and confidence is not None and confidence >= 0.8:
        return "overconfident_wrong"

    if adv_type and clean_lookup and row.get("based_on_id") in clean_lookup:
        clean_row = clean_lookup[row["based_on_id"]]
        if clean_row.get("passed") and not row["passed"]:
            return "adversarial_robustness_gap"

    return "uncategorized"  # flag for manual review — do not silently drop


def cluster_failures(baseline_path="results/baseline.json",
                      adversarial_path="results/adversarial_results.json",
                      output_path="results/failure_clusters.json"):
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)["results"]
    clean_lookup = {r["id"]: r for r in baseline}

    all_results = list(baseline)
    if os.path.exists(adversarial_path):
        with open(adversarial_path, "r", encoding="utf-8") as f:
            all_results += json.load(f)["results"]

    failures = [r for r in all_results if not r["passed"] or r.get("hallucination")]

    clusters = {}
    for row in failures:
        label = classify_failure(row, clean_lookup=clean_lookup)
        clusters.setdefault(label, []).append(row["id"])

    output = {}
    for label, ids in clusters.items():
        info = CLUSTER_DEFINITIONS.get(label, {
            "description": "NEEDS MANUAL REVIEW — did not match any defined cluster rule.",
            "fix_proposal": "TBD after manual inspection.",
        })
        output[label] = {
            "n_cases": len(ids),
            "example_ids": ids[:5],
            "all_ids": ids,
            "description": info["description"],
            "fix_proposal": info["fix_proposal"],
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== FAILURE CLUSTERS ({len(failures)} total failing/hallucinated cases) ===")
    for label, info in sorted(output.items(), key=lambda kv: -kv[1]["n_cases"]):
        print(f"\n[{label}] — {info['n_cases']} cases")
        print(f"  Example IDs: {info['example_ids']}")
        print(f"  Root cause: {info['description'][:150]}...")

    n_uncategorized = output.get("uncategorized", {}).get("n_cases", 0)
    if n_uncategorized:
        print(f"\nWARNING: {n_uncategorized} failures didn't match any rule — "
              f"read these by hand in {output_path} under 'uncategorized' and "
              f"either add a new cluster rule or manually assign them.")

    print(f"\nFull cluster data written to {output_path}")
    return output


if __name__ == "__main__":
    cluster_failures()