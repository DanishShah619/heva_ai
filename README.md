# Adversarial Evaluation Harness for Grounded Document QA

An end-to-end evaluation harness designed to probe, stress-test, and adversarially break an LLM-based structured question-answering system.

Built for the **Heva AI — AI/ML Engineer Assignment (Adversarial Evaluation Harness)**.

---

## Executive Summary & System Architecture

Production AI reliability is governed by worst-case behavior, not happy-path accuracy. This project builds a **Span-Grounded Contract & Commercial Lease Question-Answering System** and subjects it to an exhaustive, multi-dimensional adversarial evaluation harness that measures:
- **Baseline Accuracy & Extraction Precision** across 8 reasoning categories
- **Hallucination Detection** via a multi-layered programmatic verifier
- **Adversarial Robustness** across 4 distinct perturbation vectors (121 variants)
- **Confidence Calibration** and Expected Calibration Error (ECE)
- **Failure Mode Clustering** into 8 distinct model-behavior failure archetypes with concrete fix proposals
- **Regression Testing** for continuous prompt and architecture iteration

```
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                   INPUT QUERY / ADVERSARIAL              │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │
                                                              ▼
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                  CORPUS INGESTION ENGINE                 │
                                 │       (Full Multi-Document Context Ingestion)            │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │
                                                              ▼
                                 ┌──────────────────────────────────────────────────────────┐
                                 │                  SYSTEM UNDER TEST                       │
                                 │   Qwen2.5-3B-Instruct (4-bit NF4, greedy do_sample=False)│
                                 │     Structured JSON Schema + Forced Span-Grounding       │
                                 └────────────────────────────┬─────────────────────────────┘
                                                              │
                                                              ▼
                                 ┌──────────────────────────────────────────────────────────┐
                                 │               EVALUATION & HARNESS SUITE                 │
                                 ├────────────────────────────┬─────────────────────────────┤
                                 │ 1. Answer Scorer           │ Semantic & Numeric Matcher  │
                                 │ 2. Hallucination Verifier  │ 3-Layer Programmatic Check  │
                                 │ 3. Calibration Engine      │ Binning & ECE Calculation   │
                                 │ 4. Failure Mode Clusterer  │ Behavioral Root-Cause Logic │
                                 │ 5. Regression Diff Runner  │ Snapshot & Delta Tracker    │
                                 └──────────────────────────────────────────────────────────┘
```

---

## Deliberate Design Decisions

A naive QA implementation typically asks an LLM for free-form answers or applies generic vector RAG. This system makes two deliberate, non-obvious design decisions:

### 1. Forced Span-Grounded Extraction with Mechanical Validation
* **Naive approach**: Asking the model for an answer directly, or relying on LLM-as-a-judge to grade output quality.
* **Our decision**: The model must output strict JSON containing the extracted answer, an exact `supporting_quote` from the corpus, the `source_doc_id`, and a `confidence` estimate ($0.0 \dots 1.0$). The harness then mechanically validates the supporting quote against the raw corpus document on disk before accepting the response.
* **Why it matters**: It decouples "factual recall" from "citation fabrication". An answer might be factually correct by coincidence, but if the cited quote is fabricated, the model hallucinated its grounding.

### 2. Full-Context Multi-Document Ingestion vs. Vector RAG
* **Naive approach**: Chunking documents into embeddings and retrieving top-$k$ passages with a vector database.
* **Our decision**: The entire 4-document lease corpus (~5,000 words) is placed directly into the context window of `Qwen2.5-3B-Instruct`.
* **Why it matters**: For corpora that fit within modern context windows, RAG introduces retrieval failure as an uncontrolled confounder. Ingestion of the full corpus isolates model reasoning failures (e.g. cross-document temporal precedence conflicts) from retrieval misses. The harness then explicitly tests retrieval failure as an adversarial dimension (`distribution_edge`).

### 3. Deterministic Decoding & Custom Stopping Criteria
* Generation is locked to `do_sample=False` with `seed=42`.
* A custom `StoppingCriteria` terminates token generation immediately once a balanced, syntactically complete JSON object is produced. This prevents smaller instruct models from generating rambling post-JSON trailing tokens up to `max_new_tokens`.

---

## Ground Truth Dataset

The ground truth dataset ([`eval/ground_truth.jsonl`](file:///d:/adverserial_llm/eval/ground_truth.jsonl)) contains **50 hand-authored, manually verified test cases** based on a fictional commercial lease agreement for "Meridian Robotics".

### Taxonomy of Test Categories

| Category | Count | Cognitive / Reasoning Task Tested |
| :--- | :---: | :--- |
| `direct` | 26 | Direct single-span factual extraction (e.g. rent amount, security deposit). |
| `unanswerable` | 5 | Information absent from all corpus documents; tests refusal capability. |
| `compound_conditional` | 5 | Multi-clause legal logic (e.g. lock-in period termination exceptions). |
| `temporal_conflict` | 4 | Resolving conflicting terms between the original lease and later addenda. |
| `citation_check_fail` | 4 | Verifying whether an assertion in a letter matches the clause it cites. |
| `derived_numeric` | 3 | Arithmetic over stated facts (e.g. deposit expressed in months of rent). |
| `disambiguation_trap` | 2 | Distinguishing two similar figures in the same doc (e.g. 18% late fee vs 12% deposit interest). |
| `direct_disambiguation` | 1 | Single-clause disambiguation query. |

### Dataset Defense & Verification Protocol
1. **Self-Contained Ground Truth by Construction**: Because the corpus is a custom legal contract, ground truth is strictly defined by the text of the documents, not open-world web knowledge.
2. **Manual Two-Pass Verification**: Every question, answer, supporting span, and doc citation was authored by hand and verified across all four corpus files. During authoring, two initial errors (one incorrect clause citation and one arithmetic miscalculation) were detected, audited, and corrected before freezing the benchmark.

---

## Programmatic Hallucination Detection

Unlike naive evaluation frameworks that rely on fuzzy LLM judges, this harness implements a **3-Layer Programmatic Hallucination Detector** ([`src/hallucination.py`](file:///d:/adverserial_llm/src/hallucination.py)):

```
                                      ┌───────────────────────────────┐
                                      │        MODEL RESPONSE         │
                                      │ (answer, quote, doc_id, conf) │
                                      └──────────────┬────────────────┘
                                                     │
                             ┌───────────────────────┴───────────────────────┐
                             │                                               │
                             ▼                                               ▼
               [Category == "unanswerable"]                      [Category != "unanswerable"]
                             │                                               │
                             ▼                                               ▼
                 LAYER 3: Unanswerable Check                     LAYER 1: Span Existence Check
            Did model provide quote / doc_id?              Does supporting_quote appear in doc?
               ├── Yes ➔ HALLUCINATION (Layer 3)             ├── No  ➔ HALLUCINATION (Layer 1)
               └── No  ➔ Pass                                └── Yes ➔ Proceed to Layer 2
                                                                             │
                                                                             ▼
                                                                LAYER 2: Entailment Check
                                                           Are numeric & date tokens in answer
                                                           a subset of tokens in the quote?
                                                             ├── No  ➔ HALLUCINATION (Layer 2)
                                                             └── Yes ➔ GROUNDED (Pass)
```

### Layer Details & Rules:
1. **Layer 1 (Span Existence)**: Checks if `supporting_quote` exists verbatim or near-verbatim in the claimed `source_doc_id`.
   - *Discovered & Fixed Edge Case*: A standard SequenceMatcher ratio threshold ($\ge 0.9$) allowed numbers like `"Rs. 4,50,000/-"` and `"Rs. 5,00,000/-"` to pass as matches (94.9% similarity). The verifier was hardened to require **100% exact match on all extracted numbers** while allowing fuzzy matching on surrounding text.
2. **Layer 2 (Answer-Span Entailment)**: Rule-based verification ensuring all numeric and date tokens in `model_answer` are subsets of tokens present in the cited `supporting_quote`.
3. **Layer 3 (Unanswerable Violation)**: Any response to an `unanswerable` question that produces a citation or asserts a factual claim is immediately flagged as a hallucination.

---

## Adversarial Evaluation Suite

The adversarial harness ([`src/adversarial.py`](file:///d:/adverserial_llm/src/adversarial.py)) tests **121 adversarial variants** across 4 required dimensions:

| Adversarial Dimension | Count | Implementation Method | Failure Probed |
| :--- | :---: | :--- | :--- |
| **1. Paraphrase** | 16 | Hand-authored syntactic & semantic rephrasings ([`paraphrase.jsonl`](file:///d:/adverserial_llm/eval/adversarial/paraphrase.jsonl)). | Superficial prompt-pattern matching vs true understanding. |
| **2. Injected Irrelevant Context** | 50 | Programmatic injection of an unrelated building cafeteria menu notice (`doc_99`). | Context window distractor resistance & retrieval interference. |
| **3. Subtle Factual Error** | 10 | Hand-authored questions with inverted factual premises ([`factual_error.jsonl`](file:///d:/adverserial_llm/eval/adversarial/factual_error.jsonl)). | Sycophancy & acceptance of false user presuppositions. |
| **4. Distribution Edge (Simulated Retrieval Miss)** | 45 | Programmatic exclusion of the true source document from the context window. | Graceful degradation / refusal when evidence is missing. |

### Summary of Results

```
Baseline (Clean 50 Questions):   Accuracy: 58.0% | Hallucination Rate: 32.0%
Adversarial Breakdown:
  • Paraphrased (n=16):          Accuracy: 50.0% | Hallucination Rate: 43.75%
  • Injected Irrelevant (n=50):  Accuracy: 52.0% | Hallucination Rate: 30.00%
  • Distribution Edge (n=45):    Accuracy: 51.1% | Hallucination Rate: 26.67%
  • Subtle Factual Error (n=10): Accuracy:  0.0% | Hallucination Rate: 10.00%
```

---

## Confidence Calibration & ECE

Confidence calibration evaluates whether the model's reported `confidence` reflects its empirical probability of being correct.

* **Expected Calibration Error (ECE)**: **`0.528`** across 171 pooled clean + adversarial responses.
* **Calibration Plot**: Saved to [`result/calibration_curve.png`](file:///d:/adverserial_llm/result/calibration_curve.png).

```
   Confidence Bucket      Count (n)      Avg. Confidence      Actual Accuracy
   ─────────────────      ─────────      ───────────────      ───────────────
   [0.00, 0.50)              41               0.00                 58.5%
   [0.50, 0.70)               0                 —                    —
   [0.70, 0.85)               0                 —                    —
   [0.85, 1.01)             127              99.8%                 48.8%
```

### Key Calibration Insights:
1. **Bimodal Distribution**: The model never outputs intermediate confidence (no samples between $0.50$ and $0.85$). It acts as an uncalibrated binary switch ($0.0$ or $\approx 1.0$).
2. **Severe Overconfidence**: In the $[0.85, 1.01)$ bucket, the model expresses $99.8\%$ average confidence, yet is correct only $48.8\%$ of the time.

---

## Failure Mode Clusters & Fix Proposals

Running the failure clusterer ([`src/cluster_failures.py`](file:///d:/adverserial_llm/src/cluster_failures.py)) over all 97 failed/hallucinated instances reveals **8 distinct model-level failure clusters** and 1 evaluation scoring gap:

| Failure Cluster | Count | Behavioral Root Cause | Concrete Fix Proposal |
| :--- | :---: | :--- | :--- |
| `wrong_clause_attribution` | 25 | Attention lands in topically-adjacent text without verifying it answers the specific clause conditions. | Add an intermediate reasoning step to the JSON schema requiring the model to output `clause_number` and `why_this_supports_answer`. |
| `overconfident_wrong` | 16 | Model outputs $\ge 0.80$ confidence on completely incorrect extractions due to instruction-tuning RLHF bias. | Apply post-hoc temperature scaling / Platt scaling on calibration logs prior to surfacing confidence to downstream systems. |
| `temporal_precedence_blindness` | 11 | Treats all documents as equally valid, ignoring amendment dates (e.g. Addendum overriding master agreement). | Add an explicit as-of date resolution step: force model to list all documents mentioning the fact with their dates and pick the latest. |
| `citation_fabrication` | 9 | Generates plausible-sounding citations from internal summaries rather than copying verbatim text. | Add a post-generation mechanical self-verification pass re-prompting model to confirm the exact quote exists in the text. |
| `unanswerable_overreach` | 9 | Defaulting to "helpfulness" by synthesizing answers for questions unsupported by corpus. | Add few-shot examples of valid refusals directly in the prompt demonstration block. |
| `adversarial_robustness_gap` | 5 | Fragile memorization of specific prompt phrasings; fails when syntax changes. | Augment system prompt with prompt-invariance formatting instructions and evaluate against paraphrased datasets. |
| `malformed_output` | 3 | JSON parsing syntax failure on complex generations. | Enforce JSON schema constrained decoding via grammar-guided generation (e.g. Outlines / Guidance). |
| `similar_fact_conflation` | 2 | Confuses two distinct numbers in the same document (e.g. 18% late payment vs 12% deposit refund). | Prepend clarifying semantic metadata tags to ambiguous numbers during corpus ingestion. |
| `multi_fact_entailment_scoring_gap` *(Harness Artifact)* | 16 | The harness's Layer 2 verifier assumes single-quote containment and falsely flags multi-clause arithmetic. | Update output schema to accept a list of `supporting_quotes` and verify arithmetic deterministically via a sandboxed calculator. |

---

## Regression Test Runner

The regression harness ([`src/regression.py`](file:///d:/adverserial_llm/src/regression.py)) establishes baseline snapshots ([`result/baseline_snapshot.json`](file:///d:/adverserial_llm/result/baseline_snapshot.json)) and reports diffs upon prompt or model updates:

```python
from src.regression import snapshot, diff_against_snapshot

# 1. Snapshot a trusted baseline run
snapshot(baseline_results, path="results/baseline_snapshot.json")

# 2. Diff a candidate prompt run against baseline
diff = diff_against_snapshot(new_results, snapshot_path="results/baseline_snapshot.json")
print(diff["regressions"])              # Newly broken questions
print(diff["new_passes"])               # Newly fixed questions
print(diff["new_hallucinations"])       # Regressed hallucination cases
print(diff["resolved_hallucinations"])  # Fixed hallucination cases
```

---

## Quickstart & Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/DanishShah619/huxaai.git
cd huxaai

# Create virtual environment (Python 3.10+)
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install torch transformers accelerate bitsandbytes matplotlib numpy tqdm
```

> **Hardware Requirement**: CUDA-compatible GPU with $\ge 6\text{ GB}$ VRAM (e.g. Google Colab T4 or local NVIDIA GPU).

### 2. Execution Pipeline

Run each stage of the evaluation harness sequentially:

```bash
# Stage 1: Run baseline evaluation on the 50 ground-truth questions
python src/run_eval.py

# Stage 2: Run the full adversarial suite (121 variants)
python src/adversarial.py

# Stage 3: Compute Expected Calibration Error & generate calibration curve plot
python src/calibration.py

# Stage 4: Run failure clustering and generate root-cause breakdown
python src/cluster_failures.py
```

All summary metrics and JSON results are written to `results/` (and mirrored in `result/`).

---

## Repository Structure

```
.
├── Evaluation.md                 # Full detailed evaluation report
├── README.md                     # Technical overview and reproduction guide
├── corpus/                       # Multi-document commercial lease corpus
│   ├── doc_01_lease_main.md      # Master Leave and License Agreement
│   ├── doc_02_addendum.md        # Amending Addendum (temporal override)
│   ├── doc_03_renewal_notice.md  # Formal renewal notice letter
│   └── doc_04_repair_notice.md   # Structural repair request letter
├── eval/                         # Evaluation datasets
│   ├── ground_truth.jsonl        # 50 hand-verified ground truth test cases
│   └── adversarial/
│       ├── paraphrase.jsonl      # 16 hand-authored syntactic paraphrases
│       └── factual_error.jsonl   # 10 hand-authored false-presupposition queries
├── src/                          # Evaluation harness codebase
│   ├── system.py                 # System under test (Qwen-3B + JSON decoding)
│   ├── run_eval.py               # Baseline evaluation runner & answer scorer
│   ├── hallucination.py          # 3-layer programmatic hallucination verifier
│   ├── adversarial.py            # Adversarial test generator & runner
│   ├── calibration.py            # Calibration bucket calculation & curve plotting
│   ├── cluster_failures.py       # Behavioral failure mode clustering engine
│   └── regression.py             # Snapshot diff and regression test engine
└── results/                      # Generated evaluation outputs & artifacts
    ├── baseline.json             # Baseline clean evaluation results
    ├── adversarial_results.json  # Full adversarial run output
    ├── calibration.json          # Calibration bucketing & ECE metric
    ├── calibration_curve.png     # Visual calibration curve plot
    ├── failure_clusters.json     # Clustered failure mode instances & statistics
    └── baseline_snapshot.json    # Regression testing baseline snapshot
```

---

## Interview Defense & Key Talking Points

### 1. How do you know your ground truth is correct?
* Ground truth is **authoritative by construction**: the corpus is a self-contained legal agreement authored specifically for this harness.
* Every QA pair was manually audited against the source documents with a two-pass cross-verification protocol that explicitly identified and eliminated arithmetic and clause citation discrepancies before freezing the dataset.

### 2. Where does your hallucination detection fail?
* **Semantic inversion**: Layer 2 checks for numeric/date token subset containment. If an answer asserts `"at least 15%"` when the supporting quote says `"at most 15%"`, the numbers match and it passes the check despite semantic inversion.
* **Multi-fact composition**: When an answer legitimately combines facts from multiple documents or clauses (e.g. arithmetic across rent and deposit), checking against a single quote triggers a false-positive hallucination flag (`multi_fact_entailment_scoring_gap`).

### 3. Why is the model's confidence bimodal?
* Instruction-tuned language models (especially 3B scale) are trained to emit high-probability tokens and lack calibrated internal Bayesian uncertainty estimates. Unless explicitly trained with verbalized calibration objectives, prompting for confidence produces a collapsed distribution ($0.0$ for explicit refusal vs $0.99+$ for any generated answer).

### 4. What is one additional adversarial category you would design?
* **Subtle Negation & Inverted Quantifiers in Ingested Context**: Injecting a plausible synthetic amendment that uses subtle legal double negatives (e.g., *"Notwithstanding Clause 4, the Licensee shall not be deemed non-liable for unaccrued taxes"*). This directly stress-tests the model's syntactic parsing of legal liability under adverse polarity.
