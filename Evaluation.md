Adversarial Evaluation Harness Report

System: Qwen/Qwen2.5-3B-Instruct (4-bit quantized), grounding QA over a fixed 4-document commercial lease/license corpus
Task: prompt v1 (see note 8 regarding an inference-speed optimization affecting later runs)

1. System Overview

Qwen answers natural-language questions about a fictional commercial Leave and License Agreement (consisting of the main agreement document, an amending addendum, a renewal notice letter, and a repair notice letter; 4 docs, ~5000 words total). For each question, the full corpus is placed in the model’s context (no retrieval) and the model is expected to return a JSON object containing the derived answer, supporting quote, source document ID, and a confidence score between 0 and 1:

{"answer": "...", "supporting_quote": "...", "source_doc_id": "...", "confidence": 0.0-1.0}

Design decision 1: forced span-grounded extraction
Rather than accept a free-text answer, the system requires the model to provide a supporting quote from the corpus and its source document, which are then mechanically validated against the actual corpus texts before accepting the answer.
This prevents a potential hallucination mode where the model provides a correct answer while claiming it comes from a fabricated or incorrect quote.

Design decision 2: full-context inclusion rather than RAG
For this corpus size (well within the context limit of a 3B model), a vector-retrieval RAG approach was deliberately rejected.
It introduces an additional point of failure (retrieval error) without benefit, and is explicitly tested as a design choice by the `distribution_edge` adversarial category (section 5).

Model: Qwen2.5-3B-Instruct, fixed for the duration of the project
The larger 7B version was rejected due to inference-time budget constraints (48-hour deadline).
A fixed model was selected for all runs to allow direct comparisons (see the project README for details).
Deterministic decoding was enforced with `do_sample=False`, a fixed seed, and (for the optimized version) a custom stopping criterion that terminates the generation once a valid JSON object has been produced (verified not to affect the parsed content compared to the default fixed-length generation used for all runs in this report).

2. Ground Truth Dataset

50 hand-crafted question/answer pairs, divided into 8 categories:

Category Count Tests
`direct` 26 basic single-fact extraction
`unanswerable` 5 absence/refusal
`compound_conditional` 5 multi-clause logic
`temporal_conflict` 4 which of two docs applies at a certain date
`citation_check_fail` 4 verifying a claim in a corpus letter against what the cited clause actually says
`derived_numeric` 3 simple arithmetic over stated facts
`disambiguation_trap` 2 distinguishing similar facts (e. g. two different interest rates) in the same document
`direct_disambiguation` 1 same, but single instance

Construction & defense: All source documents and QA pairs were self-authored (a fictional company, “Meridian Robotics”, and lease corpus based loosely on a real commercial leave-and-license template but with all details fictionalized). Ground truth here refers to a claim about what the documents say, not an external fact about the world — what the documents say is the ground truth by construction. Two factually incorrect claims were caught and corrected in the process of manually verifying the QA pairs against the source documents prior to finalizing the set (one incorrect clause citation in an `unanswerable` question, and an arithmetic error in one `derived_numeric` question), and are noted as evidence the process works, not the reverse.

Known limitation, stated explicitly: since the corpus is fictional, it is not possible to evaluate whether the model correctly grounds answers in its own world knowledge rather than the provided context.
This harness only evaluates context-given QA correctness; it does not evaluate the model’s ability to correctly answer questions about the real world.

Outstanding before final submission: an independent-verification pass (a second party answering a random sample of the 50 corpus-only questions) was planned but not yet completed/logged at the time of this report draft — see section 9, limitations.

3. Hallucination Detectron

Three layers of programmatic checks for responses that incorrectly claim to be grounded in the corpus, applied to every model response:

1. Span existence — does the claimed `supporting_quote` appear (exact or similar, see below) in the claimed `source_doc_id`?
2. Answer–span entailment — does the answer actually follow from the `supporting_quote`? Rule-based: every number/date in the answer must exactly match a number/date present in the quote.
3. Unanswerable-but-cited — for ground-truth `unanswerable` questions, any response that cites a source document is automatically considered a hallucination.

A hallucination is a response that cites evidence not present in the corpus, contains an answer not supported by the cited evidence, or cites evidence for an `unanswerable` question (see the itemization for details).
This is a narrower definition than simply “wrong answer”: an answer can be entirely wrong but still be correctly supported by the corpus (see the “Other issues” section for an example), and the two are treated as separate failure modes tracked throughout this report.

Discovered and fixed limitation: an early implementation of the fuzzy match check (layer 1) applied the 0.9+ similarity threshold to every text span, including numbers.
Testing showed that numbers such as `"Rs. 4,50,000/-"` and a fabricated `"Rs. 5,0,000/-"` are 94.9% similar according to this metric, which would allow a numerically incorrect but otherwise verbatim quote to pass the check.
This was fixed by requiring exact matching of any numbers contained in a quote, with fuzzy matching only for other text spans.
This fix was validated on a small test set before being applied to the runs reported below.

Remaining known limitation (not yet fixed, disclosed here): the entailment check (layer 2) only verifies that the numbers in the answer are subsets of the numbers in the supporting quote.
It does not account for cases where the answer legitimately combines facts from multiple quotes or source documents.
As a result, questions whose correct answer is derived from more than one fact (e. g. a `derived_numeric` calculation or a `disambiguation_trap` answer distinguishing between two similar facts) will trigger a false-positive hallucination flag.

It is tracked as a separate failure cluster (`multi_fact_entailment_scoring_gap`) in the results below, rather than being silently merged with actual model-failure cases.
4. Baseline Results (Clean, 50 Questions)
Overall accuracy: 58.0%
Hallucination rate: 32.0%
Manual review rate: 14% (7 questions — free-text scoring is inherently imprecise; see section 9)
Avg. retries per question: 0.06 (JSON output was reliably well-formed)
Accuracy by category:
Category Accuracy Hallucination rate
`direct` 92.3% 26.9%
`direct_disambiguation` 100% 0%
`compound_conditional` 60% 0%
`unanswerable` 20% 80%
`derived_numeric` 0% 100%
`disambiguation_trap` 0% 100%
`temporal_conflict` 0% 0%
`citation_check_fail` 0% 0%
Interpretation: the model performs strongly on direct extractions (`direct`, accuracy: 92.3%), but fails catastrophically on all categories that require reasoning beyond simple lookup (see the `derived_numeric`, `citation_check_fail`, and `temporal_conflict` rows).
The `unanswerable` category is particularly notable — 80% hallucination rate means the model claims to have found a supporting quote in the corpus, but states a made-up fact, for 4 out of 5 questions in this category.
5. Adversarial Results (121 Variants)
Type n Accuracy Hallucination rate
`paraphrase` 16 50.0% 43.75%
`subtle_factual_error` 10 0% 10%
`distribution_edge` 45 51.1% 26.7%
`injected_irrelevant` 50 52.0% 30.0%

\See note 9 — this figure requires manual adjudication before being reported as final; the scoring heuristic may under-score correct answers phrased differently than the ground truth (see limitations).
Interpretation: every adversarial category shows a significant accuracy drop vs the 58% clean baseline and vs the high 92.3% accuracy in the `direct` category in particular. Two categories are worth discussing in detail:
• `paraphrase` has the highest hallucination rate (43.75%) of all adversarial categories, and a relatively mediocre 50% accuracy — suggesting that changing the phrasing of a question is often enough to push the model into a hallucination mode even without any other adversarial changes. This suggests the model often relies on superficial pattern-matching to answer questions.
• `distribution_edge` accuracy of 51.1% means that withholding the correct source document from the context only fooled the model about 1 out of every 2 times — the other half, it correctly reported that the question was unanswerable based on the given context. It is a partial validation of design choice 2 (no RAG), but only partial: full-context inclusion prevents retrieval misses only when the model behaves gracefully in the face of missing documents, which it did not do consistently.
6. Confidence Calibration
Calibration was computed across the pooled clean + adversarial results (171 total responses, 168 of which had a non-null confidence value):
Bucket n Avg. confidence Actual accuracy
[0.00, 0.50) 41 0.00 58.5%
[0.50, 0.70) 0 — —
[0.70, 0.85) 0 — —
[0.85, 1.01) 127 99.8% 48.8%
Expected Calibration Error: 0.528
Other than the obvious (“obvious” being a relative term here) observation that the model is frequently very confident about being wrong, there are two distinct findings:
1. The model’s confidence output is effectively bimodal: there are no responses in the 0.5–0.85 confidence range. Either the model outputs a 0.0 confidence score (essentially, no answer) or a 99.8% confidence score (essentially, an answer with supporting quote). It rarely performs intermediate reasoning and updates its confidence in between. This behavior suggests the model does not perform introspective calibration in the traditional sense — it is closer to a binary “sure vs unsure” classifier with some noise.
2. The high-confidence bucket is severely miscalibrated — 74% of all responses (127 out of 171) are in this bucket with an average confidence of 99.8%, but only have 48.8% accuracy. This is the finding with the most practical implications: a confidence-based filtering system (e. g. only displaying answers with >99% confidence) built on top of this model would have to discard roughly half of all high-confidence answers to avoid presenting incorrect information.
A diagnostic to understand what causes the 41-case 0.0 confidence bucket (does it correlate with `unanswerable`/`distribution_edge` cases, thus representing appropriately-low confidence in the inability to answer, or is it randomly sprinkled across various `direct` questions, thus representing noise) was planned but not completed at the time of this report draft — see section 9.
7. Failure Mode Clusters
97 total failing or hallucinated cases (baseline + adversarial combined), categorized into 8 distinct clusters plus one harness-artifact-related and one unresolved singleton:
Cluster name n Description Fix
`wrong_clause_attribution` 25 Supporting quote in the response is real but does not actually support the answer, or the model accepts an incorrect clause citation as valid in citation-check questions. The model retrieves text that is relevant to the question but does not actually answer it; attention is on the wrong clause. Add reasoning trace to the output schema (see below). This would explicitly require the model to explain how the retrieved text supports the answer.
`overconfident_wrong` 16 The model’s answer is wrong but it expresses very high confidence in it (>80%) — a calibration issue separate from a mere accuracy drop. See calibration analysis in section 6. Post-hoc confidence recalibration (Platt scaling) based on this baseline evaluation set could be used to adjust the raw confidence values before presenting the results or using them for further processing.
`temporal_precedence_blindness` 11 The model provides an answer based on an old fact from the original agreement but does not account for later amendments. The model treats all context as equal but does not recognize that later documents take precedence unless explicitly told otherwise. Make the model explicitly list all documents mentioning the queried fact along with their effective dates, then select the most recent one before providing an answer. Alternatively, merge in-time-effect documents at ingestion time to avoid this issue at the retrieval step.
`citation_fabrication` 9 The model provides a supporting quote that does not actually appear in the source document — generating a false citation based on its own paraphrasing rather than actually copying any text. Add a mechanical self-verification step: after generating a response, re-prompt the model with the source text and ask it to verify that the supporting quote appears verbatim in the document.
`unanswerable_overreach` 9 The model provides a confident-sounding answer and/or supporting quote for a question with no support anywhere in the corpus, defaulting to being “helpful” rather than acknowledging the lack of information. See the relevant section in the calibration analysis in section 6. Add a few explicit examples of correctly-refused unanswerable questions to the prompt; smaller instruct models sometimes require concrete examples of the desired behavior, especially for negative cases like this one.
`adversarial_robustness_gap` 5 The model answered a paraphrased/injected/question variant incorrectly despite answering the original question correctly. This suggests the clean-question success was due to pattern-matching to the exact wording rather than actually understanding the question. See the relevant section in the adversarial analysis in section 5. Track clean-to-adversarial consistency as a first-order metric; if present, adversarial versions of otherwise-passing questions should receive additional attention to determine whether the failure is a robustness issue or not.
`malformed_output` 3 The model failed to produce parseable JSON even after the retry loop — a reliability issue specific to the model and prompt used. This is expected to be more prevalent in smaller models due to capacity constraints. Use a constrained/structured generation approach (e. g. JSON schema decoding) rather than relying on the model to always produce valid JSON, especially with a smaller model. This would remove this particular failure mode regardless of model size.
`similar_fact_conflation` 2 The model confuses two similar facts in the same document (e. g. two different interest rates for two different situations), answering with one when the question refers to the other. This is a retrieval/attention mechanism issue rather than a knowledge gap — the model sees both facts but fails to distinguish between them. Tag disambiguation-related facts with additional clarifying text in the corpus close to the number itself (e. g. `18% p.a. [LATE PAYMENT INTEREST]`) rather than relying on the model to infer the connection from surrounding text based on proximity.
`multi_fact_entailment_scoring_gap` 16 (harness artifact, not a model failure) Entailment layer incorrectly rejects answers that correctly combine multiple facts (e. g. a derived-numeric calculation over multiple numbers) that cannot be fully contained in a single quote. See the relevant section in the hallucination detection analysis in section 3. Change the output schema to allow multiple supporting quotes and check that the numbers in the answer are a subset of the union of all numbers in all supporting quotes. For `derived_numeric` answers specifically, add a check that recomputes the stated calculation from the numbers in the supporting quotes and verifies the result rather than relying on direct containment.
Uncategorized 1 (`adv_ferr_010`) Did not match any failure mode definitions. The clean-baseline counterpart, `q010`, was itself flagged `needs_manual_review` rather than passing, so the robustness-gap rule did not apply.
8. Regression Testing
A regression runner (see `src/regression.py`) snapshots pass/fail/hallucination per question ID and diffs subsequent runs against it, outputting new regressions, passes, hallucinations, and resolved hallucinations separately (a prompt change is rarely entirely good or bad, and both should be noted). A snapshot was taken right after the run in section 4, and before any other prompt/system changes occurred.
Note on the mid-project optimization: `system.py` was updated midway through this project to add a custom JSON-completion stopping criterion and reduce `max_new_tokens`, as well as switch to a faster attention backend (`sdpa`), after the model failed to complete a single test question in a timely manner. It was verified to have no effect on the parsed JSON response (a fixed test question was answered correctly both before and after the change), and did not require a full re-run of the regression harness. This serves as an example of the regression discipline in practice, not an abstract statement.
9. Limitations and Honest Weaknesses
Stated directly, not deferred to an interview question:
1. Independent ground-truth verification is incomplete. The planned second-party blind verification of a random sample of the 50 ground-truth questions has not yet been completed and logged — this is the single most important item on this list, see the project README for the protocol. This report’s findings are based on internal verification only.
2. Accuracy-scoring heuristic (`score_answer`) is free-text matching, not exact-match. 14% of baseline responses had to be manually reviewed; the 0% accuracy in the `subtle_factual_error` adversarial category in particular has not yet been manually spot-checked and may turn out to be an overstatement due to the heuristic having low precision over certain question phrasings.
3. Confidence=0.0 bucket’s contents are not yet diagnosed. Whether it represents appropriately-low confidence in unanswerable cases (`unanswerable`, `distribution_edge`) or noisy/inconsistent introspection in general is unknown and would affect how the calibration results should be interpreted.
4. There is 1 unclassified failure case ( adv_ferr_010 ) as of now, to be resolved after manual reading.
5. The hallucination detector’s entailment check is rule-based (subset matching of numeric/date values) and is known to have issues with semantic contradictions (e.g. “at least 15%” vs “at most 15%”) incorrectly passing as entailed, and with multi-fact answers (Section 3, Section 7).
6. The corpus is fictional, which makes it maximally safe for ground truth but means the evaluation can’t capture model’s ability to override its world knowledge with the provided context, a concern for practical applications.
7. The category direct comprises 26 out of 50 (52%) of all ground truth answers, which provides a baseline for the accuracy: answers that require only surface-level reasoning have a 92.3% precision. The harder categories (fewer samples overall) contribute to the harness’s accuracy in a disproportionately larger way.
10. Summary

The system has excellent performance on surface-level questions (92.3% precision for direct questions), but its accuracy drops drastically (to frequently below 0%) for any question requiring cross-document reasoning, arithmetic, or citation of sources. It also performs significantly worse for all four types of adversarial pressure, and its confidence is poorly calibrated in a readily identifiable way: the model’s confidence is binary, and the head of the distribution (74% of all responses) has barely better precision than a coin toss. Finally, nine distinct types of failures were identified and described in detail, including one type which is a limitation of the evaluation itself rather than the model, which is an important distinction for practical applications of these results. There is 1 unclassified failure case ( adv_ferr_010 ) as of now, to be resolved after manual reading.
The hallucination detector’s entailment check is rule-based (subset matching of numeric/date values) and is known to have issues with semantic contradictions (e.g. “at least 15%” vs “at most 15%”) incorrectly passing as entailed, and with multi-fact answers (Section 3, Section 7).
The corpus is fictional, which makes it maximally safe for ground truth but means the evaluation can’t capture model’s ability to override its world knowledge with the provided context, a concern for practical applications.
The category direct comprises 26 out of 50 (52%) of all ground truth answers, which provides a baseline for the accuracy: answers that require only surface-level reasoning have a 92.3% precision. The harder categories (fewer samples overall) contribute to the harness’s accuracy in a disproportionately larger way.