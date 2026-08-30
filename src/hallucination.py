
# Three layered hallucination checks, run against every response from system.py.
# Layer 1: span existence     — does the claimed supporting_quote actually
#                                appear in the claimed source_doc_id?
# Layer 2: answer entailment  — does the answer actually follow from the quote?
#                                (rule-based: number/date/entity overlap)
# Layer 3: unanswerable-cited — for ground-truth "unanswerable" questions, any
#                                citation at all is an automatic fail.

import re
from difflib import SequenceMatcher


def check_span_exists(quote, doc_text, fuzzy_threshold=0.9, window_stride=20):
    """Returns True if quote appears in doc_text, exact or fuzzy match.
    Returns None if quote is None (not applicable — e.g. unanswerable case)."""
    if quote is None:
        return None

    norm_quote = " ".join(quote.replace("**", "").split())
    norm_doc = " ".join(doc_text.replace("**", "").split())

    if norm_quote in norm_doc:
        return True
    quote_numbers = set(re.findall(r"\d[\d,]*\.?\d*", norm_quote))
    if quote_numbers:
        doc_numbers = set(re.findall(r"\d[\d,]*\.?\d*", norm_doc))
        if not quote_numbers.issubset(doc_numbers):
            return False
  
    n = len(norm_quote)
    if n == 0:
        return False
    best_ratio = 0.0
    for i in range(0, max(1, len(norm_doc) - n), window_stride):
        window = norm_doc[i:i + n]
        ratio = SequenceMatcher(None, norm_quote, window).ratio()
        best_ratio = max(best_ratio, ratio)
        if best_ratio >= fuzzy_threshold:
            break
    return best_ratio >= fuzzy_threshold



_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*%?")
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)


def _extract_numbers(text):
    """Extracts numeric tokens, normalized (commas stripped) for comparison."""
    if not text:
        return set()
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def _extract_months(text):
    if not text:
        return set()
    return {m.lower()[:3] for m in _MONTH_RE.findall(text)}


def check_entailment(answer, quote):
    """Rule-based entailment check. Returns (ok: bool, reason: str|None).

    KNOWN LIMITATION (disclose this explicitly in the report — do not treat
    it as a hidden edge case): this only checks surface-level numeric/date
    overlap. It will NOT catch semantic contradictions where the numbers
    match but the relationship is inverted — e.g. answer says "at least 15%"
    when the quote says "at most 15%" would pass this check even though the
    meaning is reversed. This is a deliberate scope boundary given the 48-hour
    build budget, not an oversight, and is the single biggest thing to
    improve if this project continues past this assignment.
    """
    if answer is None or quote is None:
        return None, "not_applicable"

    answer_numbers = _extract_numbers(answer)
    quote_numbers = _extract_numbers(quote)
    missing_numbers = answer_numbers - quote_numbers
    if missing_numbers:
        return False, f"answer contains numbers not present in quote: {missing_numbers}"

    answer_months = _extract_months(answer)
    quote_months = _extract_months(quote)
    missing_months = answer_months - quote_months
    if missing_months:
        return False, f"answer references months not present in quote: {missing_months}"

    return True, None



def detect_hallucination(response, docs, ground_truth_category):
    """Runs all three layers against one system response.

    response: the dict returned by system.query() — must have ['parsed']
    docs: dict of {doc_id: text}, the full corpus (same one passed to the system)
    ground_truth_category: the 'category' field from the ground_truth.jsonl row
                            for this question (e.g. 'direct', 'unanswerable', ...)

    Returns None if no hallucination detected, otherwise a dict describing
    the failure type and layer, for use in failure clustering later.
    """
    parsed = response.get("parsed")
    if parsed is None:
        return {"layer": "output_format", "type": "malformed_output",
                "detail": response.get("error", "unparseable JSON")}

    answer = parsed.get("answer")
    quote = parsed.get("supporting_quote")
    doc_id = parsed.get("source_doc_id")

    if ground_truth_category == "unanswerable":
        if quote not in (None, "null", "") or doc_id not in (None, "null", ""):
            return {"layer": "layer3_unanswerable_cited", "type": "cited_on_unanswerable",
                    "detail": f"model cited doc_id={doc_id!r} quote={quote!r} "
                              f"for a question with no ground-truth source"}
        return None  # correctly reported unanswerable with no citation


    if quote is None or doc_id is None:
        return None

    # --- Layer 1: span existence ---
    if doc_id not in docs:
        return {"layer": "layer1_span_existence", "type": "invalid_doc_id",
                "detail": f"model cited nonexistent doc_id={doc_id!r}"}

    span_ok = check_span_exists(quote, docs[doc_id])
    if span_ok is False:
        return {"layer": "layer1_span_existence", "type": "fabricated_citation",
                "detail": f"quote not found in {doc_id}: {quote!r}"}

    # --- Layer 2: answer entailment ---
    entailment_ok, reason = check_entailment(answer, quote)
    if entailment_ok is False:
        return {"layer": "layer2_entailment", "type": "answer_not_entailed",
                "detail": reason}

    return None  # no hallucination detected by any layer


if __name__ == "__main__":
    fake_docs = {
        "doc_01_lease_main": (
            "The Licensee shall pay to the Licensor a monthly license fee "
            "of Rs. 4,50,000/- in advance before the 10th day of each "
            "calendar month."
        )
    }

   
    good_response = {"parsed": {
        "answer": "Rs. 4,50,000/- per month",
        "supporting_quote": "a monthly license fee of Rs. 4,50,000/-",
        "source_doc_id": "doc_01_lease_main",
        "confidence": 0.95,
    }}
    print("Case 1 (clean):", detect_hallucination(good_response, fake_docs, "direct"))

    bad_response = {"parsed": {
        "answer": "Rs. 5,00,000/- per month",
        "supporting_quote": "a monthly license fee of Rs. 5,00,000/-",
        "source_doc_id": "doc_01_lease_main",
        "confidence": 0.9,
    }}
    print("Case 2 (fabricated):", detect_hallucination(bad_response, fake_docs, "direct"))

    over_cited = {"parsed": {
        "answer": "30 days",
        "supporting_quote": "some text",
        "source_doc_id": "doc_01_lease_main",
        "confidence": 0.6,
    }}
    print("Case 3 (over-cited):", detect_hallucination(over_cited, fake_docs, "unanswerable"))

    correct_unanswerable = {"parsed": {
        "answer": "unanswerable",
        "supporting_quote": None,
        "source_doc_id": None,
        "confidence": 0.85,
    }}
    print("Case 4 (correct unanswerable):",
          detect_hallucination(correct_unanswerable, fake_docs, "unanswerable"))