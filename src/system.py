# system.py
# System under test: grounded QA over the fixed lease-document corpus.
# Model: Qwen/Qwen2.5-3B-Instruct, 4-bit quantized, greedy decoding for determinism.
#
# OPTIMIZATION NOTES (added after a hang was observed on the free Colab T4):
#   - Custom stopping criterion halts generation the moment a complete,
#     balanced JSON object has been produced, instead of always running to
#     max_new_tokens. Without this, a model that doesn't emit EOS cleanly
#     after JSON (common for smaller instruct models, especially with a
#     long document context) will generate garbage tokens all the way to
#     the cap on every single call — this was the most likely cause of the
#     observed multi-minute stall.
#   - max_new_tokens reduced 400 -> 220: the output schema rarely needs
#     more than ~150-200 tokens even for the longest compound-conditional
#     answers; this caps worst-case time per call even if the stopping
#     criterion doesn't trigger for some reason.
#   - attn_implementation="sdpa": PyTorch's built-in fused attention kernel,
#     meaningfully faster than the default "eager" backend on a T4, with no
#     change to output quality/correctness.
#
# COLAB SETUP (run this first, in its own cell):
#   !pip install transformers accelerate bitsandbytes -q
#   Runtime -> Change runtime type -> T4 GPU (must be selected before loading the model)

import glob
import json
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)

CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "model_revision": None,  # TODO: once loaded once, pin the exact commit hash here
    "seed": 42,
    "max_new_tokens": 220,   # reduced from 400 — see optimization notes above
    "quantization": "4bit_nf4",
    "attn_implementation": "sdpa",
}

set_seed(CONFIG["seed"])

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading {CONFIG['model_name']} ...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    quantization_config=_bnb_config,
    device_map="auto",
    attn_implementation=CONFIG["attn_implementation"],
)
model.eval()
print("Model loaded.")


def load_corpus(corpus_dir="corpus"):
    """Loads all .md files in corpus_dir into a dict of {doc_id: text}."""
    docs = {}
    for path in sorted(glob.glob(f"{corpus_dir}/*.md")):
        doc_id = path.split("/")[-1].replace(".md", "")
        with open(path, "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    return docs

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a careful legal-document question-answering assistant. "
    "You answer ONLY using the documents provided. "
    "You must respond with ONLY a single valid JSON object and nothing else — "
    "no preamble, no explanation, no markdown code fences. "
    "Keep your answer field concise. Stop generating immediately after the "
    "closing brace of the JSON object."
)

PROMPT_TEMPLATE = """{documents}

Question: {question}

Respond with ONLY this JSON structure, filled in, then STOP:
{{
  "answer": "<your answer in plain text, or 'unanswerable' if the documents do not state this>",
  "supporting_quote": "<a verbatim quote copied exactly from one of the documents above that supports your answer, or null if unanswerable>",
  "source_doc_id": "<the doc id the quote came from, e.g. doc_01_lease_main, or null if unanswerable>",
  "confidence": <a number between 0 and 1 representing how confident you are>
}}"""


def build_messages(docs, question, excluded_doc_id=None):
    """Builds the chat messages for one query.
    excluded_doc_id: optionally drop one document from context, used for
    the distribution-edge / simulated-retrieval-miss adversarial test.
    """
    doc_block = "\n\n".join(
        f'<document id="{doc_id}">\n{text}\n</document>'
        for doc_id, text in docs.items()
        if doc_id != excluded_doc_id
    )
    user_content = PROMPT_TEMPLATE.format(documents=doc_block, question=question)
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Custom stopping criterion: halt as soon as a complete, brace-balanced
# JSON object has been generated, instead of always running to the token cap.
# ---------------------------------------------------------------------------
class JSONCompleteStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, check_every=4):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_every = check_every  # only decode/check every N tokens — decoding every step is wasteful
        self._step = 0

    def __call__(self, input_ids, scores, **kwargs):
        self._step += 1
        if self._step % self.check_every != 0:
            return False
        text = self.tokenizer.decode(input_ids[0][self.prompt_len:], skip_special_tokens=True)
        stripped = text.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return False
        # Cheap brace-balance check. Not a full JSON parse (too slow to run
        # every few tokens), but sufficient to detect "a complete-looking
        # object has been closed" and stop generating filler after it.
        if stripped.count("{") == stripped.count("}") and stripped.count("{") > 0:
            return True
        return False


# ---------------------------------------------------------------------------
# Generation (greedy, deterministic)
# ---------------------------------------------------------------------------
def _generate(messages):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList(
        [JSONCompleteStoppingCriteria(tokenizer, prompt_len)]
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,       # greedy decoding — the local-inference
            temperature=None,      # equivalent of API temperature=0
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )

    new_tokens = output_ids[0][prompt_len:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_text.strip()


def _extract_json(raw_text):
    """Qwen sometimes wraps JSON in markdown fences or adds stray text.
    Strip fences, then find the first {...} block and parse it."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def query(question, docs, excluded_doc_id=None, max_retries=2):
    """Runs one question through the system. Returns a dict with raw output,
    parsed JSON (or None on failure), and metadata."""
    messages = build_messages(docs, question, excluded_doc_id=excluded_doc_id)

    for attempt in range(max_retries + 1):
        raw = _generate(messages)
        parsed = _extract_json(raw)
        if parsed is not None:
            return {
                "question": question,
                "raw": raw,
                "parsed": parsed,
                "retries_needed": attempt,
                "excluded_doc_id": excluded_doc_id,
            }
        # Retry: append a corrective instruction and try again
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "That was not valid JSON. Respond with ONLY the JSON object, nothing else."},
        ]

    return {
        "question": question,
        "raw": raw,
        "parsed": None,
        "retries_needed": max_retries,
        "excluded_doc_id": excluded_doc_id,
        "error": "invalid_json_after_retries",
    }


# ---------------------------------------------------------------------------
# Quick smoke test — run this cell after loading, before building anything else
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    docs = load_corpus()
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}")

    test_question = "What is the monthly license fee under the Agreement?"
    t0 = time.time()
    result = query(test_question, docs)
    print(f"Smoke test completed in {time.time() - t0:.1f}s")
    print(json.dumps(result, indent=2))