

import glob
import json
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)


CONFIG = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "model_revision": None,  # TODO: once loaded once, pin the exact commit hash here
    "seed": 42,
    "max_new_tokens": 400,
    "quantization": "4bit_nf4",
}

set_seed(CONFIG["seed"])


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


SYSTEM_INSTRUCTION = (
    "You are a careful legal-document question-answering assistant. "
    "You answer ONLY using the documents provided. "
    "You must respond with ONLY a single valid JSON object and nothing else — "
    "no preamble, no explanation, no markdown code fences."
)

PROMPT_TEMPLATE = """{documents}

Question: {question}

Respond with ONLY this JSON structure, filled in:
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



def _generate(messages):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=False,       # greedy decoding — the local-inference
            temperature=None,      # equivalent of API temperature=0
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
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

if __name__ == "__main__":
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}")

    test_question = "What is the monthly license fee under the Agreement?"
    result = query(test_question, docs)
    print(json.dumps(result, indent=2))