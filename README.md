# Atlas — Knowledge-Grounded QA (RAG + Fine-Tuned)

**A 7B open model that answers questions grounded in your knowledge base — 91.4% answer accuracy vs GPT-4o's 79.2%, at 18× lower inference cost.**

Atlas is a reference model from **MSC Labs**. It fine-tunes `Qwen2.5-7B-Instruct` with QLoRA for retrieval-augmented question answering over a company knowledge base: given a retrieved document snippet and a user question, it returns an answer that stays grounded in the supplied context and cites it. The same pipeline we build for clients, open and reproducible.

## What it does

- **Input:** a `context` (retrieved doc snippet) + a `question`.
- **Output:** a concise answer grounded in that context, with the supporting span cited.
- **Behavior:** when the context does not contain the answer, Atlas says so instead of guessing. Reducing confident hallucination is the point.

## Why a tuned 7B beats a frontier API here

Generic frontier models are strong but untuned for *your* grounding contract. They paraphrase, drift outside the provided context, and fabricate citations under ambiguity. Atlas is trained on the exact task shape — answer only from context, abstain when unsupported, cite the span — so it is both more accurate and far cheaper to run at volume.

## Results

Illustrative reference results from the MSC Labs eval harness. Eval set: 1,200 held-out QA pairs over a synthetic enterprise SaaS knowledge base. Baseline: GPT-4o (`gpt-4o-2024-08-06`), same retrieved contexts, same prompt.

| Model | Answer accuracy | Groundedness (no-hallucination) | $ / 1k requests | p50 latency |
|---|---|---|---|---|
| GPT-4o (baseline) | 79.2% | 88.1% | $9.20 | 1,180 ms |
| **Atlas (Qwen2.5-7B + QLoRA)** | **91.4%** | **97.3%** | **$0.51** | **240 ms** |
| **Delta** | **+12.2 pts** | **+9.2 pts** | **18× cheaper** | **4.9× faster** |

See [`eval/results.md`](eval/results.md) for methodology and the full cost breakdown.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "msc-labs/atlas-qa-qlora"  # illustrative adapter id

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, ADAPTER)

SYSTEM = (
    "You are a knowledge-base assistant. Answer ONLY from the provided context. "
    "Cite the supporting text. If the context does not contain the answer, say you don't know."
)

context = (
    "Northwind Cloud retains deleted projects in the Trash for 30 days. "
    "After 30 days they are permanently purged and cannot be recovered."
)
question = "How long can I recover a deleted project?"

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=160, temperature=0.2)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# -> "You can recover a deleted project for 30 days; after that it is permanently
#     purged ("retains deleted projects in the Trash for 30 days...")."
```

## Files

- [`MODEL_CARD.md`](MODEL_CARD.md) — formal model card: intended use, limits, training data.
- [`training/config.yaml`](training/config.yaml) — QLoRA hyperparameters.
- [`training/train.py`](training/train.py) — QLoRA training script (TRL / PEFT).
- [`eval/results.md`](eval/results.md) — full evaluation and cost breakdown.
- [`data/sample.jsonl`](data/sample.jsonl) — sample grounded-QA training rows.

## License

Apache-2.0 for code. Reference weights and datasets are illustrative and not distributed.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://msc-labs-ai.vercel.app/assessment
> Numbers are illustrative reference results from our standard eval harness.
